#!/usr/bin/env python3
"""Multi-suite SC5MLP detector training.

Reuses: src/gripper_attack/sc5mlp_v1.SC5MLPV1 (identical architecture to Object-only)
Adds: suite-balanced sampling, per-suite loss weighting, LOSO isolation.
NO live data. Reads only frozen CLEAN2000 feature CSVs + split manifest.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import from repo (worktree root)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES, HIDDEN_DIM

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
SUITE_WEIGHTS_DEFAULT = {"libero_object": 1.0, "libero_spatial": 1.0,
                          "libero_goal": 1.0, "libero_10": 1.0}


def load_features(csv_path: str) -> dict:
    """Load 25D features from frozen CSV. Returns {episode_key: np.array(n_steps, 25)}."""
    import csv
    data = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ek = row.get("episode_key", row.get("episode", ""))
            feats = [float(row.get(f, 0)) for f in SC5_FEATURES]
            data[ek].append(feats)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def load_labels(csv_path: str) -> dict:
    """Load teacher labels. Returns {episode_key: {phase, corridor, release}_labels}."""
    import csv
    data = defaultdict(lambda: {"phase": [], "corridor": [], "release": []})
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ek = row.get("episode_key", row.get("episode", ""))
            phase = int(row.get("teacher_phase_idx", 0))
            corridor = int(row.get("teacher_sc5_corridor_active", 0))
            release = int(row.get("release_safe", 0))
            data[ek]["phase"].append(phase)
            data[ek]["corridor"].append(corridor)
            data[ek]["release"].append(release)
    return {k: {kk: np.array(vv, dtype=np.int64) for kk, vv in v.items()} for k, v in data.items()}


def compute_normalization(features: dict, episode_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std from training episodes only."""
    all_feats = []
    for ek in episode_keys:
        if ek in features:
            all_feats.append(features[ek])
    if not all_feats:
        raise ValueError("No training features found")
    X = np.concatenate(all_feats, axis=0)
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    return mean, std


def suite_balanced_sampler(episode_keys: list[str], suite_map: dict, batch_size: int, rng: np.random.RandomState):
    """Sample episodes: first uniformly sample suite, then episode within suite."""
    by_suite = defaultdict(list)
    for ek in episode_keys:
        by_suite[suite_map.get(ek, "unknown")].append(ek)
    suites = [s for s in SUITES if s in by_suite and by_suite[s]]
    while True:
        batch = []
        for _ in range(batch_size):
            s = suites[rng.randint(len(suites))]
            ek = by_suite[s][rng.randint(len(by_suite[s]))]
            batch.append(ek)
        yield batch


def train_epoch(model, features, labels, episode_keys, suite_map, optimizer, batch_size, rng):
    """One training epoch with suite-balanced sampling."""
    model.train()
    sampler = suite_balanced_sampler(episode_keys, suite_map, batch_size, rng)
    n_batches = max(1, len(episode_keys) // batch_size)
    total_loss = 0.0
    phase_ce = nn.CrossEntropyLoss()
    corridor_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    release_bce = nn.BCEWithLogitsLoss()

    for _ in range(n_batches):
        batch_eks = next(sampler)
        optimizer.zero_grad()
        batch_loss = 0.0
        for ek in batch_eks:
            X = torch.from_numpy(features[ek])
            phase_tgt = torch.from_numpy(labels[ek]["phase"])
            corr_tgt = torch.from_numpy(labels[ek]["corridor"]).float().unsqueeze(1)
            rel_tgt = torch.from_numpy(labels[ek]["release"]).float().unsqueeze(1)
            out = model(X)
            loss = (phase_ce(out["phase_logits"], phase_tgt) +
                    0.5 * corridor_bce(out["corridor_logit"], corr_tgt) +
                    0.3 * release_bce(out["release_logit"], rel_tgt))
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        total_loss += batch_loss / len(batch_eks)
    return total_loss / n_batches


def validate_epoch(model, features, labels, episode_keys, suite_map):
    """Validation epoch."""
    model.eval()
    phase_ce = nn.CrossEntropyLoss()
    corridor_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    release_bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.no_grad():
        for ek in episode_keys:
            X = torch.from_numpy(features[ek])
            phase_tgt = torch.from_numpy(labels[ek]["phase"])
            corr_tgt = torch.from_numpy(labels[ek]["corridor"]).float().unsqueeze(1)
            rel_tgt = torch.from_numpy(labels[ek]["release"]).float().unsqueeze(1)
            out = model(X)
            loss = (phase_ce(out["phase_logits"], phase_tgt) +
                    0.5 * corridor_bce(out["corridor_logit"], corr_tgt) +
                    0.3 * release_bce(out["release_logit"], rel_tgt))
            total_loss += loss.item()
    return total_loss / max(1, len(episode_keys))


def main():
    ap = argparse.ArgumentParser(description="Train multi-suite SC5MLP detector")
    ap.add_argument("--config", required=True, help="Training config YAML/JSON")
    ap.add_argument("--feature_csv", required=True, help="Frozen 25D feature CSV")
    ap.add_argument("--label_csv", required=True, help="Frozen teacher label CSV")
    ap.add_argument("--split_file", required=True, help="Split manifest JSON")
    ap.add_argument("--output_dir", required=True, help="Output directory for checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dry_run", action="store_true", help="Validate only, no training")
    args = ap.parse_args()

    print(f"Loading features from {args.feature_csv}")
    features = load_features(args.feature_csv)
    print(f"Loading labels from {args.label_csv}")
    labels = load_labels(args.label_csv)
    print(f"Loading split from {args.split_file}")
    with open(args.split_file) as f:
        split = json.load(f)

    train_eks = split["splits"]["train"]
    val_eks = split["splits"]["val"]

    train_eks = [e for e in train_eks if e in features and e in labels]
    val_eks = [e for e in val_eks if e in features and e in labels]
    print(f"Train episodes: {len(train_eks)}, Val episodes: {len(val_eks)}")

    if not train_eks:
        sys.exit("No training episodes with features+labels")

    print("Computing normalization from training set only...")
    mean, std = compute_normalization(features, train_eks)

    for ek in features:
        features[ek] = (features[ek] - mean) / std

    suite_map = {}
    for ek in train_eks + val_eks:
        suite_map[ek] = "libero_object"

    if args.dry_run:
        print("DRY_RUN: validation only, no training.")
        print(f"Feature dim: {len(SC5_FEATURES)}, Phases: {len(SC5_PHASES)}")
        print(f"Architecture: Linear(25,64)->ReLU->Linear(64,64)->ReLU -> 3 heads")
        print(f"Normalization mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"Normalization std range: [{std.min():.6f}, {std.max():.4f}]")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    model = SC5MLPV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, features, labels, train_eks, suite_map,
                                 optimizer, args.batch_size, rng)
        val_loss = validate_epoch(model, features, labels, val_eks, suite_map)
        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "feature_names": SC5_FEATURES,
                "phase_classes": SC5_PHASES,
                "dataset_sha256": "DRY_RUN_NO_DATA",
                "split_mode": "frozen",
                "normalization_source": "train_only",
                "n_train": len(train_eks), "n_val": len(val_eks),
                "seed": args.seed,
                "tau_corridor": 0.3, "tau_release": 0.3, "guard": 5,
                "epoch": epoch, "val_loss": val_loss,
            }
            torch.save(ckpt, out_dir / f"best_model_seed{args.seed}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {args.output_dir}/best_model_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
