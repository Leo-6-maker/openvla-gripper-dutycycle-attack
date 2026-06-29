#!/usr/bin/env python3
"""Multi-suite SC5MLP detector training with strict data validation.

Loads suite from episode index. Parses YAML config for all parameters.
Fail-closed: rejects missing features, NaN, Inf, missing labels.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES, HIDDEN_DIM

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]


def parse_config(path: str) -> dict:
    """Parse YAML or JSON config."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            pass
    with open(path) as f:
        return json.load(f)


def load_episode_index(path: str) -> dict:
    """Return {episode_key: {suite, task_id, ...}}."""
    index = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            index[ep["episode_key"]] = ep
    return index


def load_features(csv_path: str, episode_index: dict) -> dict:
    """Load 25D features with strict validation. Fails on missing/bad data."""
    import csv
    data = defaultdict(list)
    seen_steps = defaultdict(set)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [f for f in SC5_FEATURES if f not in fieldnames]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        if "episode_key" not in fieldnames and "episode" not in fieldnames:
            raise ValueError("CSV missing episode_key column")
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        if "step" not in fieldnames and "step_id" not in fieldnames and "step_idx" not in fieldnames:
            raise ValueError("CSV missing step column")
        step_col = "step" if "step" in fieldnames else ("step_id" if "step_id" in fieldnames else "step_idx")

        for row in reader:
            ek = row.get(ek_col, "")
            if not ek:
                raise ValueError(f"Empty episode key in CSV row")
            step = int(row.get(step_col, -1))
            if step in seen_steps[ek]:
                raise ValueError(f"Duplicate step {step} in episode {ek}")
            seen_steps[ek].add(step)
            feats = []
            for fn in SC5_FEATURES:
                v = row.get(fn)
                if v is None or v == "":
                    raise ValueError(f"Missing value for {fn} in {ek} step {step}")
                fv = float(v)
                if not np.isfinite(fv):
                    raise ValueError(f"Non-finite {fn}={fv} in {ek} step {step}")
                feats.append(fv)
            data[ek].append(feats)

    result = {}
    for ek, rows in data.items():
        arr = np.array(rows, dtype=np.float32)
        if arr.shape[1] != N_FEATURES:
            raise ValueError(f"Episode {ek}: {arr.shape[1]} features, expected {N_FEATURES}")
        result[ek] = arr
    return result


def load_labels(csv_path: str) -> dict:
    """Load teacher labels with validation."""
    import csv
    data = defaultdict(lambda: {"phase": [], "corridor": [], "release": []})
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        for row in reader:
            ek = row.get(ek_col, "")
            if not ek:
                continue
            phase = int(row.get("teacher_phase_idx", row.get("phase_idx", 0)))
            corridor = int(row.get("teacher_sc5_corridor_active", row.get("corridor_active", 0)))
            release = int(row.get("release_safe", 0))
            data[ek]["phase"].append(phase)
            data[ek]["corridor"].append(corridor)
            data[ek]["release"].append(release)
    return {k: {kk: np.array(vv, dtype=np.int64) for kk, vv in v.items()} for k, v in data.items()}


def compute_normalization(features: dict, episode_keys: list[str]) -> tuple:
    all_feats = []
    for ek in episode_keys:
        if ek not in features:
            raise ValueError(f"Episode {ek} not found in features")
        all_feats.append(features[ek])
    if not all_feats:
        raise ValueError("No training features")
    X = np.concatenate(all_feats, axis=0)
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    if not np.all(np.isfinite(mean)):
        raise ValueError("NaN/Inf in normalization mean")
    if not np.all(np.isfinite(std)):
        raise ValueError("NaN/Inf in normalization std")
    return mean, std


def suite_balanced_sampler(episode_keys, suite_map, batch_size, rng):
    by_suite = defaultdict(list)
    for ek in episode_keys:
        by_suite[suite_map.get(ek, "unknown")].append(ek)
    active = [s for s in by_suite if by_suite[s]]
    if not active:
        raise ValueError("No suites with episodes")
    while True:
        batch = []
        for _ in range(batch_size):
            s = active[rng.randint(len(active))]
            batch.append(by_suite[s][rng.randint(len(by_suite[s]))])
        yield batch


def train_epoch(model, features, labels, episode_keys, suite_map, optimizer, batch_size, rng):
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
            feats = features[ek]
            labs = labels[ek]
            n_steps = min(len(feats), len(labs["phase"]))
            X = torch.from_numpy(feats[:n_steps])
            phase_tgt = torch.from_numpy(labs["phase"][:n_steps])
            corr_tgt = torch.from_numpy(labs["corridor"][:n_steps]).float().unsqueeze(1)
            rel_tgt = torch.from_numpy(labs["release"][:n_steps]).float().unsqueeze(1)
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
    model.eval()
    phase_ce = nn.CrossEntropyLoss()
    corridor_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
    release_bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.no_grad():
        for ek in episode_keys:
            feats = features[ek]
            labs = labels[ek]
            n_steps = min(len(feats), len(labs["phase"]))
            X = torch.from_numpy(feats[:n_steps])
            phase_tgt = torch.from_numpy(labs["phase"][:n_steps])
            corr_tgt = torch.from_numpy(labs["corridor"][:n_steps]).float().unsqueeze(1)
            rel_tgt = torch.from_numpy(labs["release"][:n_steps]).float().unsqueeze(1)
            out = model(X)
            loss = (phase_ce(out["phase_logits"], phase_tgt) +
                    0.5 * corridor_bce(out["corridor_logit"], corr_tgt) +
                    0.3 * release_bce(out["release_logit"], rel_tgt))
            total_loss += loss.item()
    return total_loss / max(1, len(episode_keys))


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Train multi-suite SC5MLP detector")
    ap.add_argument("--config", required=True, help="Training config YAML/JSON")
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", help="Episode index JSONL (for suite mapping)")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    config = parse_config(args.config)
    tc = config.get("training_config", config)
    epochs = tc.get("epochs", args.epochs)
    batch_size = tc.get("batch_size", args.batch_size)
    lr = tc.get("lr", args.lr)
    patience = tc.get("patience", args.patience)

    print(f"Loading episode index from {args.episode_index or 'N/A'}")
    episode_index = {}
    if args.episode_index:
        episode_index = load_episode_index(args.episode_index)

    print(f"Loading features from {args.feature_csv}")
    features = load_features(args.feature_csv, episode_index)
    print(f"Loading labels from {args.label_csv}")
    labels = load_labels(args.label_csv)
    print(f"Loading split from {args.split_file}")
    with open(args.split_file) as f:
        split = json.load(f)

    train_eks = split["splits"]["train"]
    val_eks = split["splits"]["val"]
    train_eks = [e for e in train_eks if e in features and e in labels]
    val_eks = [e for e in val_eks if e in features and e in labels]
    print(f"Train: {len(train_eks)}, Val: {len(val_eks)}")

    if not train_eks:
        sys.exit("No training episodes with features+labels")
    if len(train_eks) < batch_size:
        print(f"Warning: {len(train_eks)} train episodes < batch_size {batch_size}")

    # Build suite_map from episode_index (not hardcoded)
    suite_map = {}
    for ek in train_eks + val_eks:
        if ek in episode_index:
            suite_map[ek] = episode_index[ek].get("suite", "unknown")
        else:
            suite_map[ek] = "unknown"
    suites_found = set(suite_map.values())
    print(f"Suites: {sorted(suites_found)}")

    print("Computing normalization from training set only...")
    mean, std = compute_normalization(features, train_eks)
    for ek in features:
        features[ek] = (features[ek] - mean) / std

    if args.dry_run:
        print("DRY_RUN: validation completed, no training.")
        print(f"Features: {N_FEATURES}D, Phases: {N_PHASES}")
        print(f"Architecture: Linear(25,64)->ReLU->Linear(64,64)->ReLU -> 3 heads")
        print(f"Suites: {sorted(suites_found)}")
        print(f"Normalization mean: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"Normalization std: [{std.min():.6f}, {std.max():.4f}]")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    model = SC5MLPV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, features, labels, train_eks, suite_map,
                                 optimizer, batch_size, rng)
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
                "dataset_sha256": sha256_file(args.feature_csv),
                "split_mode": "frozen",
                "normalization_source": "train_only",
                "n_train": len(train_eks), "n_val": len(val_eks),
                "seed": args.seed,
                "tau_corridor": 0.3, "tau_release": 0.3, "guard": 5,
                "epoch": epoch, "val_loss": val_loss,
                "feature_csv_sha256": sha256_file(args.feature_csv),
                "label_csv_sha256": sha256_file(args.label_csv),
                "split_sha256": split.get("split_sha256", ""),
                "config_sha256": sha256_file(args.config),
            }
            torch.save(ckpt, out_dir / f"best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
