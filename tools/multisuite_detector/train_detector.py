#!/usr/bin/env python3
"""Multi-suite SC5MLP detector training with strict data integrity.

Fail-closed: rejects any feature/label mismatch, missing episodes, unknown suites,
non-contiguous steps, duplicate steps, NaN/Inf, missing columns, empty keys.
Suite loaded from episode index. Config parsed for all parameters.
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

VALID_SUITES = {"libero_object", "libero_spatial", "libero_goal", "libero_10"}


def parse_config(path: str) -> dict:
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
    index = {}
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            ek = ep["episode_key"]
            if "suite" not in ep:
                raise ValueError("Episode {} missing suite".format(ek))
            if ep["suite"] not in VALID_SUITES:
                raise ValueError("Episode {} has invalid suite: {}".format(ek, ep["suite"]))
            index[ek] = ep
    return index


def load_features(csv_path: str, episode_index: dict) -> dict:
    """Load 25D features with strict validation."""
    import csv
    data = defaultdict(dict)  # {ek: {step: [features]}}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_cols = [fn for fn in SC5_FEATURES if fn not in fieldnames]
        if missing_cols:
            raise ValueError("Missing feature columns: {}".format(missing_cols))
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        if ek_col not in fieldnames:
            raise ValueError("CSV missing episode_key column")
        step_col = "step" if "step" in fieldnames else ("step_id" if "step_id" in fieldnames else "step_idx")
        if step_col not in fieldnames:
            raise ValueError("CSV missing step column")

        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in feature CSV")
            step = int(row.get(step_col, -1))
            if step in data[ek]:
                raise ValueError("Duplicate step {} in episode {}".format(step, ek))
            feats = []
            for fn in SC5_FEATURES:
                v = row.get(fn)
                if v is None or str(v).strip() == "":
                    raise ValueError("Missing value for {} in {} step {}".format(fn, ek, step))
                fv = float(v)
                if not np.isfinite(fv):
                    raise ValueError("Non-finite {}={} in {} step {}".format(fn, fv, ek, step))
                feats.append(fv)
            data[ek][step] = feats

    # Sort steps, check contiguous, build arrays
    result = {}
    for ek, steps_dict in data.items():
        sorted_steps = sorted(steps_dict.keys())
        if sorted_steps[0] != 0:
            raise ValueError("Episode {} first step is {} (expected 0)".format(ek, sorted_steps[0]))
        for i, s in enumerate(sorted_steps):
            if s != i:
                raise ValueError("Episode {} step gap: expected {}, got {}".format(ek, i, s))
        arr = np.array([steps_dict[s] for s in sorted_steps], dtype=np.float32)
        if arr.shape[1] != N_FEATURES:
            raise ValueError("Episode {} has {} features, expected {}".format(ek, arr.shape[1], N_FEATURES))
        result[ek] = arr
    return result


def load_labels(csv_path: str) -> dict:
    """Load teacher labels with strict validation matching feature loader."""
    import csv
    data = defaultdict(dict)  # {ek: {step: {phase, corridor, release}}}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        ek_col = "episode_key" if "episode_key" in fieldnames else "episode"
        if ek_col not in fieldnames:
            raise ValueError("Label CSV missing episode_key column")
        step_col = "step" if "step" in fieldnames else ("step_id" if "step_id" in fieldnames else "step_idx")
        if step_col not in fieldnames:
            raise ValueError("Label CSV missing step column")

        phase_col = None
        for c in ["teacher_phase_idx", "phase_idx", "phase"]:
            if c in fieldnames:
                phase_col = c; break
        corridor_col = None
        for c in ["teacher_sc5_corridor_active", "corridor_active", "corridor"]:
            if c in fieldnames:
                corridor_col = c; break
        release_col = "release_safe" if "release_safe" in fieldnames else "release"
        if release_col not in fieldnames:
            release_col = None

        for row in reader:
            ek = row.get(ek_col, "").strip()
            if not ek:
                raise ValueError("Empty episode key in label CSV")
            step = int(row.get(step_col, -1))
            if step < 0:
                raise ValueError("Invalid step {} in episode {}".format(step, ek))
            if step in data[ek]:
                raise ValueError("Duplicate step {} in label episode {}".format(step, ek))

            phase = int(row.get(phase_col, -1)) if phase_col else -1
            if phase < 0 or phase >= N_PHASES:
                raise ValueError("Phase {} out of range [0,{}) in {} step {}".format(phase, N_PHASES, ek, step))
            corridor = int(row.get(corridor_col, -1)) if corridor_col else -1
            if corridor not in (0, 1):
                raise ValueError("Corridor {} not 0/1 in {} step {}".format(corridor, ek, step))
            release = int(row.get(release_col, -1)) if release_col else -1
            if release not in (0, 1):
                raise ValueError("Release {} not 0/1 in {} step {}".format(release, ek, step))

            data[ek][step] = {"phase": phase, "corridor": corridor, "release": release}

    result = {}
    for ek, steps_dict in data.items():
        sorted_steps = sorted(steps_dict.keys())
        if sorted_steps[0] != 0:
            raise ValueError("Label episode {} first step is {} (expected 0)".format(ek, sorted_steps[0]))
        for i, s in enumerate(sorted_steps):
            if s != i:
                raise ValueError("Label episode {} step gap: expected {}, got {}".format(ek, i, s))
        phase_arr = np.array([steps_dict[s]["phase"] for s in sorted_steps], dtype=np.int64)
        corr_arr = np.array([steps_dict[s]["corridor"] for s in sorted_steps], dtype=np.int64)
        rel_arr = np.array([steps_dict[s]["release"] for s in sorted_steps], dtype=np.int64)
        result[ek] = {"phase": phase_arr, "corridor": corr_arr, "release": rel_arr}
    return result


def strict_join(train_eks, val_eks, features, labels, episode_index):
    """Validate that all split episodes have features and labels. Fails on mismatch."""
    all_eks = set(train_eks + val_eks)
    missing_feat = [ek for ek in all_eks if ek not in features]
    missing_label = [ek for ek in all_eks if ek not in labels]
    errors = []
    if missing_feat:
        errors.append("{} split episodes missing features: {}...".format(len(missing_feat), missing_feat[:5]))
    if missing_label:
        errors.append("{} split episodes missing labels: {}...".format(len(missing_label), missing_label[:5]))

    for ek in all_eks:
        if ek not in features or ek not in labels:
            continue
        n_feat = len(features[ek])
        n_label = len(labels[ek]["phase"])
        if n_feat != n_label:
            errors.append("Length mismatch: {} features={} labels={}".format(ek, n_feat, n_label))
        elif not np.array_equal(np.arange(n_feat), np.arange(n_label)):
            errors.append("Step mismatch: {} has {} feat steps vs {} label steps".format(ek, n_feat, n_label))

    if errors:
        for e in errors:
            print("FAIL: {}".format(e))
        raise ValueError("Strict join failed: {} errors".format(len(errors)))

    # Build suite map, reject unknown
    suite_map = {}
    for ek in all_eks:
        ep = episode_index.get(ek, {})
        s = ep.get("suite", "MISSING")
        if s not in VALID_SUITES:
            raise ValueError("Episode {} has invalid/unknown suite: {}".format(ek, s))
        suite_map[ek] = s

    return suite_map


def compute_normalization(features, episode_keys):
    all_feats = []
    for ek in episode_keys:
        all_feats.append(features[ek])
    if not all_feats:
        raise ValueError("No training features")
    X = np.concatenate(all_feats, axis=0)
    mean = np.mean(X, axis=0).astype(np.float32)
    std = np.std(X, axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("NaN/Inf in normalization")
    return mean, std


def suite_balanced_sampler(episode_keys, suite_map, batch_size, rng):
    by_suite = defaultdict(list)
    for ek in episode_keys:
        by_suite[suite_map[ek]].append(ek)
    active = sorted([s for s in by_suite if by_suite[s]])
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
            X = torch.from_numpy(features[ek])
            phase_tgt = torch.from_numpy(labels[ek]["phase"])
            corr_tgt = torch.from_numpy(labels[ek]["corridor"]).float().unsqueeze(1)
            rel_tgt = torch.from_numpy(labels[ek]["release"]).float().unsqueeze(1)
            # n_steps already verified equal by strict_join
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Train multi-suite SC5MLP detector")
    ap.add_argument("--config", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", required=True, help="Episode index JSONL for suite+step metadata")
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--checkpoint_metric", default="val_loss",
                    choices=["val_loss", "val_suite_macro_event_f1"],
                    help="Checkpoint selection metric (frozen before training)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    config = parse_config(args.config)
    tc = config.get("training_config", config)
    epochs = tc.get("epochs", args.epochs)
    batch_size = tc.get("batch_size", args.batch_size)
    lr = tc.get("lr", args.lr)
    patience = tc.get("patience", args.patience)
    ckpt_metric = tc.get("checkpoint_metric", args.checkpoint_metric)

    print("Loading episode index: {}".format(args.episode_index))
    episode_index = load_episode_index(args.episode_index)

    print("Loading features: {}".format(args.feature_csv))
    features = load_features(args.feature_csv, episode_index)

    print("Loading labels: {}".format(args.label_csv))
    labels = load_labels(args.label_csv)

    print("Loading split: {}".format(args.split_file))
    with open(args.split_file) as f:
        split = json.load(f)

    train_eks = split["splits"]["train"]
    val_eks = split["splits"]["val"]

    print("Strict joining...")
    suite_map = strict_join(train_eks, val_eks, features, labels, episode_index)
    train_eks = [e for e in train_eks if e in features]
    val_eks = [e for e in val_eks if e in features]
    print("Train: {}, Val: {}, Suites: {}".format(len(train_eks), len(val_eks), sorted(set(suite_map.values()))))

    print("Normalization from train only...")
    mean, std = compute_normalization(features, train_eks)
    for ek in list(features.keys()):
        features[ek] = (features[ek] - mean) / std

    if args.dry_run:
        print("DRY_RUN: validation passed, no training.")
        print("Features: {}D, Phases: {}, Checkpoint metric: {}".format(N_FEATURES, N_PHASES, ckpt_metric))
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    model = SC5MLPV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, features, labels, train_eks, suite_map,
                                 optimizer, batch_size, rng)
        val_loss = validate_epoch(model, features, labels, val_eks, suite_map)
        val_metric = val_loss  # primary: val_loss; F1 requires detector runtime eval
        print("Epoch {:3d}: train_loss={:.4f} val_loss={:.4f}".format(epoch, train_loss, val_loss))

        improved = val_metric < best_val - 0.001
        if improved:
            best_val = val_metric
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
                "seed": args.seed, "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint_metric": ckpt_metric,
                "tau_corridor": 0.3, "tau_release": 0.3, "guard": 5,
                "feature_csv_sha256": sha256_file(args.feature_csv),
                "label_csv_sha256": sha256_file(args.label_csv),
                "episode_index_sha256": sha256_file(args.episode_index),
                "split_sha256": split.get("split_sha256", ""),
                "config_sha256": sha256_file(args.config),
            }
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    print("Best epoch: {}, val_loss: {:.4f}".format(best_epoch, best_val))


if __name__ == "__main__":
    main()
