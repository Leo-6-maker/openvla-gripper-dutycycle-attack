#!/usr/bin/env python3
"""Multi-suite SC5MLP detector training with strict data integrity.

Uses shared strict_loader for feature/label/index loading.
Checkpoint selection: val_loss (default) or val_suite_macro_event_f1 (via detector replay).
Fail-closed on F1 if detector runtime unavailable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strict_loader import (
    load_episode_index, load_features, load_labels, strict_join,
    compute_normalization, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES, VALID_SUITES,
)
from gripper_attack.sc5mlp_v1 import SC5MLPV1


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


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_git_info(repo: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        return {"commit": commit, "dirty": bool(dirty), "dirty_files": len(dirty.split(chr(10))) if dirty else 0}
    except Exception:
        return {"commit": "UNAVAILABLE", "dirty": None, "dirty_files": 0}


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


def compute_val_event_f1(checkpoint_path, features, labels, episode_keys, suite_map, fsm_version, tau_c, tau_r, guard):
    """Compute validation suite-macro event F1 using detector runtime replay."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R
    except ImportError:
        raise RuntimeError("F1 checkpoint metric requires detector runtime (src/gripper_attack/sc5_detector_runtime_v1r.py)")

    detector = SC5DetectorRuntimeV1R(checkpoint_path, tau_corridor=tau_c, tau_release=tau_r, guard=guard, fsm_version=fsm_version)

    by_suite_results = defaultdict(list)
    for ek in episode_keys:
        detector.reset()
        feats = features[ek]
        labs = labels[ek]
        n = min(len(feats), len(labs["phase"]))
        emitted = False
        emit_step = -1
        for step in range(n):
            d = detector.update(feats[step], step)
            if d.get("emitted"):
                emitted = True
                emit_step = d.get("emit_step", -1)
                break
        # Determine if episode has a teacher event (any corridor_active=1 step)
        has_teacher = int((labs["corridor"][:n] == 1).any())
        # Determine if emit is within teacher corridor window
        in_window = False
        if emitted and has_teacher:
            corridor_steps = np.where(labs["corridor"][:n] == 1)[0]
            if len(corridor_steps) > 0:
                wstart = corridor_steps[0]
                wend = corridor_steps[-1]
                in_window = wstart <= emit_step <= wend
        by_suite_results[suite_map[ek]].append({
            "emitted": emitted, "in_window": in_window,
            "has_teacher": has_teacher,
        })

    per_suite_f1 = {}
    for s in sorted(by_suite_results):
        rs = by_suite_results[s]
        tp = sum(1 for r in rs if r["emitted"] and r["in_window"])
        fp = sum(1 for r in rs if r["emitted"] and not r["in_window"])
        fn = sum(1 for r in rs if not r["emitted"] and r["has_teacher"])
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(0.001, prec + rec)
        per_suite_f1[s] = f1

    if not per_suite_f1:
        return 0.0
    return float(np.mean(list(per_suite_f1.values())))


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
            out = model(X)
            loss = (phase_ce(out["phase_logits"], phase_tgt) +
                    0.5 * corridor_bce(out["corridor_logit"], corr_tgt) +
                    0.3 * release_bce(out["release_logit"], rel_tgt))
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        total_loss += batch_loss / len(batch_eks)
    return total_loss / n_batches


def validate_epoch(model, features, labels, episode_keys):
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--feature_csv", required=True)
    ap.add_argument("--label_csv", required=True)
    ap.add_argument("--episode_index", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cohort", default="all",
                    choices=["primary_eligible", "safety_abstention", "all"],
                    help="Episode cohort filter (primary_eligible filtering should be done at split build time)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--checkpoint_metric", default="val_loss",
                    choices=["val_loss", "val_suite_macro_event_f1"])
    ap.add_argument("--fsm_version", default="legacy_v1")
    ap.add_argument("--tau_corridor", type=float, default=0.3)
    ap.add_argument("--tau_release", type=float, default=0.3)
    ap.add_argument("--guard", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Fail early if F1 requested but may be unavailable (checked at first save)
    if args.checkpoint_metric == "val_suite_macro_event_f1":
        try:
            sys.path.insert(0, str(REPO / "src"))
            from gripper_attack.sc5_detector_runtime_v1r import SC5DetectorRuntimeV1R  # noqa: F401
        except ImportError:
            sys.exit("F1 checkpoint metric requires detector runtime. Use --checkpoint_metric val_loss or install runtime.")

    config = parse_config(args.config)
    tc = config.get("training_config", config)
    epochs = tc.get("epochs", args.epochs)
    batch_size = tc.get("batch_size", args.batch_size)
    lr = tc.get("lr", args.lr)
    patience = tc.get("patience", args.patience)
    ckpt_metric = tc.get("checkpoint_metric", args.checkpoint_metric)

    # Reject output_dir if non-empty (prevent cross-fold contamination)
    out_dir = Path(args.output_dir)
    if out_dir.exists() and list(out_dir.iterdir()):
        existing = [p.name for p in out_dir.iterdir()]
        sys.exit("Output directory not empty: {} (existing: {})".format(out_dir, existing[:10]))

    print("Loading episode index: {} [cohort={}]".format(args.episode_index, args.cohort))
    episode_index = load_episode_index(args.episode_index, cohort=args.cohort)
    print("  {} episodes in cohort".format(len(episode_index)))

    print("Loading features: {}".format(args.feature_csv))
    features = load_features(args.feature_csv)
    print("Loading labels: {}".format(args.label_csv))
    labels = load_labels(args.label_csv)
    print("Loading split: {}".format(args.split_file))
    with open(args.split_file) as f:
        split = json.load(f)

    train_eks_raw = split["splits"]["train"]
    val_eks_raw = split["splits"]["val"]

    print("Strict joining ({} train + {} val)...".format(len(train_eks_raw), len(val_eks_raw)))
    suite_map = strict_join(train_eks_raw, val_eks_raw, features, labels, episode_index)
    train_eks = [e for e in train_eks_raw if e in features]
    val_eks = [e for e in val_eks_raw if e in features]
    print("Train: {}, Val: {}, Suites: {}".format(len(train_eks), len(val_eks), sorted(set(suite_map.values()))))

    if len(train_eks) < batch_size:
        print("WARNING: {} train episodes < batch_size {}".format(len(train_eks), batch_size))

    print("Normalization from train only...")
    mean, std, norm_sha = compute_normalization(features, train_eks)
    for ek in list(features.keys()):
        features[ek] = (features[ek] - mean) / std

    git_info = get_git_info(REPO)

    print("Config: {}, metric: {}, cohort: {}".format(args.config, ckpt_metric, args.cohort))

    if args.dry_run:
        print("DRY_RUN: validation passed, no training.")
        print("Features: {}D, Phases: {}".format(N_FEATURES, N_PHASES))
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    model = SC5MLPV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_metric = float("-inf") if ckpt_metric == "val_suite_macro_event_f1" else float("inf")
    best_epoch = 0
    patience_counter = 0
    saved_ckpt_path = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, features, labels, train_eks, suite_map, optimizer, batch_size, rng)
        val_loss = validate_epoch(model, features, labels, val_eks)
        print("Epoch {:3d}: train_loss={:.4f} val_loss={:.4f}".format(epoch, train_loss, val_loss))

        # Compute checkpoint selection metric
        if ckpt_metric == "val_suite_macro_event_f1":
            # Save temp checkpoint for detector runtime to load
            tmp_ckpt = out_dir / ".tmp_ckpt.pt"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                        "feature_names": SC5_FEATURES, "phase_classes": SC5_PHASES,
                        "dataset_sha256": sha256_file(args.feature_csv),
                        "split_mode": "frozen", "normalization_source": "train_only",
                        "n_train": len(train_eks), "n_val": len(val_eks), "seed": args.seed}, tmp_ckpt)
            current_metric = compute_val_event_f1(str(tmp_ckpt), features, labels, val_eks,
                                                   suite_map, args.fsm_version,
                                                   args.tau_corridor, args.tau_release, args.guard)
            tmp_ckpt.unlink(missing_ok=True)
            improved = current_metric > best_metric
            print("  val F1 (suite-macro): {:.4f}  best: {:.4f}".format(current_metric, best_metric))
        else:
            current_metric = val_loss
            improved = current_metric < best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_ckpt_path = out_dir / "best_model.pt"
            ckpt = {
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "feature_names": list(SC5_FEATURES),
                "phase_classes": list(SC5_PHASES),
                "dataset_sha256": sha256_file(args.feature_csv),
                "split_mode": "frozen",
                "normalization_source": "train_only",
                "normalization_sha256": norm_sha,
                "n_train": len(train_eks), "n_val": len(val_eks),
                "seed": args.seed, "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint_metric": ckpt_metric,
                ("val_suite_macro_event_f1" if ckpt_metric == "val_suite_macro_event_f1" else ckpt_metric): best_metric,
                "tau_corridor": args.tau_corridor,
                "tau_release": args.tau_release,
                "guard": args.guard,
                "fsm_version": args.fsm_version,
                "cohort": args.cohort,
                "feature_csv_sha256": sha256_file(args.feature_csv),
                "label_csv_sha256": sha256_file(args.label_csv),
                "episode_index_sha256": sha256_file(args.episode_index),
                "split_file_sha256": sha256_file(args.split_file),
                "split_definition_sha256": sha256_bytes(json.dumps(split, sort_keys=True).encode()),
                "config_sha256": sha256_file(args.config),
                "repo_commit": git_info["commit"],
                "git_dirty": git_info["dirty"],
                "n_suites": len(set(suite_map.values())),
            }
            torch.save(ckpt, saved_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    print("Best epoch: {}, {}: {:.4f}".format(best_epoch, ckpt_metric, best_metric))
    if saved_ckpt_path:
        print("Checkpoint: {}".format(saved_ckpt_path))


if __name__ == "__main__":
    main()
