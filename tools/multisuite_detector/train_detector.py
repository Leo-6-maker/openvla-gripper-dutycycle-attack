#!/usr/bin/env python3
"""Multi-suite SC5MLP detector training with strict data integrity.

Uses shared strict_loader for feature/label/index loading.
Checkpoint selection: val_loss (default) or val_suite_macro_event_f1 (via detector replay).
Fail-closed on F1 if detector runtime unavailable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
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
from score_fsm_legacy_v1 import run_fsm_legacy_v1, model_to_scores


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
        dirty_out = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("Invalid git commit: {}".format(commit[:20]))
        dirty = bool(dirty_out)
        return {"commit": commit, "dirty": dirty, "dirty_files": len(dirty_out.split(chr(10))) if dirty_out else 0}
    except Exception as e:
        raise RuntimeError("Git unavailable — formal training requires a clean git worktree: {}".format(e))


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


def compute_val_event_f1(model, features, labels, episode_keys, suite_map, tau_c, tau_r, guard):
    """Compute validation suite-macro event F1 using model logits + inline FSM replay.

    Correct TP/FP/FN accounting:
      TP = has_teacher AND emitted AND in_window
      FP = emitted AND NOT in_window
      FN = has_teacher AND NOT in_window
      TN = NOT has_teacher AND NOT emitted

    A wrong-time emission on a teacher-positive episode counts as BOTH FP and FN.
    """
    model.eval()

    # Track input/excluded/scored separately (excluded counted BEFORE continue)
    n_input_by_suite = defaultdict(int)
    n_excluded_by_suite = defaultdict(int)
    excluded_keys = []
    by_suite = defaultdict(list)

    with torch.no_grad():
        for ek in episode_keys:
            s = suite_map[ek]
            n_input_by_suite[s] += 1

            feats = torch.from_numpy(features[ek])
            cp, rp, phase_names = model_to_scores(model, feats)
            emitted, emit_step = run_fsm_legacy_v1(cp, rp, phase_names, tau_c, tau_r, guard)

            labs = labels[ek]
            n_steps = len(labs["phase"])

            corr = labs["corridor"][:n_steps]
            has_teacher = int(corr.any())

            # Multi-event detection (counted before exclude)
            n_regions = 1
            if has_teacher:
                diff = np.diff(np.concatenate([[0], corr, [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                n_regions = len(starts)

            if n_regions > 1:
                n_excluded_by_suite[s] += 1
                excluded_keys.append(ek)
                continue

            in_window = False
            if has_teacher and n_regions == 1:
                wstart, wend = starts[0], ends[0] - 1
                in_window = emitted and (emit_step >= wstart and emit_step <= wend)

            by_suite[s].append({
                "emitted": emitted, "in_window": in_window,
                "has_teacher": has_teacher,
                "n_corridor_regions": n_regions,
            })

    # Compute per-suite F1 with verified input/scored/excluded counts
    per_suite = {}
    total_excluded = sum(n_excluded_by_suite.values())
    for s in sorted(n_input_by_suite):
        n_in = n_input_by_suite[s]
        n_excl = n_excluded_by_suite[s]
        rs = by_suite.get(s, [])
        n_scored = len(rs)

        if n_scored == 0:
            raise RuntimeError(
                "Suite {} has 0 scored episodes ({} input, {} excluded). "
                "All validation episodes excluded — cannot compute F1.".format(s, n_in, n_excl))
        if n_in != n_scored + n_excl:
            raise RuntimeError(
                "Suite {} count mismatch: input={} scored={} excluded={}".format(s, n_in, n_scored, n_excl))

        tp = sum(1 for r in rs if r["has_teacher"] and r["emitted"] and r["in_window"])
        fp = sum(1 for r in rs if r["emitted"] and not r["in_window"])
        fn = sum(1 for r in rs if r["has_teacher"] and not r["in_window"])
        tn = sum(1 for r in rs if not r["has_teacher"] and not r["emitted"])

        # Mutually-exclusive outcome buckets (partition of n_scored)
        tp_in = sum(1 for r in rs if r["has_teacher"] and r["emitted"] and r["in_window"])
        wt_pos = sum(1 for r in rs if r["has_teacher"] and r["emitted"] and not r["in_window"])
        missed = sum(1 for r in rs if r["has_teacher"] and not r["emitted"])
        fe_neg = sum(1 for r in rs if not r["has_teacher"] and r["emitted"])
        tn_neg = sum(1 for r in rs if not r["has_teacher"] and not r["emitted"])
        partition_sum = tp_in + wt_pos + missed + fe_neg + tn_neg
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(0.001, prec + rec)
        per_suite[s] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": float(prec), "recall": float(rec), "f1": float(f1),
                         "n_input_episodes": n_in, "n_scored_episodes": n_scored,
                         "excluded_multi_event": n_excl,
                         "outcome_buckets": {
                             "tp_in_window": tp_in,
                             "wrong_time_positive": wt_pos,
                             "missed_positive": missed,
                             "false_emit_negative": fe_neg,
                             "tn_negative": tn_neg,
                             "partition_sum": partition_sum,
                         }}

    # Verify all expected training suites have at least some scored validation episodes
    expected_suites = set(suite_map.values())
    actual_suites = set(per_suite.keys())
    missing_suites = expected_suites - actual_suites
    if missing_suites:
        raise RuntimeError(
            "Validation missing expected suites: {}. Expected: {}. Actual: {}".format(
                sorted(missing_suites), sorted(expected_suites), sorted(actual_suites)))

    if not per_suite:
        raise RuntimeError("No suites with scored episodes — cannot compute F1")

    # Bind excluded episode keys SHA for auditability
    excluded_keys.sort()
    excluded_sha = hashlib.sha256(json.dumps(excluded_keys, sort_keys=True).encode()).hexdigest()

    macro_f1 = float(np.mean([v["f1"] for v in per_suite.values()]))
    return macro_f1, per_suite, total_excluded, excluded_keys, excluded_sha


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

    # F1 checkpoint metric uses model logits + inline FSM — no runtime dependency needed
    if args.checkpoint_metric not in ("val_loss", "val_suite_macro_event_f1"):
        sys.exit("Unknown checkpoint_metric: {}".format(args.checkpoint_metric))

    config = parse_config(args.config)
    tc = config.get("training_config", config)
    epochs = tc.get("epochs", args.epochs)
    batch_size = tc.get("batch_size", args.batch_size)
    lr = tc.get("lr", args.lr)
    patience = tc.get("patience", args.patience)
    ckpt_metric = tc.get("checkpoint_metric", args.checkpoint_metric)
    if ckpt_metric not in ("val_loss", "val_suite_macro_event_f1"):
        sys.exit("Invalid checkpoint_metric: {} (must be val_loss or val_suite_macro_event_f1)".format(ckpt_metric))

    # Reject output_dir if non-empty (only in non-dry_run mode)
    out_dir = Path(args.output_dir)
    if not args.dry_run and out_dir.exists() and list(out_dir.iterdir()):
        existing = [p.name for p in out_dir.iterdir()]
        sys.exit("Output directory not empty: {} (existing: {})".format(out_dir, existing[:10]))

    # Pre-training environment checks (before any data loading)
    try:
        git_info = get_git_info(REPO)
    except Exception as e:
        if args.dry_run:
            git_info = {"commit": "DRY_RUN", "dirty": False, "dirty_files": 0}
        else:
            sys.exit("Git unavailable — formal training requires a clean git worktree: {}".format(e))
    if not args.dry_run:
        if git_info["dirty"]:
            sys.exit("Git checkout is dirty ({} files) — formal training requires clean checkout".format(git_info["dirty_files"]))
        if not re.fullmatch(r"[0-9a-f]{40}", git_info["commit"]):
            sys.exit("Invalid repo_commit: {}".format(git_info["commit"]))

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

    # Fail-closed: only legacy_v1 FSM implemented for F1 replay
    if args.fsm_version != "legacy_v1":
        sys.exit("F1 checkpoint selection only supports fsm_version=legacy_v1. Got: {}".format(args.fsm_version))

    print("Config: {}, metric: {}, cohort: {}, FSM: {}".format(args.config, ckpt_metric, args.cohort, args.fsm_version))

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
    best_false_emits = float("inf")
    best_epoch = 0
    patience_counter = 0
    saved_ckpt_path = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, features, labels, train_eks, suite_map, optimizer, batch_size, rng)
        val_loss = validate_epoch(model, features, labels, val_eks)
        print("Epoch {:3d}: train_loss={:.4f} val_loss={:.4f}".format(epoch, train_loss, val_loss))

        # Compute checkpoint selection metric with tie-breaker
        if ckpt_metric == "val_suite_macro_event_f1":
            current_metric, f1_details, f1_excluded, f1_excluded_keys, f1_excluded_sha = compute_val_event_f1(
                model, features, labels, val_eks, suite_map,
                args.tau_corridor, args.tau_release, args.guard)

            # Tie-breaker tuple: (macro_f1, -false_emits, -post_release, -epoch)
            total_fp = sum(d["fp"] for d in f1_details.values())
            # Denominator: scored episodes only (excluded multi-event not in population)
            total_scored = sum(d["n_scored_episodes"] for d in f1_details.values())
            false_emits_per_ep = total_fp / max(1, total_scored)
            current_tuple = (round(current_metric, 4), -false_emits_per_ep, epoch)

            if best_metric == float("-inf"):
                improved = True
            else:
                # Primary: macro F1 with min_delta 0.001
                if current_metric > best_metric + 0.001:
                    improved = True
                elif current_metric > best_metric - 0.001:
                    # Within min_delta: use tie-breakers
                    if false_emits_per_ep < best_false_emits - 0.001:
                        improved = True
                    else:
                        improved = False
                else:
                    improved = False

            if epoch == 1 or improved:
                for s, d in sorted(f1_details.items()):
                    print("  val F1 {}: tp={} fp={} fn={} tn={} prec={:.3f} rec={:.3f} f1={:.3f} in={} scored={} excl={}".format(
                        s, d["tp"], d["fp"], d["fn"], d["tn"], d["precision"], d["recall"], d["f1"],
                        d["n_input_episodes"], d["n_scored_episodes"], d["excluded_multi_event"]))
            print("  val F1 (suite-macro): {:.4f}  best: {:.4f}  false_emits/ep: {:.3f}".format(
                current_metric, best_metric if best_metric != float("-inf") else 0.0, false_emits_per_ep))
        else:
            current_metric = val_loss
            improved = current_metric < best_metric

        if improved:
            best_metric = current_metric
            if ckpt_metric == "val_suite_macro_event_f1":
                best_false_emits = false_emits_per_ep
            best_epoch = epoch
            patience_counter = 0
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_ckpt_path = out_dir / "best_model.pt"
            ckpt = {
                "schema_version": "sc5mlp_v1.0",
                "model_type": "SC5MLPV1",
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "feature_names": list(SC5_FEATURES),
                "phase_classes": list(SC5_PHASES),
                "split_mode": "frozen",
                "normalization_source": "train_only",
                "normalization_sha256": norm_sha,
                "n_train": len(train_eks), "n_val": len(val_eks),
                "n_suites": len(set(suite_map.values())),
                "seed": args.seed, "epoch": epoch,
                "val_loss": val_loss,
                "checkpoint_metric": ckpt_metric,
                "tau_corridor": float(args.tau_corridor),
                "tau_release": float(args.tau_release),
                "guard": int(args.guard),
                "K": 10,
                "fsm_version": "legacy_v1",
                "cohort": args.cohort,
                "feature_csv_sha256": sha256_file(args.feature_csv),
                "label_csv_sha256": sha256_file(args.label_csv),
                "episode_index_sha256": sha256_file(args.episode_index),
                "split_file_sha256": sha256_file(args.split_file),
                "split_definition_sha256": sha256_bytes(json.dumps(split, sort_keys=True).encode()),
                "config_sha256": sha256_file(args.config),
                "repo_commit": git_info["commit"],
                "git_dirty": bool(git_info["dirty"]),
            }
            # Bind selection metrics
            if ckpt_metric == "val_suite_macro_event_f1":
                ckpt["val_suite_macro_event_f1"] = float(best_metric)
                ckpt["val_suite_macro_event_f1_details"] = {
                    s: {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": d["tn"],
                        "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
                        "n_input_episodes": d["n_input_episodes"], "n_scored_episodes": d["n_scored_episodes"],
                        "excluded_multi_event": d["excluded_multi_event"],
                        "outcome_buckets": d["outcome_buckets"]}
                    for s, d in f1_details.items()}
                ckpt["selection_false_emits_per_episode"] = float(best_false_emits)
                ckpt["multi_event_excluded_total"] = int(f1_excluded)
                ckpt["multi_event_excluded_keys_sha256"] = f1_excluded_sha
                ckpt["selection_tie_breaker"] = "macro_f1 > best+0.001; within delta: lower false_emits"
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
