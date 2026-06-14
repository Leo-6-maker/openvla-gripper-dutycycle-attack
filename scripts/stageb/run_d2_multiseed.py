#!/usr/bin/env python3
"""D2 Phase E: Post-hoc multi-seed stability analysis.
Uses the frozen D1b.1 configuration. All seeds trained independently
on the same 90-train / 20-val split. Test on same 21-test split.
Marked POST_HOC — test was already seen.

Reports per-seed metrics, mean/median/min/max, baseline comparison,
tie count, seed42 status relative to distribution.
"""

import argparse, csv, hashlib, json, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stageb"))

from train_d1b_detector import (
    CandidateRanker, load_normalization, normalize_features,
    FEATURE_NAMES, TIE_TOLERANCE, ZERO_STDEV_THRESHOLD, CLIP_RANGE,
    TRAINING_SEED, BATCH_SIZE_TRACES, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    LR, WEIGHT_DECAY, MARGIN, sha256_file, per_trace_top1_accuracy, train_epoch,
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, 101, 202, 303, 404, 505]
N_SEEDS = len(SEEDS)


def evaluate_test(model, test_traces, means, stdevs, impute, device):
    from evaluate_d1b_detector import evaluate_model
    return evaluate_model(model, test_traces, means, stdevs, impute, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--norm-csv", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--end-seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    manifest = {r["trace_id"]: r for r in csv.DictReader(open(args.manifest))}
    candidates = list(csv.DictReader(open(args.candidate_table)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}
    means, stdevs, impute = load_normalization(args.norm_csv)

    by_trace = defaultdict(list)
    for c in candidates:
        tid = c["trace_id"]
        if tid in manifest:
            by_trace[tid].append(c)

    train_traces = {tid: cands for tid, cands in by_trace.items()
                    if manifest[tid]["split"] == "train"}
    val_traces = {tid: cands for tid, cands in by_trace.items()
                  if manifest[tid]["split"] == "val"}
    test_traces = {tid: cands for tid, cands in by_trace.items()
                   if tid in split and split[tid] == "test"}
    print(f"Train: {len(train_traces)} Val: {len(val_traces)} Test: {len(test_traces)}")

    seeds_to_run = SEEDS[args.start_seed:args.end_seed] if args.end_seed > 0 else SEEDS
    all_results = []

    for seed in seeds_to_run:
        print(f"\n=== SEED {seed} ===")
        torch.manual_seed(seed); np.random.seed(seed)
        model = CandidateRanker(n_features=16).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_acc = -1.0; best_val_mae = float("inf"); best_epoch = -1
        best_state = None; patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, optimizer, train_traces, means, stdevs, impute, device)
            val_acc, val_errs, val_cov, val_uniq = per_trace_top1_accuracy(
                model, val_traces, means, stdevs, impute, device)
            val_mae = np.mean([e["abs_error"] for e in val_errs
                              if e["abs_error"] >= 0 and e["unique_decision"] == 1])

            is_better = False
            if val_acc > best_val_acc + TIE_TOLERANCE:
                is_better = True
            elif abs(val_acc - best_val_acc) < TIE_TOLERANCE:
                if val_mae < best_val_mae - TIE_TOLERANCE:
                    is_better = True
                elif abs(val_mae - best_val_mae) < TIE_TOLERANCE and epoch < best_epoch:
                    is_better = True

            if is_better:
                best_val_acc = val_acc; best_val_mae = val_mae; best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        # Reload best and evaluate test
        model.load_state_dict(best_state)
        test_results = evaluate_test(model, test_traces, means, stdevs, impute, device)
        n = len(test_results)
        n_correct = sum(r["is_correct_top1"] for r in test_results)
        n_top2 = sum(r["is_top2"] for r in test_results)
        n_ties = sum(1 for r in test_results if r["n_ties_for_max"] > 1)
        test_acc = n_correct / n if n > 0 else 0

        all_results.append({
            "seed": seed, "best_epoch": best_epoch, "val_top1": best_val_acc,
            "val_mae": best_val_mae, "test_top1": round(test_acc, 4),
            "test_top1_n": f"{n_correct}/{n}", "test_top2": round(n_top2/n, 4),
            "test_n_ties": n_ties,
        })
        print(f"  best_epoch={best_epoch} val={best_val_acc:.3f} test={test_acc:.3f} ties={n_ties}")

    # Summary
    test_vals = [r["test_top1"] for r in all_results]
    print(f"\n=== MULTI-SEED SUMMARY ({len(all_results)} seeds) ===")
    print(f"  Test top-1: mean={np.mean(test_vals):.4f} median={np.median(test_vals):.4f} "
          f"min={np.min(test_vals):.4f} max={np.max(test_vals):.4f}")
    n_better = sum(1 for v in test_vals if v > 0.0952)  # baseline
    n_seed42 = next((r["test_top1"] for r in all_results if r["seed"] == 42), None)
    print(f"  Seeds beating baseline: {n_better}/{len(all_results)}")
    print(f"  Seed 42: {n_seed42}")
    seed42_rank = sum(1 for v in test_vals if v > n_seed42) + 1 if n_seed42 else -1
    print(f"  Seed 42 rank: {seed42_rank}/{len(all_results)}")

    with open(out / "d2_multiseed_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    with open(out / "d2_multiseed_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"]); w.writeheader()
        w.writerows([
            {"metric": "n_seeds", "value": len(all_results)},
            {"metric": "mean_test_top1", "value": round(np.mean(test_vals), 4)},
            {"metric": "median_test_top1", "value": round(np.median(test_vals), 4)},
            {"metric": "min_test_top1", "value": round(np.min(test_vals), 4)},
            {"metric": "max_test_top1", "value": round(np.max(test_vals), 4)},
            {"metric": "n_beating_baseline", "value": n_better},
            {"metric": "seed42_value", "value": n_seed42},
            {"metric": "seed42_rank", "value": seed42_rank},
            {"metric": "post_hoc", "value": "True"},
        ])

    print(f"\nD2 MULTI-SEED COMPLETE")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
