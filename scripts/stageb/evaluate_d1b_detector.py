#!/usr/bin/env python3
"""D1b.1: Learned critical-CLOSE candidate ranker — evaluation runner.

Frozen protocol:
  - Loads checkpoint frozen by training runner.
  - Evaluates on test split ONLY (single pass, no retraining).
  - 100% coverage: always selects highest-scoring candidate per trace.
  - Tie tolerance = 0.001; ties → incorrect.
  - Baseline: rule-based total_score, same tie rule.
  - Reports trace-level metrics + per-task breakdown.

All SHAs recorded. Single evaluation pass only.
"""

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from train_d1b_detector import (
    CandidateRanker, load_normalization, normalize_features,
    FEATURE_NAMES, TIE_TOLERANCE, ZERO_STDEV_THRESHOLD, CLIP_RANGE,
    TRAINING_SEED, sha256_file,
)


def evaluate_model(model, traces, means, stdevs, impute, device):
    """Per-trace evaluation. Returns list of per-trace metric dicts."""
    model.eval()
    results = []
    with torch.no_grad():
        for tid, candidates in traces.items():
            X = normalize_features(candidates, means, stdevs, impute).to(device)
            scores = model(X).cpu().numpy()
            best_idx = int(np.argmax(scores))
            max_score = scores[best_idx]
            ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]
            n_higher = 0
            n_equal = len(ties)

            tp_step = None; tp_score = None
            for c in candidates:
                if int(c.get("is_teacher_p", 0)) == 1:
                    tp_step = int(c["candidate_step"]); break
            pred_step = int(candidates[best_idx]["candidate_step"])

            # Compute TP rank: 1 + n_higher (competition rank)
            tp_idx = None
            for j, c in enumerate(candidates):
                if int(c.get("is_teacher_p", 0)) == 1: tp_idx = j; break
            if tp_idx is not None:
                tp_model_score = scores[tp_idx]
                n_higher = sum(1 for j, s in enumerate(scores) if s > tp_model_score + TIE_TOLERANCE)
                n_equal_tp = sum(1 for j, s in enumerate(scores) if abs(s - tp_model_score) < TIE_TOLERANCE)
            else:
                n_higher = -1; n_equal_tp = -1

            is_correct = (n_higher == 0 and n_equal_tp == 1)
            is_top2 = (n_higher <= 1)
            is_near = abs(pred_step - tp_step) <= 4 if tp_step is not None else False
            abs_err = abs(pred_step - tp_step) if tp_step is not None else -1

            results.append({
                "trace_id": tid,
                "task_key": candidates[0]["task_key"],
                "state_id": candidates[0]["state_id"],
                "n_candidates": len(candidates),
                "teacher_p_step": tp_step if tp_step is not None else -1,
                "predicted_step": pred_step,
                "abs_error": abs_err,
                "is_correct_top1": int(is_correct),
                "is_top2": int(is_top2),
                "is_near_pm4": int(is_near),
                "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
                "n_ties_for_max": n_equal,
                "model_max_score": round(float(max_score), 6),
            })
    return results


def evaluate_baseline(traces):
    """Rule-based baseline: total_score column. Same tie rule."""
    results = []
    for tid, candidates in traces.items():
        scores = np.array([float(c.get("total_score", 0)) for c in candidates])
        best_idx = int(np.argmax(scores))
        max_score = scores[best_idx]
        ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]

        tp_step = None
        for c in candidates:
            if int(c.get("is_teacher_p", 0)) == 1:
                tp_step = int(c["candidate_step"]); break
        pred_step = int(candidates[best_idx]["candidate_step"])

        tp_idx = None
        for j, c in enumerate(candidates):
            if int(c.get("is_teacher_p", 0)) == 1: tp_idx = j; break
        if tp_idx is not None:
            tp_score = scores[tp_idx]
            n_higher = sum(1 for s in scores if s > tp_score + TIE_TOLERANCE)
            n_equal_tp = sum(1 for s in scores if abs(s - tp_score) < TIE_TOLERANCE)
        else:
            n_higher = -1; n_equal_tp = -1

        is_correct = (n_higher == 0 and n_equal_tp == 1)
        is_top2 = (n_higher <= 1)
        is_near = abs(pred_step - tp_step) <= 4 if tp_step is not None else False
        abs_err = abs(pred_step - tp_step) if tp_step is not None else -1

        results.append({
            "trace_id": tid,
            "task_key": candidates[0]["task_key"],
            "state_id": candidates[0]["state_id"],
            "n_candidates": len(candidates),
            "teacher_p_step": tp_step if tp_step is not None else -1,
            "predicted_step": pred_step,
            "abs_error": abs_err,
            "is_correct_top1": int(is_correct),
            "is_top2": int(is_top2),
            "is_near_pm4": int(is_near),
            "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
            "n_ties_for_max": len(ties),
            "baseline_max_score": round(float(max_score), 4),
        })
    return results


def summarize(results, label):
    n = len(results)
    n_correct = sum(r["is_correct_top1"] for r in results)
    n_top2 = sum(r["is_top2"] for r in results)
    n_near = sum(r["is_near_pm4"] for r in results)
    errors = [r["abs_error"] for r in results if r["abs_error"] >= 0]
    mae = np.mean(errors) if errors else 0
    med_ae = np.median(errors) if errors else 0
    print(f"\n{label} (n={n}):")
    print(f"  Top-1 accuracy: {n_correct}/{n} = {n_correct/n:.4f}")
    print(f"  Top-2 accuracy: {n_top2}/{n} = {n_top2/n:.4f}")
    print(f"  Near-correct (±4): {n_near}/{n} = {n_near/n:.4f}")
    print(f"  MAE: {mae:.1f} steps  Median AE: {med_ae:.1f} steps")
    return {"n": n, "top1": n_correct, "top1_rate": round(n_correct/n, 4),
            "top2": n_top2, "top2_rate": round(n_top2/n, 4),
            "near": n_near, "near_rate": round(n_near/n, 4),
            "mae": round(mae, 1), "median_ae": round(med_ae, 1)}


def per_task_summary(results):
    by_task = defaultdict(list)
    for r in results: by_task[r["task_key"]].append(r)
    rows = []
    for task in sorted(by_task):
        tr = by_task[task]
        n = len(tr)
        n_correct = sum(r["is_correct_top1"] for r in tr)
        errors = [r["abs_error"] for r in tr if r["abs_error"] >= 0]
        mae = np.mean(errors) if errors else 0
        rows.append({"task_key": task, "n_traces": n, "n_top1_correct": n_correct,
                     "top1_rate": round(n_correct/n, 4), "mae": round(mae, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-table", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)

    # Provenance
    artifact_hashes = {
        "runner_sha": sha256_file(__file__),
        "checkpoint_sha": sha256_file(args.checkpoint),
        "candidate_table_sha": sha256_file(args.candidate_table),
        "split_manifest_sha": sha256_file(args.split_manifest),
    }
    print("=== EVALUATION ARTIFACTS ===")
    for k, v in artifact_hashes.items():
        print(f"  {k}: {v[:16]}...")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']} val_acc={ckpt['val_top1_acc']:.4f} val_mae={ckpt['val_mae']:.1f}")
    artifact_hashes["training_manifest_sha"] = ckpt["artifact_hashes"].get("manifest_sha", "unknown")
    artifact_hashes["training_runner_sha"] = ckpt["artifact_hashes"].get("runner_sha", "unknown")

    # Load data
    candidates = list(csv.DictReader(open(args.candidate_table)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}

    # Group by trace, test split only
    by_trace = defaultdict(list)
    for c in candidates:
        tid = c["trace_id"]
        if tid in split and split[tid] == "test":
            by_trace[tid].append(c)
    print(f"Test traces: {len(by_trace)}")

    # Verify each trace has exactly one positive
    for tid, cands in by_trace.items():
        n_pos = sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1)
        assert n_pos == 1, f"Trace {tid}: expected 1 positive, got {n_pos}"

    # Load normalization from checkpoint
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    # Build model
    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Model evaluation (SINGLE PASS) ──
    model_results = evaluate_model(model, by_trace, means, stdevs, impute, device)
    model_summary = summarize(model_results, "MODEL (learned ranker)")

    # ── Baseline evaluation (same traces) ──
    baseline_results = evaluate_baseline(by_trace)
    baseline_summary = summarize(baseline_results, "BASELINE (rule-based total_score)")

    # ── Per-task breakdown ──
    model_tasks = per_task_summary(model_results)
    baseline_tasks = per_task_summary(baseline_results)

    # ── Write outputs ──
    mfields = list(model_results[0].keys())
    with open(out / "d1b_model_test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields); w.writeheader(); w.writerows(model_results)

    bfields = list(baseline_results[0].keys())
    with open(out / "d1b_baseline_test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bfields); w.writeheader(); w.writerows(baseline_results)

    with open(out / "d1b_test_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "model", "baseline"]); w.writeheader()
        for key in ["n", "top1", "top1_rate", "top2", "top2_rate", "near", "near_rate", "mae", "median_ae"]:
            w.writerow({"metric": key, "model": model_summary[key], "baseline": baseline_summary[key]})

    with open(out / "d1b_test_per_task.csv", "w", newline="") as f:
        fields = ["task_key", "n_traces",
                  "model_top1_correct", "model_top1_rate", "model_mae",
                  "baseline_top1_correct", "baseline_top1_rate", "baseline_mae"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for mt, bt in zip(model_tasks, baseline_tasks):
            w.writerow({
                "task_key": mt["task_key"], "n_traces": mt["n_traces"],
                "model_top1_correct": mt["n_top1_correct"], "model_top1_rate": mt["top1_rate"],
                "model_mae": mt["mae"],
                "baseline_top1_correct": bt["n_top1_correct"], "baseline_top1_rate": bt["top1_rate"],
                "baseline_mae": bt["mae"],
            })

    # Run log
    end_time = datetime.now(timezone.utc)
    with open(out / "d1b_evaluation_run_log.txt", "w") as f:
        f.write(f"D1b EVALUATION RUN LOG\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        for k, v in artifact_hashes.items():
            f.write(f"{k}: {v}\n")
        f.write(f"test_traces: {len(by_trace)}\n")
        f.write(f"model_top1: {model_summary['top1_rate']}\n")
        f.write(f"baseline_top1: {baseline_summary['top1_rate']}\n")
        f.write(f"evaluation_pass_count: 1\n")

    print(f"\nEVALUATION COMPLETE — single pass. Output: {out}")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
