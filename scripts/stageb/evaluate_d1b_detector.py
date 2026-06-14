#!/usr/bin/env python3
"""D1b.2: Learned critical-CLOSE candidate ranker — evaluation runner.

Frozen protocol:
  - Loads frozen checkpoint. Fail-closed artifact verification at startup.
  - Evaluates on test split ONLY (single pass, one-shot sentinel).
  - 100% coverage: always selects highest-scoring candidate per trace.
  - Tie tolerance = 0.001; ties for max → no unique decision (predicted_step=-1).
  - Baseline: rule-based total_score, same tie rule.
  - Reports trace-level metrics, coverage, conditional MAE, per-task breakdown.

All SHAs verified before execution. Single evaluation pass only.
"""

import argparse, csv, hashlib, json, math, os, sys, time, traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_d1b_detector import (
    CandidateRanker, load_normalization, normalize_features,
    FEATURE_NAMES, TIE_TOLERANCE, ZERO_STDEV_THRESHOLD, CLIP_RANGE,
    TRAINING_SEED, sha256_file,
)


def evaluate_model(model, traces, means, stdevs, impute, device):
    """Per-trace evaluation with tie-robust semantics.
    Tie for max → predicted_step=-1, abs_error=-1, no unique decision.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for tid, candidates in traces.items():
            X = normalize_features(candidates, means, stdevs, impute).to(device)
            scores = model(X).cpu().numpy()
            max_score = scores.max()
            ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]

            tp_step = None; tp_idx = None
            for j, c in enumerate(candidates):
                if int(c.get("is_teacher_p", 0)) == 1:
                    tp_step = int(c["candidate_step"]); tp_idx = j; break

            # TP competition rank
            if tp_idx is not None:
                tp_s = scores[tp_idx]
                n_higher = sum(1 for s in scores if s > tp_s + TIE_TOLERANCE)
                n_equal_tp = sum(1 for s in scores if abs(s - tp_s) < TIE_TOLERANCE)
            else:
                n_higher = -1; n_equal_tp = -1

            is_unique_top1 = (n_higher == 0 and n_equal_tp == 1)
            is_top2 = (n_higher <= 1)

            if len(ties) == 1:
                best_idx = ties[0]
                pred_step = int(candidates[best_idx]["candidate_step"])
                abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
                is_near = abs_err <= 4 if abs_err >= 0 else False
            else:
                pred_step = -1  # no unique decision
                abs_err = -1
                is_near = False

            results.append({
                "trace_id": tid,
                "task_key": candidates[0]["task_key"],
                "state_id": candidates[0]["state_id"],
                "n_candidates": len(candidates),
                "teacher_p_step": tp_step if tp_step is not None else -1,
                "predicted_step": pred_step,
                "abs_error": abs_err,
                "is_correct_top1": int(is_unique_top1),
                "is_top2": int(is_top2),
                "is_near_pm4": int(is_near),
                "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
                "n_ties_for_max": len(ties),
                "unique_decision": int(len(ties) == 1),
                "model_max_score": round(float(max_score), 6),
            })
    return results


def evaluate_baseline(traces):
    """Rule-based baseline with same tie semantics as model."""
    results = []
    for tid, candidates in traces.items():
        scores = np.array([float(c.get("total_score", 0)) for c in candidates])
        max_score = scores.max()
        ties = [i for i, s in enumerate(scores) if abs(s - max_score) < TIE_TOLERANCE]

        tp_step = None; tp_idx = None
        for j, c in enumerate(candidates):
            if int(c.get("is_teacher_p", 0)) == 1:
                tp_step = int(c["candidate_step"]); tp_idx = j; break

        if tp_idx is not None:
            tp_s = scores[tp_idx]
            n_higher = sum(1 for s in scores if s > tp_s + TIE_TOLERANCE)
            n_equal_tp = sum(1 for s in scores if abs(s - tp_s) < TIE_TOLERANCE)
        else:
            n_higher = -1; n_equal_tp = -1

        is_unique_top1 = (n_higher == 0 and n_equal_tp == 1)
        is_top2 = (n_higher <= 1)

        if len(ties) == 1:
            best_idx = ties[0]
            pred_step = int(candidates[best_idx]["candidate_step"])
            abs_err = abs(pred_step - tp_step) if tp_step is not None else -1
            is_near = abs_err <= 4 if abs_err >= 0 else False
        else:
            pred_step = -1; abs_err = -1; is_near = False

        results.append({
            "trace_id": tid,
            "task_key": candidates[0]["task_key"],
            "state_id": candidates[0]["state_id"],
            "n_candidates": len(candidates),
            "teacher_p_step": tp_step if tp_step is not None else -1,
            "predicted_step": pred_step,
            "abs_error": abs_err,
            "is_correct_top1": int(is_unique_top1),
            "is_top2": int(is_top2),
            "is_near_pm4": int(is_near),
            "tp_competition_rank": n_higher + 1 if n_higher >= 0 else -1,
            "n_ties_for_max": len(ties),
            "unique_decision": int(len(ties) == 1),
            "baseline_max_score": round(float(max_score), 4),
        })
    return results


def summarize(results, label):
    n = len(results)
    n_correct = sum(r["is_correct_top1"] for r in results)
    n_top2 = sum(r["is_top2"] for r in results)
    n_near = sum(r["is_near_pm4"] for r in results)
    n_unique = sum(r["unique_decision"] for r in results)
    n_tied = n - n_unique
    coverage = n_unique / n if n > 0 else 0
    # Conditional MAE on unique-decision traces only
    unique_errors = [r["abs_error"] for r in results if r["unique_decision"] == 1 and r["abs_error"] >= 0]
    cond_mae = np.mean(unique_errors) if unique_errors else 0
    cond_med_ae = np.median(unique_errors) if unique_errors else 0
    cond_near = sum(1 for r in results if r["unique_decision"] == 1 and r["is_near_pm4"] == 1)
    print(f"\n{label} (n={n}):")
    print(f"  Unique top-1 accuracy: {n_correct}/{n} = {n_correct/n:.4f}")
    print(f"  Competition top-2: {n_top2}/{n} = {n_top2/n:.4f}")
    print(f"  Unique-decision coverage: {n_unique}/{n} = {coverage:.4f}")
    print(f"  Tied-for-max traces: {n_tied}/{n}")
    print(f"  Conditional ±4 (unique-decision only): {cond_near}/{n_unique} = {cond_near/n_unique:.4f}" if n_unique > 0 else "  Conditional ±4: N/A")
    print(f"  Conditional MAE: {cond_mae:.1f} steps  Median AE: {cond_med_ae:.1f} steps")
    return {"n": n, "top1": n_correct, "top1_rate": round(n_correct/n, 4),
            "top2": n_top2, "top2_rate": round(n_top2/n, 4),
            "coverage": round(coverage, 4), "n_tied": n_tied,
            "cond_near": cond_near, "cond_near_rate": round(cond_near/n_unique, 4) if n_unique > 0 else "",
            "cond_mae": round(cond_mae, 1), "cond_median_ae": round(cond_med_ae, 1)}


def per_task_summary(results):
    by_task = defaultdict(list)
    for r in results: by_task[r["task_key"]].append(r)
    rows = []
    for task in sorted(by_task):
        tr = by_task[task]; n = len(tr)
        n_correct = sum(r["is_correct_top1"] for r in tr)
        n_unique = sum(r["unique_decision"] for r in tr)
        ue = [r["abs_error"] for r in tr if r["unique_decision"] == 1 and r["abs_error"] >= 0]
        rows.append({"task_key": task, "n_traces": n,
                     "n_top1_correct": n_correct, "top1_rate": round(n_correct/n, 4),
                     "coverage": round(n_unique/n, 4),
                     "cond_mae": round(np.mean(ue), 1) if ue else ""})
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

    # ── D1b.2: Fail-closed artifact verification ──
    from verify_d1b_artifacts import verify_all
    seal_ok, seal_failures, seal_results = verify_all()
    print("=== ARTIFACT VERIFICATION ===")
    for k, v in seal_results.items():
        print(f"  {k}: {v[:16]}... OK")
    if not seal_ok:
        print(f"FATAL: artifact seal failed:")
        for f in seal_failures: print(f"  {f}")
        sys.exit(1)
    print("  SEAL PASS\n")

    # ── D1b.2: One-shot test sentinel ──
    sentinel_path = out / "test_eval_started.json"
    if sentinel_path.exists():
        print("FATAL: test evaluation already started (sentinel exists).")
        print("RESULT_CLASS: TEST_EXECUTION_ABORTED_AFTER_SENTINEL")
        sys.exit(2)

    start_time = datetime.now(timezone.utc)
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()
    except:
        git_commit = "unknown"

    artifact_hashes = {
        "runner_sha": sha256_file(__file__),
        "checkpoint_sha": sha256_file(args.checkpoint),
        "candidate_table_sha": sha256_file(args.candidate_table),
        "split_manifest_sha": sha256_file(args.split_manifest),
    }
    sentinel = {
        "checkpoint_sha": artifact_hashes["checkpoint_sha"],
        "candidate_table_sha": artifact_hashes["candidate_table_sha"],
        "split_manifest_sha": artifact_hashes["split_manifest_sha"],
        "evaluation_runner_sha": artifact_hashes["runner_sha"],
        "timestamp": start_time.isoformat(),
        "git_commit": git_commit,
    }
    with open(sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    print(f"Test sentinel created: {sentinel_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']} val_acc={ckpt['val_top1_acc']:.4f} val_mae={ckpt['val_mae']:.1f}")

    # Load data
    candidates = list(csv.DictReader(open(args.candidate_table)))
    split = {r["trace_id"]: r["split"] for r in csv.DictReader(open(args.split_manifest))}
    by_trace = defaultdict(list)
    for c in candidates:
        tid = c["trace_id"]
        if tid in split and split[tid] == "test":
            by_trace[tid].append(c)
    print(f"Test traces: {len(by_trace)}")
    assert len(by_trace) == 21, f"Expected 21 test traces, got {len(by_trace)}"

    for tid, cands in by_trace.items():
        assert sum(1 for c in cands if int(c.get("is_teacher_p", 0)) == 1) == 1

    # Load normalization from checkpoint
    means = ckpt["normalization"]["means"]
    stdevs = ckpt["normalization"]["stdevs"]
    impute = ckpt["normalization"]["impute"]

    model = CandidateRanker(n_features=16).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Single-pass model evaluation ──
    print("\n=== MODEL EVALUATION ===")
    model_results = evaluate_model(model, by_trace, means, stdevs, impute, device)
    model_summary = summarize(model_results, "MODEL (learned ranker)")

    # ── Baseline on same traces ──
    print("\n=== BASELINE EVALUATION ===")
    baseline_results = evaluate_baseline(by_trace)
    baseline_summary = summarize(baseline_results, "BASELINE (rule-based total_score)")

    model_tasks = per_task_summary(model_results)
    baseline_tasks = per_task_summary(baseline_results)

    # ── Write outputs ──
    mfields = list(model_results[0].keys())
    with open(out / "d1b_model_test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields); w.writeheader(); w.writerows(model_results)
    bfields = list(baseline_results[0].keys())
    with open(out / "d1b_baseline_test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bfields); w.writeheader(); w.writerows(baseline_results)

    summary_fields = ["metric", "model", "baseline"]
    with open(out / "d1b_test_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields); w.writeheader()
        for key in ["n", "top1", "top1_rate", "top2", "top2_rate", "coverage",
                     "n_tied", "cond_near", "cond_near_rate", "cond_mae", "cond_median_ae"]:
            w.writerow({"metric": key, "model": model_summary.get(key, ""),
                        "baseline": baseline_summary.get(key, "")})

    with open(out / "d1b_test_per_task.csv", "w", newline="") as f:
        fields = ["task_key", "n_traces",
                  "model_top1", "model_top1_rate", "model_coverage", "model_cond_mae",
                  "baseline_top1", "baseline_top1_rate", "baseline_coverage", "baseline_cond_mae"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for mt, bt in zip(model_tasks, baseline_tasks):
            w.writerow({
                "task_key": mt["task_key"], "n_traces": mt["n_traces"],
                "model_top1": mt["n_top1_correct"], "model_top1_rate": mt["top1_rate"],
                "model_coverage": mt["coverage"], "model_cond_mae": mt["cond_mae"],
                "baseline_top1": bt["n_top1_correct"], "baseline_top1_rate": bt["top1_rate"],
                "baseline_coverage": bt["coverage"], "baseline_cond_mae": bt["cond_mae"],
            })

    end_time = datetime.now(timezone.utc)
    with open(out / "d1b_evaluation_run_log.txt", "w") as f:
        f.write(f"D1b EVALUATION RUN LOG\n")
        f.write(f"start: {start_time.isoformat()}\nend: {end_time.isoformat()}\n")
        for k, v in artifact_hashes.items():
            f.write(f"{k}: {v}\n")
        for k, v in seal_results.items():
            f.write(f"seal_{k}: {v}\n")
        f.write(f"git_commit: {git_commit}\n")
        f.write(f"test_traces: {len(by_trace)}\n")
        f.write(f"evaluation_pass_count: 1\n")
        f.write(f"sentinel: {str(sentinel_path)}\n")

    print(f"\nEVALUATION COMPLETE — single pass.")
    print(f"Output: {out}")
    print("TEST_EVALUATIONS: 1")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc(); sys.exit(1)
