#!/usr/bin/env python3
"""evaluate_rule_based_early_grasp_trigger.py — Mandatory baseline: rule-based CLOSE->OPEN
transition detector.

T_rule = first step where gripper_command < 0.5 for K consecutive steps.
"""

from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-path", default="data/detector/object_clean_sequences_v1.npz")
    ap.add_argument("--meta-csv", default="data/detector/object_clean_sequences_v1_meta.csv")
    ap.add_argument("--output-csv", default="tables/object_rule_based_trigger_eval.csv")
    ap.add_argument("--output-report", default="reports/OBJECT_RULE_BASED_TRIGGER_BASELINE.md")
    ap.add_argument("--close-K", type=int, default=2,
                    help="Consecutive steps with gripper_command < threshold")
    ap.add_argument("--close-threshold", type=float, default=0.5,
                    help="Threshold below which gripper_command is considered OPEN")
    return ap.parse_args()


def find_rule_trigger(gripper_commands, K=2, threshold=0.5):
    """Find first sustained OPEN command."""
    streak = 0
    for t, gc in enumerate(gripper_commands):
        if gc < threshold:
            streak += 1
            if streak >= K:
                return t - K + 1
        else:
            streak = 0
    return None


def main():
    args = parse_args()

    if not os.path.exists(args.npz_path):
        print(f"ERROR: NPZ not found: {args.npz_path}")
        print("Run prepare_object_detector_dataset.py first.")
        sys.exit(1)

    data = np.load(args.npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    mask = data["mask"]
    episode_ids = data.get("episode_ids", np.array([f"ep_{i}" for i in range(len(X))]))
    feature_names = list(data.get("feature_names", []))

    # Find gripper_command index
    grip_idx = None
    for i, fn in enumerate(feature_names):
        if fn == "gripper_command":
            grip_idx = i
            break
    if grip_idx is None:
        print("ERROR: gripper_command not found in features")
        sys.exit(1)

    if os.path.exists(args.meta_csv):
        with open(args.meta_csv, newline="") as f:
            meta = {r["episode_id"]: r for r in csv.DictReader(f)}
    else:
        meta = {}

    N = len(X)
    print(f"Loaded {N} episodes")

    results = []
    per_task = defaultdict(list)

    for i in range(N):
        eid = str(episode_ids[i])
        m = mask[i]
        T = int(m.sum())
        gc = X[i, :T, grip_idx]

        # Find rule-based trigger
        T_rule = find_rule_trigger(gc, K=args.close_K, threshold=args.close_threshold)

        # Find oracle T_gform from meta
        ep_meta = meta.get(eid, {})
        T_gform_str = ep_meta.get("T_gform", "")
        T_gform = int(T_gform_str) if T_gform_str else None
        task = ep_meta.get("task_name", "unknown")

        # Compute errors
        error = None
        error_abs = None
        if T_rule is not None and T_gform is not None:
            error = T_rule - T_gform
            error_abs = abs(error)

        result = {
            "episode_id": eid,
            "task_name": task,
            "T_gform": T_gform if T_gform is not None else "",
            "T_rule": T_rule if T_rule is not None else "",
            "rule_triggered": T_rule is not None,
            "error": error if error is not None else "",
            "abs_error": error_abs if error_abs is not None else "",
            "within_5": (error_abs is not None and error_abs <= 5),
            "within_10": (error_abs is not None and error_abs <= 10),
            "early_trigger": (error is not None and error < -5),
            "late_trigger": (error is not None and error > 5),
        }
        results.append(result)

        if error_abs is not None:
            per_task[task].append(error_abs)

    # ── Aggregate metrics ──
    valid = [r for r in results if r["error"] != ""]
    n_valid = len(valid)
    errors = [r["error"] for r in valid]
    abs_errors = [r["abs_error"] for r in valid]

    metrics = {
        "n_episodes": N,
        "n_rule_triggered": sum(1 for r in results if r["rule_triggered"]),
        "n_oracle_labeled": sum(1 for r in results if r["T_gform"] != ""),
        "n_both": n_valid,
        "pct_triggered": round(100 * sum(1 for r in results if r["rule_triggered"]) / N, 1),
        "MAE": round(np.mean(abs_errors), 2) if abs_errors else None,
        "MedAE": round(np.median(abs_errors), 2) if abs_errors else None,
        "std_AE": round(np.std(abs_errors), 2) if abs_errors else None,
        "min_AE": min(abs_errors) if abs_errors else None,
        "max_AE": max(abs_errors) if abs_errors else None,
        "within_5_pct": round(100 * sum(1 for r in valid if r["within_5"]) / n_valid, 1) if n_valid else 0,
        "within_10_pct": round(100 * sum(1 for r in valid if r["within_10"]) / n_valid, 1) if n_valid else 0,
        "early_trigger_pct": round(100 * sum(1 for r in valid if r["early_trigger"]) / n_valid, 1) if n_valid else 0,
        "late_trigger_pct": round(100 * sum(1 for r in valid if r["late_trigger"]) / n_valid, 1) if n_valid else 0,
    }

    # ── Per-task metrics ──
    per_task_metrics = {}
    for task, errs in sorted(per_task.items()):
        per_task_metrics[task] = {
            "n": len(errs),
            "MAE": round(np.mean(errs), 2),
            "MedAE": round(np.median(errs), 2),
            "within_5_pct": round(100 * sum(1 for e in errs if e <= 5) / len(errs), 1),
            "within_10_pct": round(100 * sum(1 for e in errs if e <= 10) / len(errs), 1),
        }

    # ── Write CSV ──
    csv_fields = ["episode_id", "task_name", "T_gform", "T_rule", "rule_triggered",
                  "error", "abs_error", "within_5", "within_10", "early_trigger", "late_trigger"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {len(results)} rows to {args.output_csv}")

    # ── Write Report ──
    report = f"""# Object Rule-Based Early-Grasp Trigger Baseline

**Date**: 2026-06-04
**Config**: K={args.close_K}, threshold={args.close_threshold} (CLOSE->OPEN transition)

---

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Total episodes | {N} |
| Rule triggered | {metrics['n_rule_triggered']} / {N} ({metrics['pct_triggered']}%) |
| Oracle labeled | {metrics['n_oracle_labeled']} |
| Both (for error calc) | {n_valid} |
| **MAE** | **{metrics['MAE']}** |
| **MedAE** | **{metrics['MedAE']}** |
| Std AE | {metrics['std_AE']} |
| Min / Max AE | {metrics['min_AE']} / {metrics['max_AE']} |
| Within 5 steps | {metrics['within_5_pct']}% |
| Within 10 steps | {metrics['within_10_pct']}% |
| Early trigger (>5 before) | {metrics['early_trigger_pct']}% |
| Late trigger (>5 after) | {metrics['late_trigger_pct']}% |

## Per-Task Metrics

| Task | n | MAE | MedAE | Within 5 | Within 10 |
|------|---|-----|-------|----------|-----------|
""" + "\n".join(
        f"| {t} | {m['n']} | {m['MAE']} | {m['MedAE']} | {m['within_5_pct']}% | {m['within_10_pct']}% |"
        for t, m in sorted(per_task_metrics.items())
    ) + f"""

## Interpretation

The rule-based baseline simply fires on the first sustained CLOSE->OPEN transition.
With K={args.close_K}, threshold={args.close_threshold}:

- MAE = {metrics['MAE']} steps — this is the average absolute error between
  rule-based trigger and oracle T_gform (from heuristic phase labeling).
- {metrics['within_5_pct']}% within 5 steps of oracle.
- {metrics['within_10_pct']}% within 10 steps of oracle.

This baseline is **mandatory** before training any learned detector.
A learned detector must demonstrate lower MAE and higher within-5/10 rates
to justify its additional complexity.

## Recommendation

"""

    if metrics["MAE"] is not None and metrics["MAE"] < 5:
        report += (
            "Rule-based baseline already achieves MAE < 5 steps. "
            "A learned detector is unlikely to add significant value for trigger timing. "
            "Focus on rule-based trigger + fixed delay attack pipeline."
        )
    elif metrics["MAE"] is not None and metrics["MAE"] < 15:
        report += (
            "Rule-based baseline achieves moderate accuracy (MAE 5-15 steps). "
            "A learned detector may improve timing, especially for tasks with "
            "high per-task variance. Train and compare."
        )
    else:
        report += (
            "Rule-based baseline has high error (MAE > 15 steps). "
            "A learned causal TCN detector is strongly justified. "
            "Proceed to training."
        )

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w") as f:
        f.write(report)
    print(f"Wrote report to {args.output_report}")


if __name__ == "__main__":
    main()
