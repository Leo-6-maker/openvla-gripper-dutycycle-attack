#!/usr/bin/env python3
"""Detector v3 diagnostic scaffold.

Dry-run validates labels_v3 readiness and prints planned comparisons. It does
not train unless a future explicit implementation is added.
"""

from __future__ import annotations

import argparse
import os


COMPARISONS = [
    "prevalence / always_positive",
    "task_key_only",
    "phase_only",
    "D_causal_safe LR",
    "D_causal_safe RF if available",
    "D_causal_safe + qpos_verified features if available",
    "task + phase + causal_safe diagnostic",
]
METRICS = ["balanced_accuracy", "macro_F1", "F1_pos", "F1_neg", "negative_recall", "positive_recall", "MCC", "FP count", "FN count", "LOTO if feasible", "source-batch breakdown", "hard-negative FP reduction vs v2"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v3_candidate.csv")
    ap.add_argument("--readiness-report", default="reports/OBJECT_PHASE_RESPONSE_LABEL_READINESS_V3_CANDIDATE.md")
    ap.add_argument("--output-metrics", default="tables/detector_v3_object_hardneg_metrics.csv")
    ap.add_argument("--output-predictions", default="tables/detector_v3_predictions.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V3_OBJECT_HARDNEG_DIAGNOSTIC.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def readiness_status(path):
    if not os.path.exists(path):
        return "BLOCKED_NOT_READY", "labels_v3 readiness report missing"
    text = open(path, encoding="utf-8").read()
    if "READY_FOR_DETECTOR_V3" in text:
        return "READY", "readiness report allows detector v3 diagnostic"
    return "BLOCKED_NOT_READY", "readiness report is not READY_FOR_DETECTOR_V3"


def main():
    args = parse_args()
    status, reason = readiness_status(args.readiness_report)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    lines = [
        "# Detector V3 Object Hard-Negative Diagnostic",
        "",
        f"**Status**: {status}",
        f"**Reason**: {reason}",
        f"**Labels CSV**: `{args.labels_csv}`",
        f"**Metrics output**: `{args.output_metrics}`",
        f"**Predictions output**: `{args.output_predictions}`",
        "",
        "This scaffold does not train detector v3. It is CPU-only dry-run support.",
        "",
        "## Required Comparisons",
        "",
    ]
    lines.extend(f"- {c}" for c in COMPARISONS)
    lines.extend(["", "## Metrics", ""])
    lines.extend(f"- {m}" for m in METRICS)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if args.dry_run:
        print(f"DRY RUN: detector v3 diagnostic status={status}; reason={reason}")
        for c in COMPARISONS:
            print(f"  comparison: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
