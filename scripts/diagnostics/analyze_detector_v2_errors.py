#!/usr/bin/env python3
"""Analyze detector v2 prediction errors for Batch4 hard-negative planning.

CPU-only. This script reads labels/predictions, writes a compact error table,
and does not train detector models or run rollout/VIS.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
LABEL_CANDIDATES = [
    "label_vulnerability_ready",
    "y_true",
    "true_label",
    "label",
    "target",
    "vulnerability_ready",
]
PRED_CANDIDATES = [
    "y_pred",
    "pred_label",
    "prediction",
    "predicted_label",
    "label_pred",
    "vulnerability_ready_pred",
]
SCORE_CANDIDATES = ["score", "probability", "pred_score", "positive_score", "prob_positive"]
OUTPUT_FIELDS = [
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "source_batch",
    "candidate_role",
    "phase_bin_proxy",
    "denominator_type",
    "provenance_status",
    "label_status",
    "y_true",
    "y_pred",
    "score",
    "error_type",
    "priority",
    "recommended_batch4_role",
    "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--predictions-csv", default="tables/detector_v2_predictions.csv")
    ap.add_argument("--metrics-csv", default="tables/detector_v2_metrics.csv")
    ap.add_argument("--output-csv", default="tables/detector_v2_error_analysis.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V2_ERROR_ANALYSIS.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def parse_binary(value):
    v = lower(value)
    if v in {"1", "true", "yes", "y", "positive", "pos"}:
        return 1
    if v in {"0", "false", "no", "n", "negative", "neg"}:
        return 0
    return None


def parse_float(value):
    try:
        text = norm(value)
        if text == "":
            return ""
        return f"{float(text):.6g}"
    except (TypeError, ValueError):
        return norm(value)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(field).lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({norm(k).lstrip("\ufeff"): v for k, v in row.items()})
        return fields, rows


def key(row):
    return tuple(norm(row.get(field)) for field in KEY_FIELDS)


def first_present(row, candidates):
    for field in candidates:
        if field in row and norm(row.get(field)) != "":
            return row.get(field)
    return ""


def row_metadata(row):
    return {
        "source_batch": norm(row.get("source_batch")),
        "candidate_role": norm(row.get("candidate_role")),
        "phase_bin_proxy": norm(row.get("phase_bin_proxy")),
        "denominator_type": norm(row.get("denominator_type")),
        "provenance_status": norm(row.get("provenance_status")),
        "label_status": norm(row.get("label_status")),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, status, rows, notes, metrics_rows=0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = Counter(row["error_type"] for row in rows)
    fp_count = counts.get("FP", 0)
    fn_count = counts.get("FN", 0)
    lines = [
        "# Detector V2 Error Analysis",
        "",
        f"**Status**: {status}",
        f"**Labels CSV**: `{args.labels_csv}`",
        f"**Predictions CSV**: `{args.predictions_csv}`",
        f"**Metrics CSV**: `{args.metrics_csv}`",
        f"**Metrics rows**: {metrics_rows}",
        f"**Error rows written**: {len(rows)}",
        f"**False positives**: {fp_count}",
        f"**False negatives**: {fn_count}",
        "",
        "This is a CPU-only analysis. It does not run rollout, VIS, GPU work, or detector training.",
        "",
        "## Blocking / Review Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.extend(["", "## Error Rows", ""])
    if rows:
        lines.extend(
            [
                "| Error | Task | State | Window | Source | Role | Phase | Score | Batch4 role |",
                "|---|---|---:|---|---|---|---|---:|---|",
            ]
        )
        for row in rows:
            window = f"{row['window_start']}-{row['window_end']}"
            lines.append(
                f"| {row['error_type']} | {row['task_key']} | {row['state_id']} | {window} | "
                f"{row['source_batch']} | {row['candidate_role']} | {row['phase_bin_proxy']} | "
                f"{row['score']} | {row['recommended_batch4_role']} |"
            )
    else:
        lines.append("- No error rows available.")
    lines.extend(
        [
            "",
            "## Expected Gate",
            "",
            "- The current Batch4 planning assumption expects 6 detector-v2 false positives and 1 false negative.",
            "- If the observed count differs, Batch4 candidates should be reviewed before any server-side execution.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    if not os.path.exists(args.labels_csv) or not os.path.exists(args.predictions_csv):
        if not os.path.exists(args.labels_csv):
            notes.append(f"BLOCKED_MISSING_INPUT: labels CSV not found: {args.labels_csv}")
        if not os.path.exists(args.predictions_csv):
            notes.append(f"BLOCKED_MISSING_INPUT: predictions CSV not found: {args.predictions_csv}")
        write_csv(args.output_csv, [])
        write_report(args.output_report, args, "BLOCKED_MISSING_INPUTS", [], notes, 0)
        return 0

    label_fields, label_rows = read_csv(args.labels_csv)
    pred_fields, pred_rows = read_csv(args.predictions_csv)
    metrics_rows = 0
    if os.path.exists(args.metrics_csv):
        _, metrics = read_csv(args.metrics_csv)
        metrics_rows = len(metrics)
    else:
        notes.append(f"metrics CSV not found: {args.metrics_csv}")

    labels_by_key = {key(row): row for row in label_rows}
    missing_keys = []
    errors = []
    for pred in pred_rows:
        joined = dict(labels_by_key.get(key(pred), {}))
        joined.update({k: v for k, v in pred.items() if norm(v) != ""})
        if not labels_by_key.get(key(pred)):
            missing_keys.append(key(pred))

        y_true = parse_binary(first_present(joined, LABEL_CANDIDATES))
        y_pred = parse_binary(first_present(pred, PRED_CANDIDATES))
        if y_pred is None:
            y_pred = parse_binary(first_present(joined, PRED_CANDIDATES))
        if y_true is None or y_pred is None:
            notes.append(f"unparseable label/prediction at key={key(pred)}")
            continue
        if y_true == y_pred:
            continue

        error_type = "FP" if y_true == 0 and y_pred == 1 else "FN"
        meta = row_metadata(joined)
        out = {
            "task_key": norm(joined.get("task_key")),
            "state_id": norm(joined.get("state_id")),
            "window_start": norm(joined.get("window_start")),
            "window_end": norm(joined.get("window_end")),
            **meta,
            "y_true": str(y_true),
            "y_pred": str(y_pred),
            "score": parse_float(first_present(joined, SCORE_CANDIDATES)),
            "error_type": error_type,
            "priority": "1" if error_type == "FP" else "2",
            "recommended_batch4_role": "hard_negative_fp_control" if error_type == "FP" else "missed_positive_followup",
            "reason": "detector_false_positive_hard_negative" if error_type == "FP" else "detector_false_negative_positive_gap",
        }
        errors.append(out)

    errors.sort(key=lambda r: (int(r["priority"]), r["task_key"], int(float(r["state_id"] or 0)), int(float(r["window_start"] or 0))))
    if missing_keys:
        notes.append(f"prediction rows missing matching labels: {len(missing_keys)}")
    fp_count = sum(1 for row in errors if row["error_type"] == "FP")
    fn_count = sum(1 for row in errors if row["error_type"] == "FN")
    status = "PASS_EXPECTED_ERROR_COUNTS" if fp_count == 6 and fn_count == 1 else "COUNT_MISMATCH_REVIEW_NEEDED"
    if fp_count != 6 or fn_count != 1:
        notes.append(f"expected 6 FP and 1 FN, observed FP={fp_count}, FN={fn_count}")

    write_csv(args.output_csv, errors)
    write_report(args.output_report, args, status, errors, notes, metrics_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
