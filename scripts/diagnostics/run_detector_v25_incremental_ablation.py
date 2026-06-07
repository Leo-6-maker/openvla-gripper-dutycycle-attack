#!/usr/bin/env python3
"""Run detector v2.5 ablation by reusing the existing v1 diagnostic trainer.

The existing entrypoint is scripts/train_vulnerability_ready_detector_v1.py.
This wrapper prepares V0/V1 label CSVs and calls that script so metrics and LOTO
logic stay centralized. It is CPU-only and forces CUDA visibility off.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
SUMMARY_FIELDS = [
    "variant", "status", "labels_csv", "train_rows", "positive_rows",
    "negative_rows", "metrics_csv", "predictions_csv", "report_path",
    "sample_weight_status", "readiness_status", "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv", "--dataset", dest="dataset_csv", default="tables/detector_v25_incremental_dataset.csv")
    ap.add_argument("--existing-trainer", default="scripts/train_vulnerability_ready_detector_v1.py")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--work-dir", default="reports_temp/detector_v25_incremental_ablation")
    ap.add_argument("--output-metrics", default="tables/detector_v25_weighted_metrics.csv")
    ap.add_argument("--output-predictions", default="tables/detector_v25_weighted_predictions.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V25_WEIGHTED_SILVER_ABLATION.md")
    ap.add_argument("--min-rows", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_train(row):
    return lower(row.get("train_use")) == "true" and lower(row.get("label_status")) in {"positive", "negative"}


def is_gold(row):
    return lower(row.get("label_source")) == "gold_v2"


def is_silver_pos(row):
    return lower(row.get("label_source")) == "provisional_1r_positive"


def trainer_label(row, silver_weight=None):
    out = dict(row)
    out["label_status"] = lower(row.get("label_status"))
    out["label_vulnerability_ready"] = norm(row.get("label_vulnerability_ready"))
    if is_silver_pos(row) and silver_weight is not None:
        out["sample_weight"] = str(silver_weight)
    return out


def make_variant(rows, variant):
    train = [r for r in rows if is_train(r)]
    if variant == "V0_labels_v2_gold":
        selected = [trainer_label(r) for r in train if is_gold(r)]
        reason = "labels_v2 gold only"
        weight_column = ""
    elif variant == "V1_gold_plus_silver_positive_1r":
        selected = [trainer_label(r) for r in train if is_gold(r) or is_silver_pos(r)]
        reason = "gold plus provisional 1R positives; unweighted"
        weight_column = ""
    elif variant == "V1w_gold_plus_silver_positive_1r_weight05":
        selected = [trainer_label(r, 0.5) for r in train if is_gold(r) or is_silver_pos(r)]
        reason = "gold plus provisional 1R positives; sample_weight=0.5 for silver"
        weight_column = "sample_weight"
    elif variant == "V1w025_gold_plus_silver_positive_1r_weight025":
        selected = [trainer_label(r, 0.25) for r in train if is_gold(r) or is_silver_pos(r)]
        reason = "gold plus provisional 1R positives; sample_weight=0.25 for silver"
        weight_column = "sample_weight"
    else:
        selected = []
        reason = "unknown variant"
        weight_column = ""
    return selected, reason, weight_column


def count_labels(rows):
    pos = sum(1 for r in rows if norm(r.get("label_vulnerability_ready")) == "1")
    neg = sum(1 for r in rows if norm(r.get("label_vulnerability_ready")) == "0")
    return pos, neg


def readiness(rows):
    hard_neg = sum(
        1 for r in rows
        if is_train(r) and norm(r.get("label_vulnerability_ready")) == "0"
        and "hard_negative" in " ".join(lower(r.get(f)) for f in ["candidate_role", "expected_role", "source_error_type"])
    )
    if len([r for r in rows if is_train(r)]) < 30:
        return "EXPLORATORY_ONLY_UNDERPOWERED"
    if len([r for r in rows if is_train(r) and is_gold(r)]) >= 35 and hard_neg >= 6:
        return "V3_DIAGNOSTIC_POSSIBLE_IF_CALIBRATION_PASSES"
    return "NOT_READY_FOR_V3"


def run_trainer(args, labels_csv, metrics_csv, preds_csv, report_path, weight_column=""):
    cmd = [
        sys.executable, args.existing_trainer,
        "--labels-csv", labels_csv,
        "--descriptors", args.descriptors,
        "--output-metrics", metrics_csv,
        "--output-predictions", preds_csv,
        "--output-report", report_path,
        "--min-rows", str(args.min_rows),
    ]
    if weight_column:
        cmd.extend(["--sample-weight-column", weight_column])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(cmd, cwd=os.getcwd(), env=env, text=True, capture_output=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def copy_if_exists(src, variant, out_rows):
    if not os.path.exists(src):
        return
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new = {norm(k).lstrip("\ufeff"): v for k, v in row.items()}
            new["variant"] = variant
            out_rows.append(new)


def _metric_float(row, field):
    try:
        return float(row.get(field) or 0)
    except Exception:
        return 0.0


def _best_rows(metrics):
    out = []
    for variant in sorted({r.get("variant", "") for r in metrics}):
        rows = [r for r in metrics if r.get("variant") == variant and r.get("balanced_accuracy") not in ("", None)]
        if not rows:
            continue
        best = max(rows, key=lambda r: _metric_float(r, "balanced_accuracy"))
        d_lr = next((r for r in rows if r.get("feature_set") == "D_causal_safe" and r.get("model") == "LR"), None)
        out.append(("best", best))
        if d_lr is not None and d_lr is not best:
            out.append(("D_causal_safe_LR", d_lr))
    return out


def write_report(path, summaries, notes, metrics=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Detector V2.5 Incremental Ablation",
        "",
        "This wrapper reuses `scripts/train_vulnerability_ready_detector_v1.py` for metrics and LOTO.",
        "No GPU, VIS, rollout, watcher, or final detector training was run.",
        "",
        "## Existing Training Entrypoint",
        "",
        "- Entrypoint: `scripts/train_vulnerability_ready_detector_v1.py`",
        "- Expected input CSV: rows with `label_status in {positive, negative}` and `label_vulnerability_ready in {0,1}`.",
        "- Feature inputs: `task_key`, `phase_bin_proxy`, and descriptor CSV fields such as `qpos_start`, `qpos_min`, `clean_open_ratio`, `eef_speed_mean`.",
        "- Output metrics/predictions are supplied by this wrapper per variant.",
        "- `sample_weight`: supported for LR/RF when `--sample-weight-column sample_weight` is passed; baselines remain unweighted evaluation rows.",
        "- LOTO: supported through `LeaveOneGroupOut` grouped by `task_key`, with warnings for invalid folds.",
        "",
        "## Variants",
        "",
        "| Variant | Status | Rows | Pos | Neg | Weighting | Readiness | Reason |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['status']} | {row['train_rows']} | "
            f"{row['positive_rows']} | {row['negative_rows']} | {row['sample_weight_status']} | {row['readiness_status']} | {row['reason']} |"
        )
    lines.extend([
        "",
        "## Key Questions",
        "",
        "- Compare weighted variants against V0 and unweighted V1 on balanced accuracy, negative recall, FP, and FN.",
        "- Treat any V1 gain without FP or negative-recall improvement as likely positive-class skew.",
    ])
    metric_rows = _best_rows(metrics or [])
    lines.extend(["", "## Metric Summary", ""])
    if metric_rows:
        lines.extend(["| Kind | Variant | Feature set | Model | BAcc | Macro F1 | Neg recall | FP | FN | MCC |",
                      "|---|---|---|---|---:|---:|---:|---:|---:|---:|"])
        for kind, row in metric_rows:
            lines.append(
                f"| {kind} | {row.get('variant','')} | {row.get('feature_set','')} | {row.get('model','')} | "
                f"{row.get('balanced_accuracy','')} | {row.get('macro_F1','')} | {row.get('negative_recall','')} | "
                f"{row.get('fp','')} | {row.get('fn','')} | {row.get('MCC','')} |"
            )
    else:
        lines.append("- Metrics unavailable.")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in notes) if notes else lines.append("- None.")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- V2.5 is exploratory.",
        "- V3 remains blocked until labels_v3/v4 readiness passes.",
        "- `pending_negative_1r` is not used as a train negative.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    rows = read_csv(args.dataset_csv)
    notes = []
    precheck_blockers = []
    if not rows:
        precheck_blockers.append(f"BLOCKED_MISSING_OR_EMPTY_DATASET: {args.dataset_csv}")
    if not os.path.exists(args.existing_trainer):
        precheck_blockers.append(f"BLOCKED_MISSING_EXISTING_TRAINER: {args.existing_trainer}")
    if any(lower(r.get("label_source")) == "pending_negative_1r" and lower(r.get("train_use")) == "true" for r in rows):
        precheck_blockers.append("HARD_FAIL_PENDING_NEGATIVE_1R_IN_TRAIN")
    notes.extend(precheck_blockers)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    summaries = []
    combined_metrics = []
    combined_predictions = []
    for variant in [
        "V0_labels_v2_gold",
        "V1_gold_plus_silver_positive_1r",
        "V1w_gold_plus_silver_positive_1r_weight05",
        "V1w025_gold_plus_silver_positive_1r_weight025",
    ]:
        selected, reason, weight_column = make_variant(rows, variant)
        pos, neg = count_labels(selected)
        labels_csv = str(work / f"{variant}_labels.csv")
        metrics_csv = str(work / f"{variant}_metrics.csv")
        preds_csv = str(work / f"{variant}_predictions.csv")
        report_path = str(work / f"{variant}_report.md")
        fields = sorted({k for row in selected for k in row.keys()} | {"label_status", "label_vulnerability_ready"})
        write_csv(labels_csv, fields, selected)
        status = "NOT_RUN"
        if not selected:
            status = "BLOCKED_NO_ROWS"
        elif pos == 0 or neg == 0:
            status = "BLOCKED_SINGLE_CLASS"
        elif len(selected) < args.min_rows:
            status = f"BLOCKED_MIN_ROWS_LT_{args.min_rows}"
        elif precheck_blockers:
            status = "BLOCKED_PRECHECK"
        elif args.dry_run:
            status = "DRY_RUN_READY"
        else:
            code, stdout, stderr = run_trainer(args, labels_csv, metrics_csv, preds_csv, report_path, weight_column)
            status = "OK" if code == 0 else f"TRAINER_EXIT_{code}"
            if stdout:
                notes.append(f"{variant} stdout: {stdout[:1000]}")
            if stderr:
                notes.append(f"{variant} stderr: {stderr[:1000]}")
            copy_if_exists(metrics_csv, variant, combined_metrics)
            copy_if_exists(preds_csv, variant, combined_predictions)
        summaries.append({
            "variant": variant,
            "status": status,
            "labels_csv": labels_csv,
            "train_rows": str(len(selected)),
            "positive_rows": str(pos),
            "negative_rows": str(neg),
            "metrics_csv": metrics_csv,
            "predictions_csv": preds_csv,
            "report_path": report_path,
            "sample_weight_status": "weighted:" + weight_column if weight_column else "unweighted",
            "readiness_status": readiness(rows),
            "reason": reason,
        })
    metric_fields = sorted({k for row in combined_metrics for k in row.keys()} | {"variant"})
    pred_fields = sorted({k for row in combined_predictions for k in row.keys()} | {"variant"})
    write_csv(args.output_metrics, metric_fields or ["variant"], combined_metrics)
    write_csv(args.output_predictions, pred_fields or ["variant"], combined_predictions)
    write_report(args.output_report, summaries, notes, combined_metrics)
    if args.dry_run:
        print(f"DRY RUN: variants={len(summaries)} notes={len(notes)}")
        for row in summaries:
            print(f"{row['variant']}: {row['status']} rows={row['train_rows']} pos={row['positive_rows']} neg={row['negative_rows']}")
        for note in notes:
            print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
