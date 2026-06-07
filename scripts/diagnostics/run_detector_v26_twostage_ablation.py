#!/usr/bin/env python3
"""Run detector-v2.6/two-stage exploratory ablation using existing trainer."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
METRIC_FIELDS = [
    "variant", "mode", "feature_set", "model", "status", "train_rows",
    "positive_rows", "negative_rows", "balanced_accuracy", "macro_F1",
    "f1_pos", "f1_neg", "negative_recall", "false_positive_rate", "rec_pos", "MCC",
    "tp", "fp", "fn", "tn", "phase_gate_recall_pos",
    "phase_gate_rejection_clean_controls", "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tables/detector_v26_twostage_dataset.csv")
    ap.add_argument("--existing-trainer", default="scripts/train_vulnerability_ready_detector_v1.py")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--work-dir", default="reports_temp/detector_v26_twostage_ablation")
    ap.add_argument("--output-metrics", default="tables/detector_v26_twostage_metrics.csv")
    ap.add_argument("--output-predictions", default="tables/detector_v26_twostage_predictions.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V26_TWOSTAGE_ABLATION.md")
    ap.add_argument("--min-rows", type=int, default=15)
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def is_train(row):
    return lower(row.get("train_use")).startswith("true") and lower(row.get("label_status")) in {"positive", "negative"}


def label(row):
    return norm(row.get("label_vulnerability_ready"))


def row_key(row):
    return tuple(norm(row.get(f)) for f in KEY_FIELDS)


def variant_rows(rows, variant):
    train = [r for r in rows if is_train(r)]
    if variant == "V0_gold_only":
        return [r for r in train if lower(r.get("label_source")) == "gold_v2"], "gold only"
    if variant == "V1_gold_plus_silver":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "eligible_silver_positive_1r"}], "gold plus eligible silver"
    if variant == "V2_gold_plus_clean_controls":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "clean_control_negative"}], "gold plus clean controls"
    if variant == "V3_gold_plus_silver_plus_clean_controls":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "eligible_silver_positive_1r", "clean_control_negative"}], "gold plus silver plus controls"
    return [], "unsupported"


def counts(rows):
    pos = sum(1 for r in rows if label(r) == "1")
    neg = sum(1 for r in rows if label(r) == "0")
    return pos, neg


def run_trainer(args, variant, selected):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    labels_csv = str(work / f"{variant}_labels.csv")
    metrics_csv = str(work / f"{variant}_metrics.csv")
    preds_csv = str(work / f"{variant}_predictions.csv")
    report_path = str(work / f"{variant}_report.md")
    fields = sorted({k for r in selected for k in r.keys()} | {"label_status", "label_vulnerability_ready", "sample_weight"})
    write_csv(labels_csv, fields, selected)
    cmd = [
        sys.executable, args.existing_trainer,
        "--labels-csv", labels_csv,
        "--descriptors", args.descriptors,
        "--output-metrics", metrics_csv,
        "--output-predictions", preds_csv,
        "--output-report", report_path,
        "--min-rows", str(args.min_rows),
        "--sample-weight-column", "sample_weight",
    ]
    proc = subprocess.run(cmd, cwd=os.getcwd(), text=True, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return proc.returncode, metrics_csv, preds_csv, proc.stdout.strip(), proc.stderr.strip()


def add_metrics_from_file(path, variant, rows_out, status, reason):
    if not os.path.exists(path):
        rows_out.append({"variant": variant, "mode": "vulnerability_only", "status": status, "reason": reason})
        return
    for row in read_csv(path):
        if norm(row.get("fp")) != "" and norm(row.get("tn")) != "":
            try:
                fp = float(row.get("fp") or 0)
                tn = float(row.get("tn") or 0)
                row["false_positive_rate"] = "" if fp + tn == 0 else fp / (fp + tn)
            except Exception:
                row["false_positive_rate"] = ""
        row["variant"] = variant
        row["mode"] = "vulnerability_only"
        row["status"] = status
        row["reason"] = reason
        rows_out.append(row)


def add_predictions_from_file(path, variant, rows_out):
    for row in read_csv(path):
        row["variant"] = variant
        rows_out.append(row)


def _metric_values(y_true, y_pred):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    def div(a, b):
        return 0.0 if b == 0 else a / float(b)

    rec_pos = div(tp, tp + fn)
    rec_neg = div(tn, tn + fp)
    prec_pos = div(tp, tp + fp)
    prec_neg = div(tn, tn + fn)
    f1_pos = div(2 * prec_pos * rec_pos, prec_pos + rec_pos)
    f1_neg = div(2 * prec_neg * rec_neg, prec_neg + rec_neg)
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = 0.0 if denom == 0 else ((tp * tn) - (fp * fn)) / denom
    return {
        "balanced_accuracy": (rec_pos + rec_neg) / 2.0,
        "macro_F1": (f1_pos + f1_neg) / 2.0,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "negative_recall": rec_neg,
        "false_positive_rate": div(fp, fp + tn),
        "rec_pos": rec_pos,
        "MCC": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def phase_coverage(rows):
    train = [r for r in rows if is_train(r)]
    if not train:
        return False, "no train rows"
    missing = [r for r in train if norm(r.get("phase_is_critical")) not in {"0", "1"}]
    if missing:
        return False, "missing phase_is_critical for %d/%d train rows" % (len(missing), len(train))
    return True, "complete phase_is_critical coverage"


def phase_complete_rows(rows):
    return [r for r in rows if is_train(r) and norm(r.get("phase_is_critical")) in {"0", "1"}]


def phase_gate_metrics(rows):
    train = [r for r in rows if is_train(r)]
    ok, reason = phase_coverage(train)
    if not ok:
        return None
    pos = [r for r in train if label(r) == "1"]
    controls = [r for r in train if lower(r.get("label_source")) == "clean_control_negative"]
    def crit(r):
        return norm(r.get("phase_is_critical")) == "1"
    pos_recall = "" if not pos else sum(1 for r in pos if crit(r)) / float(len(pos))
    ctrl_reject = "" if not controls else sum(1 for r in controls if not crit(r)) / float(len(controls))
    return pos_recall, ctrl_reject


def add_phase_only_metrics(rows, metrics):
    ok, reason = phase_coverage(rows)
    train = [r for r in rows if is_train(r)]
    if not ok:
        metrics.append({"variant": "V4_phase_gate_only", "mode": "phase_gate", "status": "BLOCKED_MISSING_PHASE_OUTPUTS", "reason": reason})
        complete = phase_complete_rows(rows)
        if not complete:
            return False, reason
        y_true = [int(label(r)) for r in complete]
        y_pred = [1 if norm(r.get("phase_is_critical")) == "1" else 0 for r in complete]
        row = _metric_values(y_true, y_pred)
        row.update({
            "variant": "V4_phase_gate_only_complete_case",
            "mode": "phase_gate",
            "feature_set": "phase_is_critical",
            "model": "rule",
            "status": "DIAGNOSTIC_COMPLETE_CASE",
            "train_rows": len(complete),
            "positive_rows": sum(y_true),
            "negative_rows": len(y_true) - sum(y_true),
            "reason": reason + "; excluded rows with missing phase_is_critical",
        })
        metrics.append(row)
        return False, reason
    y_true = [int(label(r)) for r in train]
    y_pred = [1 if norm(r.get("phase_is_critical")) == "1" else 0 for r in train]
    row = _metric_values(y_true, y_pred)
    row.update({
        "variant": "V4_phase_gate_only",
        "mode": "phase_gate",
        "feature_set": "phase_is_critical",
        "model": "rule",
        "status": "DIAGNOSTIC_ONLY",
        "train_rows": len(train),
        "positive_rows": sum(y_true),
        "negative_rows": len(y_true) - sum(y_true),
        "reason": reason,
    })
    metrics.append(row)
    return True, reason


def add_two_stage_metrics(rows, predictions, metrics):
    ok, reason = phase_coverage(rows)
    if not ok:
        metrics.append({"variant": "V6_two_stage_predicted_phase", "mode": "phase_plus_vulnerability", "status": "BLOCKED_MISSING_PHASE_OUTPUTS", "reason": reason})
    phase_by_key = {row_key(r): int(norm(r.get("phase_is_critical")) == "1") for r in phase_complete_rows(rows)}
    grouped = {}
    for pred in predictions:
        if pred.get("variant") != "V3_gold_plus_silver_plus_clean_controls":
            continue
        key = row_key(pred)
        if key not in phase_by_key:
            continue
        grouped.setdefault((pred.get("feature_set", ""), pred.get("model", "")), []).append(pred)
    if not grouped:
        metrics.append({"variant": "V6_two_stage_predicted_phase", "mode": "phase_plus_vulnerability", "status": "BLOCKED_NO_VULNERABILITY_PREDICTIONS", "reason": "V3 predictions unavailable"})
        return
    for (feature_set, model), preds in sorted(grouped.items()):
        y_true, y_pred = [], []
        for pred in preds:
            y_true.append(int(float(norm(pred.get("true")) or 0)))
            vuln_pred = int(float(norm(pred.get("pred")) or 0))
            y_pred.append(1 if vuln_pred == 1 and phase_by_key.get(row_key(pred), 0) == 1 else 0)
        row = _metric_values(y_true, y_pred)
        row.update({
            "variant": "V6_two_stage_predicted_phase" if ok else "V6_two_stage_predicted_phase_complete_case",
            "mode": "phase_plus_vulnerability",
            "feature_set": feature_set,
            "model": model,
            "status": "DIAGNOSTIC_ONLY" if ok else "DIAGNOSTIC_COMPLETE_CASE",
            "train_rows": len(y_true),
            "positive_rows": sum(y_true),
            "negative_rows": len(y_true) - sum(y_true),
            "reason": "V3 predicted vulnerability AND phase_is_critical rule" if ok else reason + "; complete-case V3 predicted vulnerability AND phase_is_critical rule",
        })
        metrics.append(row)


def write_report(path, metric_rows, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Detector V2.6 Two-Stage Ablation",
        "",
        "Exploratory CPU-only diagnostic. No GPU, VIS, rollout, watcher, or final detector training was run.",
        "",
        "## Variant Summary",
        "",
        "| Variant | Status | Best BAcc | Best model | Neg recall | FPR | FP | FN |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for variant in sorted({r.get("variant", "") for r in metric_rows}):
        rows = [r for r in metric_rows if r.get("variant") == variant and r.get("balanced_accuracy")]
        if not rows:
            r0 = next((r for r in metric_rows if r.get("variant") == variant), {})
            lines.append(f"| {variant} | {r0.get('status','BLOCKED')} |  |  |  |  |  |  |")
            continue
        best = max(rows, key=lambda r: float(r.get("balanced_accuracy") or 0))
        lines.append(f"| {variant} | {best.get('status','')} | {best.get('balanced_accuracy','')} | {best.get('feature_set','')}/{best.get('model','')} | {best.get('negative_recall','')} | {best.get('false_positive_rate','')} | {best.get('fp','')} | {best.get('fn','')} |")
    lines.extend(["", "## Two-Stage / Phase Gate", ""])
    if any("BLOCKED_MISSING_PHASE_OUTPUTS" in r.get("status", "") for r in metric_rows):
        lines.append("- Phase-gated variants are blocked because no explicit phase detector outputs or usable phase confidence were available.")
    if any(r.get("status") == "DIAGNOSTIC_COMPLETE_CASE" for r in metric_rows):
        lines.append("- Complete-case phase diagnostics are reported separately after excluding rows with missing `phase_is_critical`; these are not full-dataset two-stage results.")
    if not any("BLOCKED_MISSING_PHASE_OUTPUTS" in r.get("status", "") for r in metric_rows):
        lines.append("- Phase gate diagnostics are available in metric rows.")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {n}" for n in notes) if notes else lines.append("- None.")
    lines.extend([
        "",
        "## Readiness",
        "",
        "Detector v3/final training remains blocked. V2.6 suggests candidate clean controls can improve negative recall/FPR in an exploratory ablation, but control labels are candidate-derived, silver-positive quality audit is missing, phase metadata is incomplete, and no oracle phase labels are available.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    rows = read_csv(args.dataset)
    notes = []
    metrics, predictions = [], []
    for variant in ["V0_gold_only", "V1_gold_plus_silver", "V2_gold_plus_clean_controls", "V3_gold_plus_silver_plus_clean_controls"]:
        selected, reason = variant_rows(rows, variant)
        pos, neg = counts(selected)
        if not selected:
            metrics.append({"variant": variant, "mode": "vulnerability_only", "status": "BLOCKED_NO_ROWS", "train_rows": "0", "reason": reason})
            continue
        if pos == 0 or neg == 0:
            metrics.append({"variant": variant, "mode": "vulnerability_only", "status": "BLOCKED_SINGLE_CLASS", "train_rows": str(len(selected)), "positive_rows": str(pos), "negative_rows": str(neg), "reason": reason})
            continue
        if len(selected) < args.min_rows:
            metrics.append({"variant": variant, "mode": "vulnerability_only", "status": "BLOCKED_MIN_ROWS", "train_rows": str(len(selected)), "positive_rows": str(pos), "negative_rows": str(neg), "reason": reason})
            continue
        code, mpath, ppath, stdout, stderr = run_trainer(args, variant, selected)
        status = "OK" if code == 0 else f"TRAINER_EXIT_{code}"
        if stdout:
            notes.append(f"{variant} stdout: {stdout[:500]}")
        if stderr:
            notes.append(f"{variant} stderr: {stderr[:500]}")
        add_metrics_from_file(mpath, variant, metrics, status, reason)
        add_predictions_from_file(ppath, variant, predictions)
    phase_ok, phase_reason = add_phase_only_metrics(rows, metrics)
    if phase_ok:
        pg = phase_gate_metrics(rows)
        if pg is not None:
            for row in metrics:
                if row.get("variant") == "V4_phase_gate_only":
                    row["phase_gate_recall_pos"] = pg[0]
                    row["phase_gate_rejection_clean_controls"] = pg[1]
    metrics.append({
        "variant": "V5_two_stage_oracle_phase",
        "mode": "phase_plus_vulnerability",
        "status": "NOT_RUN_NO_ORACLE_PHASE_LABELS",
        "reason": "No oracle phase labels are available in the snapshot; avoiding an oracle claim.",
    })
    add_two_stage_metrics(rows, predictions, metrics)
    metric_fields = sorted(set(METRIC_FIELDS) | {k for r in metrics for k in r.keys()})
    pred_fields = sorted({k for r in predictions for k in r.keys()} | {"variant"})
    write_csv(args.output_metrics, metric_fields, metrics)
    write_csv(args.output_predictions, pred_fields or ["variant"], predictions)
    write_report(args.output_report, metrics, notes)
    print(f"metric_rows={len(metrics)} prediction_rows={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
