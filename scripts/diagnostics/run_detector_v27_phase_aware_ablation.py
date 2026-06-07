#!/usr/bin/env python3
"""Run detector-v2.7 phase-aware exploratory ablation."""

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
    "variant",
    "eval_scope",
    "mode",
    "feature_set",
    "model",
    "status",
    "train_rows",
    "positive_rows",
    "negative_rows",
    "balanced_accuracy",
    "macro_F1",
    "f1_pos",
    "f1_neg",
    "negative_recall",
    "positive_recall",
    "false_positive_rate",
    "MCC",
    "tp",
    "fp",
    "fn",
    "tn",
    "phase_gate_positive_recall",
    "phase_gate_control_rejection_rate",
    "threshold",
    "missing_phase_policy",
    "reason",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tables/detector_v27_phase_aware_dataset.csv")
    ap.add_argument("--existing-trainer", default="scripts/train_vulnerability_ready_detector_v1.py")
    ap.add_argument("--descriptors", default="tables/object_teacher_window_phase_descriptors.csv")
    ap.add_argument("--work-dir", default="reports_temp/detector_v27_phase_aware")
    ap.add_argument("--output-metrics", default="tables/detector_v27_phase_aware_metrics.csv")
    ap.add_argument("--output-predictions", default="tables/detector_v27_phase_aware_predictions.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V27_PHASE_AWARE_TWOSTAGE_ABLATION.md")
    ap.add_argument("--min-rows", type=int, default=15)
    return ap.parse_args()


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def lower(value) -> str:
    return norm(value).lower()


def read_csv(path: str) -> list[dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): norm(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: str, fields: list[str], rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(norm(row.get(f)) for f in KEY_FIELDS)


def is_train(row: dict[str, str]) -> bool:
    return lower(row.get("train_use")).startswith("true") and lower(row.get("label_status")) in {"positive", "negative"}


def y_value(row: dict[str, str]) -> int:
    return 1 if norm(row.get("label_vulnerability_ready")) == "1" else 0


def variant_rows(rows: list[dict[str, str]], variant: str) -> list[dict[str, str]]:
    train = [r for r in rows if is_train(r)]
    if variant == "V0_gold_only":
        return [r for r in train if lower(r.get("label_source")) == "gold_v2"]
    if variant == "V1_gold_plus_silver":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "eligible_1r_silver_positive", "eligible_silver_positive_1r"}]
    if variant == "V2_gold_plus_clean_controls":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "clean_control_negative"}]
    if variant == "V3_gold_plus_silver_plus_clean_controls":
        return [r for r in train if lower(r.get("label_source")) in {"gold_v2", "eligible_1r_silver_positive", "eligible_silver_positive_1r", "clean_control_negative"}]
    return []


def metric_values(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    def div(a: float, b: float) -> float:
        return 0.0 if b == 0 else a / b

    rec_pos = div(tp, tp + fn)
    rec_neg = div(tn, tn + fp)
    prec_pos = div(tp, tp + fp)
    prec_neg = div(tn, tn + fn)
    f1_pos = div(2 * prec_pos * rec_pos, prec_pos + rec_pos)
    f1_neg = div(2 * prec_neg * rec_neg, prec_neg + rec_neg)
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "balanced_accuracy": (rec_pos + rec_neg) / 2.0,
        "macro_F1": (f1_pos + f1_neg) / 2.0,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "negative_recall": rec_neg,
        "positive_recall": rec_pos,
        "false_positive_rate": div(fp, fp + tn),
        "MCC": 0.0 if denom == 0 else ((tp * tn) - (fp * fn)) / denom,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_trainer(args: argparse.Namespace, variant: str, selected: list[dict[str, str]]) -> tuple[int, str, str, str, str]:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    labels_csv = str(work / f"{variant}_labels.csv")
    metrics_csv = str(work / f"{variant}_metrics.csv")
    preds_csv = str(work / f"{variant}_predictions.csv")
    report_path = str(work / f"{variant}_report.md")
    fields = sorted({k for r in selected for k in r.keys()} | {"label_status", "label_vulnerability_ready", "sample_weight"})
    write_csv(labels_csv, fields, selected)
    cmd = [
        sys.executable,
        args.existing_trainer,
        "--labels-csv",
        labels_csv,
        "--descriptors",
        args.descriptors,
        "--output-metrics",
        metrics_csv,
        "--output-predictions",
        preds_csv,
        "--output-report",
        report_path,
        "--min-rows",
        str(args.min_rows),
        "--sample-weight-column",
        "sample_weight",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return proc.returncode, metrics_csv, preds_csv, proc.stdout.strip(), proc.stderr.strip()


def add_trainer_metrics(path: str, variant: str, rows_out: list[dict[str, object]], status: str) -> None:
    for row in read_csv(path):
        if norm(row.get("fp")) and norm(row.get("tn")):
            fp = float(row.get("fp") or 0)
            tn = float(row.get("tn") or 0)
            row["false_positive_rate"] = "" if fp + tn == 0 else fp / (fp + tn)
        if norm(row.get("rec_pos")) and not norm(row.get("positive_recall")):
            row["positive_recall"] = row.get("rec_pos")
        row.update({"variant": variant, "eval_scope": "full", "mode": "vulnerability_only", "status": status})
        rows_out.append(row)


def add_trainer_predictions(path: str, variant: str, rows_out: list[dict[str, object]]) -> None:
    for row in read_csv(path):
        row.update({"variant": variant, "mode": "vulnerability_only", "eval_scope": "full"})
        rows_out.append(row)


def phase_pred(row: dict[str, str], missing_policy: str) -> int | None:
    phase = lower(row.get("phase_is_critical"))
    if phase in {"true", "1", "critical"}:
        return 1
    if phase in {"false", "0", "noncritical", "non-critical"}:
        return 0
    if missing_policy == "unknown_pass":
        return 1
    if missing_policy == "unknown_reject":
        return 0
    return None


def phase_conf(row: dict[str, str], missing_policy: str) -> float | None:
    try:
        if norm(row.get("phase_confidence")):
            return float(row.get("phase_confidence"))
    except Exception:
        pass
    pred = phase_pred(row, missing_policy)
    if pred is None:
        return None
    return float(pred)


def rows_for_scope(rows: list[dict[str, str]], scope: str) -> list[dict[str, str]]:
    train = [r for r in rows if is_train(r)]
    if scope == "complete_case":
        return [r for r in train if phase_pred(r, "drop") is not None]
    return train


def add_phase_gate_metrics(rows: list[dict[str, str]], metrics: list[dict[str, object]], predictions: list[dict[str, object]]) -> None:
    for scope, missing_policy in [("full_unknown_pass", "unknown_pass"), ("complete_case", "drop")]:
        selected = rows_for_scope(rows, scope)
        y_true, y_pred = [], []
        for row in selected:
            p = phase_pred(row, missing_policy)
            if p is None:
                continue
            y_true.append(y_value(row))
            y_pred.append(p)
            predictions.append({**{f: row.get(f, "") for f in KEY_FIELDS}, "variant": "V4_phase_gate_only", "eval_scope": scope, "mode": "phase_gate", "true": y_value(row), "pred": p})
        if not y_true:
            metrics.append({"variant": "V4_phase_gate_only", "eval_scope": scope, "mode": "phase_gate", "status": "BLOCKED_MISSING_PHASE_OUTPUTS", "missing_phase_policy": missing_policy})
            continue
        rowm = metric_values(y_true, y_pred)
        pos = [r for r in selected if y_value(r) == 1]
        controls = [r for r in selected if lower(r.get("label_source")) == "clean_control_negative"]
        rowm.update({
            "variant": "V4_phase_gate_only",
            "eval_scope": scope,
            "mode": "phase_gate",
            "feature_set": "phase_is_critical",
            "model": "rule_proxy",
            "status": "PROXY_PHASE_GATE",
            "train_rows": len(y_true),
            "positive_rows": sum(y_true),
            "negative_rows": len(y_true) - sum(y_true),
            "phase_gate_positive_recall": "" if not pos else sum(phase_pred(r, missing_policy) == 1 for r in pos) / float(len(pos)),
            "phase_gate_control_rejection_rate": "" if not controls else sum(phase_pred(r, missing_policy) == 0 for r in controls) / float(len(controls)),
            "missing_phase_policy": missing_policy,
            "reason": "rule-based proxy phase gate; not a learned phase detector",
        })
        metrics.append(rowm)


def best_v3_predictions(metrics: list[dict[str, object]], predictions: list[dict[str, object]]) -> tuple[str, str, dict[tuple[str, str, str, str], int]]:
    candidates = [m for m in metrics if m.get("variant") == "V3_gold_plus_silver_plus_clean_controls" and norm(m.get("balanced_accuracy"))]
    if not candidates:
        return "", "", {}
    best = max(candidates, key=lambda r: float(r.get("balanced_accuracy") or 0))
    fs = norm(best.get("feature_set"))
    model = norm(best.get("model"))
    out = {}
    for row in predictions:
        if row.get("variant") == "V3_gold_plus_silver_plus_clean_controls" and row.get("feature_set") == fs and row.get("model") == model:
            out[key(row)] = int(float(norm(row.get("pred")) or 0))
    return fs, model, out


def add_cascade_metrics(rows: list[dict[str, str]], metrics: list[dict[str, object]], predictions: list[dict[str, object]]) -> None:
    fs, model, vuln = best_v3_predictions(metrics, predictions)
    if not vuln:
        metrics.append({"variant": "V5_hard_cascade", "eval_scope": "full_unknown_pass", "mode": "phase_plus_vulnerability", "status": "BLOCKED_NO_V3_PREDICTIONS"})
        return
    for variant, mode in [("V5_hard_cascade", "hard_and"), ("V6_soft_product", "soft_product")]:
        for scope, missing_policy in [("full_unknown_pass", "unknown_pass"), ("complete_case", "drop")]:
            selected = rows_for_scope(rows, scope)
            y_true, y_pred = [], []
            for row in selected:
                p_phase = phase_pred(row, missing_policy)
                if p_phase is None or key(row) not in vuln:
                    continue
                if mode == "hard_and":
                    pred = 1 if p_phase == 1 and vuln[key(row)] == 1 else 0
                else:
                    pc = phase_conf(row, missing_policy)
                    if pc is None:
                        continue
                    pred = 1 if pc * float(vuln[key(row)]) > 0.5 else 0
                y_true.append(y_value(row))
                y_pred.append(pred)
                predictions.append({**{f: row.get(f, "") for f in KEY_FIELDS}, "variant": variant, "eval_scope": scope, "mode": mode, "feature_set": fs, "model": model, "true": y_value(row), "pred": pred})
            if not y_true:
                metrics.append({"variant": variant, "eval_scope": scope, "mode": mode, "status": "BLOCKED_NO_EVALUABLE_ROWS", "feature_set": fs, "model": model})
                continue
            rowm = metric_values(y_true, y_pred)
            rowm.update({
                "variant": variant,
                "eval_scope": scope,
                "mode": "phase_plus_vulnerability",
                "feature_set": fs,
                "model": model,
                "status": "PROXY_PHASE_GATE",
                "train_rows": len(y_true),
                "positive_rows": sum(y_true),
                "negative_rows": len(y_true) - sum(y_true),
                "threshold": "0.5",
                "missing_phase_policy": missing_policy,
                "reason": "uses V3 binary LOTO predictions as vulnerability_score proxy; not calibrated probability",
            })
            metrics.append(rowm)


def write_report(path: str, dataset: list[dict[str, str]], metrics: list[dict[str, object]], notes: list[str]) -> None:
    train = [r for r in dataset if is_train(r)]
    phase_present = [r for r in train if phase_pred(r, "drop") is not None]
    coverage = 0.0 if not train else len(phase_present) / float(len(train))
    clean_controls = [r for r in train if lower(r.get("label_source")) == "clean_control_negative"]
    confirmed_controls = [r for r in clean_controls if lower(r.get("source_type")) == "clean_rollout_derived"]
    candidate_controls = [r for r in clean_controls if lower(r.get("source_type")) == "candidate_derived"]
    silver_rows = [r for r in train if "silver" in lower(r.get("label_source"))]
    lines = [
        "# Detector V2.7 Phase-Aware Two-Stage Ablation",
        "",
        "CPU-only exploratory diagnostic. No GPU, VIS, rollout, watcher, or final detector training was run.",
        "",
        f"**Train rows**: {len(train)}",
        f"**Phase coverage**: {len(phase_present)}/{len(train)} = {coverage:.3f}",
        "",
        "## Best Metrics By Variant",
        "",
        "| Variant | Scope | Status | Best BAcc | Model | Pos recall | Neg recall | FPR | FP | FN |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted({norm(m.get("variant")) for m in metrics}):
        scoped = [m for m in metrics if m.get("variant") == variant and norm(m.get("balanced_accuracy"))]
        if not scoped:
            m0 = next((m for m in metrics if m.get("variant") == variant), {})
            lines.append(f"| {variant} | {m0.get('eval_scope','')} | {m0.get('status','BLOCKED')} |  |  |  |  |  |  |  |")
            continue
        best = max(scoped, key=lambda m: float(m.get("balanced_accuracy") or 0))
        lines.append(
            f"| {variant} | {best.get('eval_scope','')} | {best.get('status','')} | {best.get('balanced_accuracy','')} | "
            f"{best.get('feature_set','')}/{best.get('model','')} | {best.get('positive_recall','')} | {best.get('negative_recall','')} | "
            f"{best.get('false_positive_rate','')} | {best.get('fp','')} | {best.get('fn','')} |"
        )
    lines.extend([
        "",
        "## Phase Detector Diagnostic",
        "",
        "- `phase_is_critical` is a rule/proxy feature table in this run unless `predicted_phase`/`phase_confidence` exists.",
        "- Full-scope phase evaluation uses `unknown_pass` to preserve recall; complete-case rows are reported separately.",
        "- Learned phase detector claims are not made here.",
        "",
        "## Interpretation",
        "",
        f"- Clean controls: {len(clean_controls)} total, {len(confirmed_controls)} clean-rollout-derived, {len(candidate_controls)} candidate-derived.",
        f"- 1R silver positives in train: {len(silver_rows)} provisional rows.",
    ])
    v3_rows = [m for m in metrics if m.get("variant") == "V3_gold_plus_silver_plus_clean_controls" and norm(m.get("balanced_accuracy"))]
    v5_rows = [m for m in metrics if m.get("variant") == "V5_hard_cascade" and norm(m.get("balanced_accuracy"))]
    v3 = max(v3_rows, key=lambda m: float(m.get("balanced_accuracy") or 0)) if v3_rows else {}
    v5 = max(v5_rows, key=lambda m: float(m.get("balanced_accuracy") or 0)) if v5_rows else {}
    if v3 and v5:
        if norm(v3.get("false_positive_rate")) == norm(v5.get("false_positive_rate")) and norm(v3.get("fp")) == norm(v5.get("fp")):
            lines.append("- Hard cascade did not reduce FPR/FP beyond the V3 vulnerability detector in this run.")
        else:
            lines.append("- Hard cascade changed FPR/FP relative to V3; inspect metrics before any claim.")
    lines.extend([
        "",
        "## Readiness Decision",
        "",
    ])
    ready = (
        len(train) >= 35
        and coverage >= 0.95
        and len(confirmed_controls) >= 6
        and len(candidate_controls) == 0
        and float(v3.get("false_positive_rate") or 1.0) < 0.4616
        and float(v3.get("negative_recall") or 0.0) > 0.5385
        and float(v3.get("positive_recall") or 0.0) >= 0.8
    )
    if ready:
        lines.append("**READY_FOR_V3_CANDIDATE**: metric gates pass, but advisor review is still required before final detector training.")
    elif len(train) >= 35 and coverage >= 0.8:
        lines.append("**READY_FOR_PHASE_COMPLETE_DIAGNOSTIC**: enough rows exist for exploratory phase-aware diagnostics, but v3/final readiness is blocked.")
    else:
        lines.append("**EXPLORATORY_ONLY**: phase coverage or label provenance is insufficient for v3 readiness.")
    lines.extend(["", "## Blockers", ""])
    blockers = []
    if coverage < 0.95:
        blockers.append("phase coverage < 95%")
    if len(confirmed_controls) < 6:
        blockers.append("confirmed clean-rollout hard negatives < 6")
    if candidate_controls:
        blockers.append("clean controls are candidate-derived, not confirmed hard negatives")
    blockers.append("clean controls are diagnostic/candidate-derived unless confirmed by clean rollout provenance")
    blockers.append("1R silver positives remain provisional and quality-audit dependent")
    blockers.append("calibration PASS is not established in this report")
    blockers.append("no pending_negative_1r is used as negative")
    blockers.append("no final detector training claim")
    lines.extend(f"- {b}" for b in blockers)
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {n}" for n in notes) if notes else lines.append("- None.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    dataset = read_csv(args.dataset)
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    notes: list[str] = []

    for variant in ["V0_gold_only", "V1_gold_plus_silver", "V2_gold_plus_clean_controls", "V3_gold_plus_silver_plus_clean_controls"]:
        selected = variant_rows(dataset, variant)
        pos = sum(y_value(r) for r in selected)
        neg = len(selected) - pos
        if len(selected) < args.min_rows or not pos or not neg:
            metrics.append({"variant": variant, "eval_scope": "full", "mode": "vulnerability_only", "status": "BLOCKED_DATASET", "train_rows": len(selected), "positive_rows": pos, "negative_rows": neg})
            continue
        code, mpath, ppath, stdout, stderr = run_trainer(args, variant, selected)
        status = "OK" if code == 0 else f"TRAINER_EXIT_{code}"
        if stdout:
            notes.append(f"{variant} stdout: {stdout[:400]}")
        if stderr:
            notes.append(f"{variant} stderr: {stderr[:400]}")
        add_trainer_metrics(mpath, variant, metrics, status)
        add_trainer_predictions(ppath, variant, predictions)

    add_phase_gate_metrics(dataset, metrics, predictions)
    add_cascade_metrics(dataset, metrics, predictions)

    metric_fields = sorted(set(METRIC_FIELDS) | {k for r in metrics for k in r.keys()})
    pred_fields = sorted({k for r in predictions for k in r.keys()} | {"variant", "eval_scope", "mode"})
    write_csv(args.output_metrics, metric_fields, metrics)
    write_csv(args.output_predictions, pred_fields, predictions)
    write_report(args.output_report, dataset, metrics, notes)
    print(f"v27_metric_rows={len(metrics)} v27_prediction_rows={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
