#!/usr/bin/env python3
"""Compare Fast VIS proxy outputs against full VIS reference labels.

This script is CPU-only. It reads CSV outputs only; it does not run rollout,
VIS, watcher, or detector training.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


NON_LABEL_STATUSES = (
    "INFRA_FAILED",
    "MEASUREMENT_FAILED",
    "BLOCKED",
    "ERROR",
    "missing",
    "schema_incomplete",
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="tables/fast_vis_calibration_candidates_v0.csv")
    ap.add_argument("--policy-only", default="tables/fast_vis_policy_only_audit_v0.csv")
    ap.add_argument("--command-proxy", default="tables/fast_vis_command_proxy_v0.csv")
    ap.add_argument("--low-budget", default="tables/fast_vis_low_budget_sweep_v0.csv")
    ap.add_argument("--output-csv", default="tables/fast_vis_proxy_comparison_v0.csv")
    ap.add_argument("--output-report", default="reports/FAST_VIS_PROXY_COMPARISON_V0.md")
    return ap.parse_args()


def read_csv(path):
    if not os.path.exists(path):
        return None, []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def key_for(row):
    ws = row.get("window_start", row.get("parent_window_start", ""))
    we = row.get("window_end", row.get("parent_window_end", ""))
    return (row.get("task_key", ""), str(row.get("state_id", "")), str(ws), str(we))


def parse_boolish(value):
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "positive", "pos", "fail", "failed", "failure"}:
        return 1
    if s in {"0", "false", "no", "n", "negative", "neg", "success", "succeeded"}:
        return 0
    return None


def parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        val = float(value)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def full_label_from(row):
    for col in ("full_vis_label", "label", "label_vulnerability_ready"):
        if col in row:
            parsed = parse_boolish(row.get(col))
            if parsed is not None:
                return parsed
    status = str(row.get("label_status", "")).strip().lower()
    if status == "positive":
        return 1
    if status == "negative":
        return 0
    return None


def is_usable_label_row(row):
    text = " ".join(str(row.get(k, "")) for k in row.keys())
    if any(tok.lower() in text.lower() for tok in NON_LABEL_STATUSES):
        return False
    confidence = str(row.get("label_confidence", "")).lower()
    if confidence.startswith("not_label"):
        return False
    return True


def proxy_prediction_from(row, dataset):
    for col in ("proxy_label", "fast_label", "predicted_label", "proxy_positive"):
        if col in row:
            parsed = parse_boolish(row.get(col))
            if parsed is not None:
                return parsed, col

    if "vis_open" in row:
        parsed = parse_boolish(row.get("vis_open"))
        if parsed is not None:
            return parsed, "vis_open"

    if "task_failure" in row:
        parsed = parse_boolish(row.get("task_failure"))
        if parsed is not None:
            return parsed, "task_failure"

    if "task_done" in row:
        parsed = parse_boolish(row.get("task_done"))
        if parsed is not None:
            return 0 if parsed == 1 else 1, "not_task_done"

    for col in ("official_success", "success", "done"):
        if col in row:
            parsed = parse_boolish(row.get(col))
            if parsed is not None:
                return 0 if parsed == 1 else 1, f"not_{col}"

    qpos = parse_float(row.get("qpos_opening_delta", row.get("qpos_delta", "")))
    if qpos is not None:
        return 1 if qpos >= 0.01 else 0, "qpos_opening_delta_ge_0.01"

    return None, "missing_proxy_prediction"


def budget_id(row, dataset):
    if row.get("budget"):
        return str(row.get("budget"))
    parts = []
    for col in ("eps_raw_pixels", "eps", "pgd_steps", "pgd_restarts"):
        if str(row.get(col, "")).strip() != "":
            parts.append(f"{col}={row.get(col)}")
    return ",".join(parts) if parts else dataset


def metric_row(dataset, budget, rows):
    tp = fp = fn = tn = 0
    runtimes = []
    full_runtimes = []
    for row in rows:
        y = int(row["full_label"])
        p = int(row["proxy_prediction"])
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
        elif y == 0 and p == 0:
            tn += 1
        rt = parse_float(row.get("runtime_sec"))
        if rt is not None:
            runtimes.append(rt)
        frt = parse_float(row.get("full_vis_runtime_sec", row.get("full_runtime_sec", "")))
        if frt is not None:
            full_runtimes.append(frt)
    pos_recall = tp / (tp + fn) if (tp + fn) else ""
    neg_specificity = tn / (tn + fp) if (tn + fp) else ""
    agreement = (tp + tn) / len(rows) if rows else ""
    balanced = (
        (float(pos_recall) + float(neg_specificity)) / 2.0
        if pos_recall != "" and neg_specificity != "" else ""
    )
    runtime_mean = sum(runtimes) / len(runtimes) if runtimes else ""
    full_runtime_mean = sum(full_runtimes) / len(full_runtimes) if full_runtimes else ""
    runtime_reduction = (
        1.0 - float(runtime_mean) / float(full_runtime_mean)
        if runtime_mean != "" and full_runtime_mean not in ("", 0) else ""
    )
    return {
        "dataset": dataset,
        "budget": budget,
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_recall": pos_recall,
        "negative_specificity": neg_specificity,
        "agreement_with_full_vis": agreement,
        "balanced_accuracy": balanced,
        "runtime_sec_mean": runtime_mean,
        "full_vis_runtime_sec_mean": full_runtime_mean,
        "runtime_reduction": runtime_reduction,
        "false_positives_on_controls": fp,
    }


def collect_comparisons(dataset, path, candidates_by_key):
    fields, rows = read_csv(path)
    if fields is None:
        return [], [{"dataset": dataset, "issue": "missing_csv", "path": path}]

    comparisons = []
    issues = []
    for row in rows:
        if not is_usable_label_row(row):
            issues.append({"dataset": dataset, "issue": "skipped_non_label_row", "path": path})
            continue
        key = key_for(row)
        ref = candidates_by_key.get(key)
        full_label = full_label_from(row)
        if full_label is None and ref is not None:
            full_label = full_label_from(ref)
        if full_label is None:
            issues.append({"dataset": dataset, "issue": "missing_full_label", "path": path})
            continue
        pred, pred_source = proxy_prediction_from(row, dataset)
        if pred is None:
            issues.append({"dataset": dataset, "issue": pred_source, "path": path})
            continue
        out = dict(row)
        out["dataset"] = dataset
        out["full_label"] = int(full_label)
        out["proxy_prediction"] = int(pred)
        out["proxy_prediction_source"] = pred_source
        out["budget"] = budget_id(row, dataset)
        comparisons.append(out)
    return comparisons, issues


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "dataset", "budget", "n", "tp", "fp", "fn", "tn",
        "positive_recall", "negative_specificity", "agreement_with_full_vis",
        "balanced_accuracy", "runtime_sec_mean", "full_vis_runtime_sec_mean",
        "runtime_reduction", "false_positives_on_controls",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def choose_recommendation(metrics):
    eligible = [m for m in metrics if m["n"] and m["positive_recall"] != "" and m["negative_specificity"] != ""]
    if not eligible:
        return "none: no complete comparable Fast VIS outputs"
    eligible.sort(
        key=lambda m: (
            float(m["balanced_accuracy"]),
            float(m["negative_specificity"]),
            float(m["positive_recall"]),
            -float(m["runtime_sec_mean"] or 1e9),
        ),
        reverse=True,
    )
    best = eligible[0]
    return f"{best['dataset']} / {best['budget']} (balanced_accuracy={fmt(best['balanced_accuracy'])})"


def write_report(path, metrics, issues):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not metrics:
        status = "BLOCKED_MISSING_OR_INCOMPLETE_FAST_VIS_OUTPUTS"
    elif any(m["false_positives_on_controls"] for m in metrics):
        status = "HAS_CONTROL_FALSE_POSITIVES"
    else:
        status = "COMPARISON_READY_FOR_REVIEW"

    lines = [
        "# Fast VIS Proxy Comparison v0",
        "",
        f"**Status**: {status}",
        "",
        "This report compares Fast cascade proxy outputs against full VIS reference labels. It is CPU-only and reads CSVs only.",
        "",
        "## Metrics",
        "",
        "| Dataset | Budget | n | TP | FP | FN | TN | Positive recall | Negative specificity | Agreement | Runtime reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['dataset']} | {m['budget']} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
            f"{fmt(m['positive_recall'])} | {fmt(m['negative_specificity'])} | "
            f"{fmt(m['agreement_with_full_vis'])} | {fmt(m['runtime_reduction'])} |"
        )

    lines.extend([
        "",
        "## Recommended Fast Budget",
        "",
        choose_recommendation(metrics),
        "",
        "## Failure Modes / Issues",
        "",
    ])
    if issues:
        counts = defaultdict(int)
        for issue in issues:
            counts[(issue["dataset"], issue["issue"])] += 1
        for (dataset, issue), count in sorted(counts.items()):
            lines.append(f"- {dataset}: {issue} ({count})")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- Policy-only outputs do not prove task-level success or failure.",
        "- Command-open proxy outputs do not prove VIS.",
        "- Silver/proxy labels are not gold labels.",
        "- INFRA_FAILED, MEASUREMENT_FAILED, BLOCKED, and ERROR rows are excluded from comparison metrics.",
        "- Agreement with full VIS is an acceleration-screening result, not detector validation.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    _, candidate_rows = read_csv(args.candidates)
    candidates_by_key = {key_for(r): r for r in candidate_rows}

    all_comparisons = []
    all_issues = []
    for dataset, path in [
        ("policy_only", args.policy_only),
        ("command_proxy", args.command_proxy),
        ("low_budget", args.low_budget),
    ]:
        comparisons, issues = collect_comparisons(dataset, path, candidates_by_key)
        all_comparisons.extend(comparisons)
        all_issues.extend(issues)

    grouped = defaultdict(list)
    for row in all_comparisons:
        grouped[(row["dataset"], row["budget"])].append(row)
    metrics = [metric_row(dataset, budget, rows) for (dataset, budget), rows in sorted(grouped.items())]

    write_csv(args.output_csv, metrics)
    write_report(args.output_report, metrics, all_issues)
    print(f"metrics={len(metrics)} issues={len(all_issues)} report={args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
