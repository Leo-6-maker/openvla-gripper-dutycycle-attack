#!/usr/bin/env python3
"""CPU-only VisualTransferHead probe scaffold.

Supports metadata-only and dummy-visual modes. Dummy visual features are not
scientific evidence.
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-csv", default="tables/visual_transfer_dataset_v0.csv")
    ap.add_argument("--feature-manifest", default="tables/visual_transfer_feature_manifest_stub_v0.csv")
    ap.add_argument("--mode", choices=["metadata_only", "dummy_visual"], default="metadata_only")
    ap.add_argument("--output-metrics", default="tables/visual_transfer_probe_metrics_v0.csv")
    ap.add_argument("--output-report", default="reports/VISUAL_TRANSFER_PROBE_V0.md")
    return ap.parse_args()


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label(row):
    return 1 if str(row.get("label_vulnerability_ready", "")).strip() == "1" else 0


def train_rows(rows):
    return [r for r in rows if r.get("label_status") in {"positive", "negative"}]


def majority(values, default=1):
    if not values:
        return default
    c = Counter(values)
    return 1 if c[1] >= c[0] else 0


def predict_by_key(rows, key_fields):
    y = [label(r) for r in rows]
    global_major = majority(y)
    preds = []
    for i, row in enumerate(rows):
        key = tuple(row.get(f, "") for f in key_fields)
        train_vals = [
            label(r)
            for j, r in enumerate(rows)
            if j != i and tuple(r.get(f, "") for f in key_fields) == key
        ]
        preds.append(majority(train_vals, global_major))
    return preds


def metrics(y, preds):
    tp = sum(1 for a, p in zip(y, preds) if a == 1 and p == 1)
    tn = sum(1 for a, p in zip(y, preds) if a == 0 and p == 0)
    fp = sum(1 for a, p in zip(y, preds) if a == 0 and p == 1)
    fn = sum(1 for a, p in zip(y, preds) if a == 1 and p == 0)
    rec_pos = tp / max(tp + fn, 1)
    rec_neg = tn / max(tn + fp, 1)
    prec_pos = tp / max(tp + fp, 1)
    prec_neg = tn / max(tn + fn, 1)
    f1_pos = 2 * prec_pos * rec_pos / max(prec_pos + rec_pos, 1e-8)
    f1_neg = 2 * prec_neg * rec_neg / max(prec_neg + rec_neg, 1e-8)
    denom = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denom)
    return {
        "balanced_accuracy": round((rec_pos + rec_neg) / 2, 4),
        "macro_F1": round((f1_pos + f1_neg) / 2, 4),
        "negative_recall": round(rec_neg, 4),
        "MCC": round(mcc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def control_false_positives(rows, preds):
    count = 0
    total = 0
    for row, pred in zip(rows, preds):
        role = row.get("candidate_role", "")
        is_control = "control" in role or row.get("label_control_negative") == "true"
        if is_control:
            total += 1
            if pred == 1:
                count += 1
    return count, total


def main():
    args = parse_args()
    rows = train_rows(read_csv(args.dataset_csv))
    feature_rows = read_csv(args.feature_manifest)
    y = [label(r) for r in rows]
    baselines = {
        "always_positive": [1 for _ in rows],
        "task_key_only": predict_by_key(rows, ["task_key"]),
        "phase_candidate_role_only": predict_by_key(rows, ["phase_bin_proxy", "candidate_role"]),
        "proprio_summary_only": predict_by_key(rows, ["gripper_qpos_at_trigger", "eef_speed_mean_pre", "open_streak_pre", "close_streak_pre"]),
        "task_plus_phase": predict_by_key(rows, ["task_key", "phase_bin_proxy", "candidate_role"]),
        "task_plus_proprio_summary": predict_by_key(rows, ["task_key", "gripper_qpos_at_trigger", "eef_speed_mean_pre"]),
    }
    if args.mode == "dummy_visual":
        available = {r["sample_id"] for r in feature_rows}
        baselines["dummy_visual_only"] = [1 if r["sample_id"] in available else majority(y) for r in rows]
    metrics_rows = []
    for name, preds in baselines.items():
        m = metrics(y, preds)
        cfp, ctot = control_false_positives(rows, preds)
        m.update({
            "model": name,
            "mode": args.mode,
            "n_rows": len(rows),
            "control_false_positives": cfp,
            "control_rows": ctot,
        })
        metrics_rows.append(m)
    os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
    fields = ["mode", "model", "n_rows", "balanced_accuracy", "macro_F1", "negative_recall", "MCC", "tp", "fp", "fn", "tn", "control_false_positives", "control_rows"]
    with open(args.output_metrics, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics_rows)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    tasks = sorted(set(r.get("task_key", "") for r in rows))
    loto = len(tasks) >= 3
    lines = [
        "# Visual Transfer Probe V0",
        "",
        f"**Mode**: `{args.mode}`",
        f"**Rows**: {len(rows)}",
        f"**Tasks**: {len(tasks)}",
        f"**LOTO feasible**: {str(loto).lower()}",
        "",
        "## Metrics",
        "",
        "| Model | Bal Acc | Macro F1 | Neg Recall | MCC | TP/FP/FN/TN | Control FP |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in metrics_rows:
        lines.append("| {model} | {balanced_accuracy} | {macro_F1} | {negative_recall} | {MCC} | {tp}/{fp}/{fn}/{tn} | {control_false_positives}/{control_rows} |".format(**r))
    lines += ["", "## Boundary", ""]
    if args.mode == "dummy_visual":
        lines.append("Visual branch not scientifically evaluated. Dummy visual features are pipeline smoke only.")
    else:
        lines.append("Metadata-only scaffold; no visual branch was evaluated.")
    lines.append("")
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Metrics: %s" % args.output_metrics)
    print("Report: %s" % args.output_report)


if __name__ == "__main__":
    main()
