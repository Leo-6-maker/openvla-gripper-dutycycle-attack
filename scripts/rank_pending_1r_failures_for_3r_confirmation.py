#!/usr/bin/env python3
"""Rank pending VIS-1R failures for future 3R confirmation.

CPU-only ranking. Pending 1R failures are never used as train negatives.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
OUTPUT_FIELDS = [
    "candidate_id", "task_key", "state_id", "window_start", "window_end",
    "expected_role", "detector_v25_prob_positive", "reason_selected",
    "confirmation_priority", "recommended_action",
]
BAD_TOKENS = ("infra", "infra_failed", "manual_review", "polluted", "action_confounded")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", default="tables/detector_v25_incremental_predictions.csv")
    ap.add_argument("--dataset-csv", "--dataset", dest="dataset_csv", default="tables/detector_v25_incremental_dataset.csv")
    ap.add_argument("--adaptive-summary", default="tables/adaptive_vis_1r_screening_summary.csv")
    ap.add_argument("--adaptive-provenance", default="tables/adaptive_vis_1r_provenance.csv")
    ap.add_argument("--candidates-csv", default="tables/object_phase_response_adaptive_candidates.csv")
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--output-csv", default="tables/pending_1r_failures_3r_confirmation_queue.csv")
    ap.add_argument("--output-report", default="reports/PENDING_1R_FAILURES_3R_CONFIRMATION_QUEUE.md")
    ap.add_argument("--dry-run", action="store_true")
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


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(norm(row.get(field)) for field in KEY_FIELDS)


def parse_float(value):
    try:
        text = norm(value)
        return 0.0 if text == "" else float(text)
    except Exception:
        return 0.0


def is_pending_negative_1r(row):
    text = " ".join(lower(row.get(field)) for field in ["label_source", "label_confidence", "label_1r", "label_status", "inclusion_status"])
    return "pending_negative_1r" in text


def has_bad_status(row):
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in BAD_TOKENS)


def prediction_scores(rows):
    vals = defaultdict(list)
    for row in rows:
        model = lower(row.get("model"))
        if model.startswith("always_") or model == "prevalence_random":
            continue
        k = key(row)
        if norm(row.get("prob_positive")):
            vals[k].append(parse_float(row.get("prob_positive")))
        elif norm(row.get("pred")):
            vals[k].append(parse_float(row.get("pred")))
        elif norm(row.get("y_pred")):
            vals[k].append(parse_float(row.get("y_pred")))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in vals.items()}


def joined_rows(dataset, adaptive, candidates, labels):
    label_keys = {key(r) for r in labels}
    by_key = {}
    for row in candidates:
        by_key[key(row)] = dict(row)
    for row in adaptive:
        cur = by_key.get(key(row), {})
        cur.update({k: v for k, v in row.items() if norm(v) != ""})
        by_key[key(row)] = cur
    for row in dataset:
        cur = by_key.get(key(row), {})
        cur.update({k: v for k, v in row.items() if norm(v) != ""})
        by_key[key(row)] = cur
    return [r for k, r in by_key.items() if k not in label_keys and is_pending_negative_1r(r)]


def score_candidate(row, prob):
    score = prob * 100.0
    reasons = [f"detector_prob={prob:.3f}"]
    role = lower(row.get("expected_role"))
    task = lower(row.get("task_key"))
    if role == "hard_negative" or "hard_negative" in role:
        score += 25
        reasons.append("hard_negative")
    if task in {"milk", "ketchup", "alphabet_soup"}:
        score += 15
        reasons.append("priority_task")
    text = " ".join(lower(row.get(f)) for f in ["candidate_role", "source_reason", "reason_selected"])
    if "contrast" in text or "same_task" in text:
        score += 10
        reasons.append("same_task_contrast")
    if lower(row.get("denominator_status")) in {"clean", ""} or lower(row.get("denominator_clean")) == "true":
        score += 5
        reasons.append("clean_denominator")
    if lower(row.get("provenance_status")) in {"ok", "complete", ""}:
        score += 5
        reasons.append("usable_provenance")
    if norm(row.get("qpos_phase_class")) or norm(row.get("phase_bin_proxy")):
        score += 3
        reasons.append("phase_metadata")
    return score, ";".join(reasons)


def write_report(path, args, rows, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Pending 1R Failures 3R Confirmation Queue",
        "",
        f"**Queue rows**: {len(rows)}",
        "",
        "This queue is ranking-only. Pending 1R failures are not used as train negatives.",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{args.predictions_csv}`",
        f"- Dataset: `{args.dataset_csv}`",
        f"- Adaptive summary: `{args.adaptive_summary}`",
        f"- Candidates: `{args.candidates_csv}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {n}" for n in notes) if notes else lines.append("- None.")
    lines.extend(["", "## Top Candidates", ""])
    if rows:
        lines.extend(["| Priority | Candidate | Task | State | Window | Prob+ | Role | Reason |",
                      "|---:|---|---|---:|---|---:|---|---|"])
        for row in rows[:30]:
            lines.append(
                f"| {row['confirmation_priority']} | {row['candidate_id']} | {row['task_key']} | "
                f"{row['state_id']} | {row['window_start']}-{row['window_end']} | "
                f"{row['detector_v25_prob_positive']} | {row['expected_role']} | {row['reason_selected']} |"
            )
    else:
        lines.append("- No pending_negative_1r rows available.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    predictions = read_csv(args.predictions_csv)
    dataset = read_csv(args.dataset_csv)
    adaptive = read_csv(args.adaptive_summary)
    candidates = read_csv(args.candidates_csv or args.adaptive_provenance)
    labels = read_csv(args.labels_v2)
    if not predictions:
        notes.append("missing_or_empty_predictions")
    if not dataset and not adaptive:
        notes.append("missing_pending_sources")
    scores = prediction_scores(predictions)
    rows = []
    for row in joined_rows(dataset, adaptive, candidates, labels):
        if has_bad_status(row):
            continue
        prob = scores.get(key(row), 0.0)
        score, reason = score_candidate(row, prob)
        rows.append({
            "candidate_id": norm(row.get("candidate_id")) or "/".join(key(row)),
            "task_key": norm(row.get("task_key")),
            "state_id": norm(row.get("state_id")),
            "window_start": norm(row.get("window_start")),
            "window_end": norm(row.get("window_end")),
            "expected_role": norm(row.get("expected_role")),
            "detector_v25_prob_positive": f"{prob:.6g}",
            "reason_selected": reason,
            "confirmation_priority": f"{score:.6g}",
            "recommended_action": "run_3R_confirmation",
        })
    rows.sort(key=lambda r: (-parse_float(r["confirmation_priority"]), r["task_key"], r["state_id"], r["window_start"]))
    write_csv(args.output_csv, rows)
    write_report(args.output_report, args, rows, notes)
    if args.dry_run:
        print(f"DRY RUN: queue_rows={len(rows)}")
        for note in notes:
            print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
