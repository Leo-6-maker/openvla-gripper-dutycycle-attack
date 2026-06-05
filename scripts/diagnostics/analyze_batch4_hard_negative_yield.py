#!/usr/bin/env python3
"""Analyze detector-v2 FP to Batch4 hard-negative full-VIS yield."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


FIELDS = ["task_key", "state_id", "window_start", "window_end", "v2_error_type", "targeted_by_batch4", "batch4_expected_role", "batch4_classification", "remaining_gap", "recommended_batch5_action"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default="tables/detector_v2_error_analysis.csv")
    ap.add_argument("--candidates", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--output-csv", default="tables/batch4_hard_negative_yield.csv")
    ap.add_argument("--output-report", default="reports/BATCH4_HARD_NEGATIVE_YIELD.md")
    return ap.parse_args()


def norm(v):
    return str(v if v is not None else "").strip()


def lower(v):
    return norm(v).lower()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def key(r):
    return (norm(r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start")), norm(r.get("window_end")))


def main():
    args = parse_args()
    errors = read_csv(args.errors)
    candidates = {key(r): r for r in read_csv(args.candidates)}
    summary = {key(r): r for r in read_csv(args.summary)}
    rows = []
    for err in errors:
        k = key(err)
        cand = candidates.get(k, {})
        summ = summary.get(k, {})
        targeted = bool(cand)
        cls = norm(summ.get("classification"))
        gap = "resolved_negative" if cls == "negative" else ("not_targeted" if not targeted else "needs_review_or_positive")
        rows.append({
            "task_key": k[0], "state_id": k[1], "window_start": k[2], "window_end": k[3],
            "v2_error_type": norm(err.get("error_type")), "targeted_by_batch4": str(targeted).lower(),
            "batch4_expected_role": norm(cand.get("expected_role")), "batch4_classification": cls,
            "remaining_gap": gap,
            "recommended_batch5_action": "target_same_task_negative" if gap != "resolved_negative" else "none",
        })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    tasks_lack_neg = sorted({r["task_key"] for r in rows if r["remaining_gap"] != "resolved_negative"})
    phases = Counter(lower(candidates.get((r["task_key"], r["state_id"], r["window_start"], r["window_end"]), {}).get("phase_bin_proxy")) for r in rows)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# Batch4 Hard Negative Yield",
            "",
            f"**Rows**: {len(rows)}",
            f"**Targeted v2 FPs**: {sum(1 for r in rows if r['targeted_by_batch4'] == 'true')}",
            f"**Resolved as full VIS negatives**: {sum(1 for r in rows if r['remaining_gap'] == 'resolved_negative')}",
            f"**Tasks still lacking negatives**: {', '.join(tasks_lack_neg) if tasks_lack_neg else 'none'}",
            "",
            "## Phase FP Distribution",
            "",
            *[f"- `{k or 'missing'}`: {v}" for k, v in sorted(phases.items())],
            "",
            "## Batch5 Guidance",
            "",
            "- Target unresolved v2 FP patterns with qpos-verified hard negatives.",
        ]))
    print(f"analyzed {len(rows)} Batch4 hard-negative yield rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
