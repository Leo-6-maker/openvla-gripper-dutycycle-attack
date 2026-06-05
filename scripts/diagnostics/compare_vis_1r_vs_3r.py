#!/usr/bin/env python3
"""Compare VIS-1R screening outputs against VIS-3R gold references."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


FIELDS = [
    "task_key", "state_id", "window_start", "window_end",
    "label_1r", "label_3r", "agreement_status", "label_tier_recommendation",
    "runtime_1r_sec", "runtime_3r_sec", "estimated_speedup",
    "audit_status", "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-1r-summary", default="tables/object_phase_response_vis_1r_summary.csv")
    ap.add_argument("--vis-3r-summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--candidates", default="tables/object_phase_response_batch4_candidates.csv")
    ap.add_argument("--output-csv", default="tables/vis_1r_vs_3r_audit.csv")
    ap.add_argument("--output-report", default="reports/VIS_1R_VS_3R_AUDIT.md")
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


def parse_label(row):
    for field in ["label_vulnerability_ready", "full_vis_label", "label", "classification"]:
        value = lower(row.get(field))
        if value in {"1", "true", "positive"}:
            return "positive"
        if value in {"0", "false", "negative"}:
            return "negative"
    status = lower(row.get("label_status"))
    if status in {"positive", "negative"}:
        return status
    return "unknown"


def parse_float(v):
    try:
        text = norm(v)
        return None if text == "" else float(text)
    except Exception:
        return None


def runtime(row):
    for field in ["runtime_sec", "elapsed_sec", "pgd_runtime_sec"]:
        val = parse_float(row.get(field))
        if val is not None:
            return val
    return None


def main():
    args = parse_args()
    one = {key(r): r for r in read_csv(args.vis_1r_summary)}
    three = {key(r): r for r in read_csv(args.vis_3r_summary)}
    candidates = read_csv(args.candidates)
    keys = sorted(set(one) | set(three) | {key(r) for r in candidates})
    rows = []
    for k in keys:
        r1 = one.get(k, {})
        r3 = three.get(k, {})
        l1 = parse_label(r1) if r1 else "missing"
        l3 = parse_label(r3) if r3 else "missing"
        rt1 = runtime(r1) if r1 else None
        rt3 = runtime(r3) if r3 else None
        speed = "" if rt1 in (None, 0) or rt3 is None else f"{rt3 / rt1:.6g}"
        if l3 in {"positive", "negative"}:
            tier = "gold_3r"
            agreement = "agree" if l1 == l3 else ("missing_1r" if l1 == "missing" else "disagree")
        elif l1 == "positive":
            tier = "silver_positive_1r"
            agreement = "pending_3r"
        elif l1 == "negative":
            tier = "pending_negative_1r"
            agreement = "pending_3r"
        else:
            tier = "missing"
            agreement = "missing"
        rows.append({
            "task_key": k[0], "state_id": k[1], "window_start": k[2], "window_end": k[3],
            "label_1r": l1, "label_3r": l3, "agreement_status": agreement,
            "label_tier_recommendation": tier,
            "runtime_1r_sec": "" if rt1 is None else f"{rt1:.6g}",
            "runtime_3r_sec": "" if rt3 is None else f"{rt3:.6g}",
            "estimated_speedup": speed,
            "audit_status": "ok" if tier != "missing" else "pending",
            "reason": "1R negative is pending only, never gold" if tier == "pending_negative_1r" else tier,
        })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    speeds = [parse_float(r["estimated_speedup"]) for r in rows if parse_float(r["estimated_speedup"]) is not None]
    counts = Counter(r["label_tier_recommendation"] for r in rows)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# VIS-1R vs VIS-3R Audit",
            "",
            f"**Rows**: {len(rows)}",
            f"**gold_3r rows**: {counts.get('gold_3r', 0)}",
            f"**silver_positive_1r rows**: {counts.get('silver_positive_1r', 0)}",
            f"**pending_negative_1r rows**: {counts.get('pending_negative_1r', 0)}",
            f"**Mean estimated speedup**: {sum(speeds)/len(speeds):.3f}" if speeds else "**Mean estimated speedup**: unavailable",
            "",
            "## Boundary",
            "",
            "- 1R positives are screening silver positives only.",
            "- 1R negatives remain pending and must not be used as gold negatives.",
            "- Gold labels require 3R full VIS.",
        ]))
    print(f"compared {len(rows)} VIS-1R/3R rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
