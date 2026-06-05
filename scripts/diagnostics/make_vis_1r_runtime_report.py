#!/usr/bin/env python3
"""Summarize VIS-1R per-candidate runtime and estimated speedup."""

from __future__ import annotations

import argparse
import csv
import os


FIELDS = ["task_key", "state_id", "window_start", "window_end", "runtime_1r_sec", "runtime_3r_sec", "estimated_speedup", "runtime_status"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-1r-summary", default="tables/object_phase_response_vis_1r_summary.csv")
    ap.add_argument("--vis-3r-summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--output-csv", default="tables/vis_1r_runtime_report.csv")
    ap.add_argument("--output-report", default="reports/VIS_1R_RUNTIME_REPORT.md")
    return ap.parse_args()


def norm(v):
    return str(v if v is not None else "").strip()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in csv.DictReader(f)]


def key(r):
    return (norm(r.get("task_key")), norm(r.get("state_id")), norm(r.get("window_start")), norm(r.get("window_end")))


def runtime(r):
    for field in ["runtime_sec", "elapsed_sec", "pgd_runtime_sec"]:
        try:
            text = norm(r.get(field))
            if text:
                return float(text)
        except Exception:
            pass
    return None


def main():
    args = parse_args()
    one = {key(r): r for r in read_csv(args.vis_1r_summary)}
    three = {key(r): r for r in read_csv(args.vis_3r_summary)}
    keys = sorted(set(one) | set(three))
    rows = []
    for k in keys:
        rt1 = runtime(one.get(k, {}))
        rt3 = runtime(three.get(k, {}))
        speed = "" if rt1 in (None, 0) or rt3 is None else f"{rt3 / rt1:.6g}"
        rows.append({
            "task_key": k[0], "state_id": k[1], "window_start": k[2], "window_end": k[3],
            "runtime_1r_sec": "" if rt1 is None else f"{rt1:.6g}",
            "runtime_3r_sec": "" if rt3 is None else f"{rt3:.6g}",
            "estimated_speedup": speed,
            "runtime_status": "complete" if rt1 is not None else "missing_1r_runtime",
        })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    speeds = [float(r["estimated_speedup"]) for r in rows if r["estimated_speedup"]]
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join([
            "# VIS-1R Runtime Report",
            "",
            f"**Rows**: {len(rows)}",
            f"**Mean estimated speedup**: {sum(speeds)/len(speeds):.3f}" if speeds else "**Mean estimated speedup**: unavailable",
            "",
            "Per-candidate runtime is read from CSV artifacts only. No GPU/VIS was run.",
        ]))
    print(f"runtime rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
