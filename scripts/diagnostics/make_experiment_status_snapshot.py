#!/usr/bin/env python3
"""Create an automatic experiment status snapshot from optional artifacts."""

from __future__ import annotations

import argparse
import csv
import os


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--labels-v3", default="tables/object_phase_response_labels_v3_candidate.csv")
    ap.add_argument("--batch4-precheck", default="tables/object_phase_response_batch4_precheck_summary.csv")
    ap.add_argument("--batch4-summary", default="tables/object_phase_response_batch4_vis_summary.csv")
    ap.add_argument("--phase-e-qpos-audit", default="tables/phaseE_qpos_cache_audit_v0.csv")
    ap.add_argument("--detector-v2-metrics", default="tables/detector_v2_metrics.csv")
    ap.add_argument("--detector-v3-metrics", default="tables/detector_v3_object_hardneg_metrics.csv")
    ap.add_argument("--output-report", default="reports/EXPERIMENT_STATUS_SNAPSHOT_AUTO.md")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def count_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def status(path):
    n = count_rows(path)
    return "MISSING" if n is None else f"present ({n} rows)"


def main():
    args = parse_args()
    items = [
        ("Batch3b / labels_v2", args.labels_v2),
        ("labels_v3 candidate", args.labels_v3),
        ("Batch4 precheck", args.batch4_precheck),
        ("Batch4 VIS summary", args.batch4_summary),
        ("Phase E qpos audit", args.phase_e_qpos_audit),
        ("detector v2 metrics", args.detector_v2_metrics),
        ("detector v3 metrics", args.detector_v3_metrics),
    ]
    lines = [
        "# Experiment Status Snapshot Auto",
        "",
        "CPU-only status snapshot. No GPU, rollout, VIS, watcher, or detector training was run.",
        "",
        "## Artifact Status",
        "",
    ]
    lines.extend(f"- {name}: {status(path)}" for name, path in items)
    lines.extend([
        "",
        "## GPU Blacklist",
        "",
        "- GPU3 and GPU7 remain blacklisted.",
        "",
        "## Next Actions",
        "",
        "- Wait for Batch4 full VIS outputs, then run closeout and labels_v3 builder.",
        "- Run Phase E qpos cache audit before aligned-window generator or canary.",
    ])
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if args.dry_run:
        print(f"DRY RUN: wrote {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
