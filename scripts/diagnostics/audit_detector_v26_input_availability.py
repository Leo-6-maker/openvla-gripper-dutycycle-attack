#!/usr/bin/env python3
"""Audit detector-v2.6/two-stage input availability in a snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter


FIELDS = ["path", "exists", "rows", "columns_present", "notes"]
PHASE_HINTS = [
    "phase_bin_proxy", "phase_label", "predicted_phase", "phase_confidence",
    "qpos_phase_class", "true_closed", "natural_open", "stable_post_lock",
    "far_too_early", "ProprioNoStep", "proprionostep",
]
CLEAN_HINTS = [
    "far_too_early", "stable_post_lock", "natural_open", "no_contact",
    "far_from_object", "after_done", "terminal", "post_lock",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-root", default="/data/liuyu/outputs/shared_detector_v25_inputs_20260606")
    ap.add_argument("--output-csv", default="tables/detector_v26_input_availability_audit.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_V26_INPUT_AVAILABILITY_AUDIT.md")
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def read_head(path, limit=200):
    if not os.path.exists(path) or not path.endswith(".csv"):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append({norm(k).lstrip("\ufeff"): v for k, v in row.items()})
        return [norm(x).lstrip("\ufeff") for x in (reader.fieldnames or [])], rows


def count_rows(path):
    if not os.path.exists(path) or not path.endswith(".csv"):
        return ""
    with open(path, newline="", encoding="utf-8") as f:
        return str(max(0, sum(1 for _ in f) - 1))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def audit_file(path):
    fields, rows = read_head(path)
    columns = set(fields)
    notes = []
    phase_cols = [c for c in fields if any(h.lower() in c.lower() for h in PHASE_HINTS)]
    clean_cols = [c for c in fields if any(h.lower() in c.lower() for h in CLEAN_HINTS)]
    if phase_cols:
        notes.append("phase_columns:" + ",".join(phase_cols[:8]))
    if clean_cols:
        notes.append("clean_control_columns:" + ",".join(clean_cols[:8]))
    text = " ".join(" ".join(str(v).lower() for v in r.values()) for r in rows)
    value_hits = [h for h in CLEAN_HINTS + PHASE_HINTS if h.lower() in text]
    if value_hits:
        notes.append("value_hits:" + ",".join(sorted(set(value_hits))[:12]))
    return {
        "path": path,
        "exists": str(os.path.exists(path)).lower(),
        "rows": count_rows(path),
        "columns_present": ",".join(fields),
        "notes": "; ".join(notes),
    }


def main():
    args = parse_args()
    root = args.snapshot_root
    expected = [
        "object_phase_response_labels_v2.csv",
        "jobs_state.csv",
        "object_phase_response_adaptive_candidates.csv",
        "adaptive_1r_silver_positive_quality_audit.csv",
        "adaptive_vis_1r_screening_summary.csv",
        "adaptive_vis_1r_provenance.csv",
        "vis_1r_vs_3r_calibration.csv",
        "VIS_1R_VS_3R_CALIBRATION.md",
        "MANIFEST.json",
        "calib_1r_summary.csv",
        "calib_3r_summary.csv",
    ]
    rows = [audit_file(os.path.join(root, name)) for name in expected]
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if path not in {r["path"] for r in rows} and any(tok in name.lower() for tok in ["phase", "proprio", "autowindow", "qpos"]):
                rows.append(audit_file(path))
    write_csv(args.output_csv, rows)
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    present = [r for r in rows if r["exists"] == "true"]
    missing = [r for r in rows if r["exists"] != "true"]
    phase = [r for r in present if "phase" in r["notes"].lower() or "qpos" in r["notes"].lower()]
    clean = [r for r in present if "clean_control" in r["notes"].lower() or any(h in r["notes"] for h in CLEAN_HINTS)]
    lines = [
        "# Detector V2.6 Input Availability Audit",
        "",
        f"**Snapshot root**: `{root}`",
        f"**Files audited**: {len(rows)}",
        f"**Present**: {len(present)}",
        f"**Missing**: {len(missing)}",
        f"**Phase-signal files**: {len(phase)}",
        f"**Clean-control-signal files**: {len(clean)}",
        "",
        "CPU-only read-only audit. No GPU, VIS, rollout, watcher, or live DeepSeek output was touched.",
        "",
        "## Missing Files",
        "",
    ]
    lines.extend(f"- `{os.path.basename(r['path'])}`" for r in missing) if missing else lines.append("- None.")
    lines.extend(["", "## Present Files", ""])
    for r in present:
        lines.append(f"- `{os.path.basename(r['path'])}`: rows={r['rows']}; {r['notes'] or 'no phase/control hints'}")
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"audited={len(rows)} present={len(present)} missing={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
