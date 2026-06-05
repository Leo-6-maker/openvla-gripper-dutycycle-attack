#!/usr/bin/env python3
"""Audit labels_v2 qpos/phase consistency for Batch4 candidate selection.

CPU-only and CSV-only. The script marks true_closed, natural_open, and
phase_proxy_mismatch so Batch4 hard negatives can be MuJoCo-qpos verified.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


OUTPUT_FIELDS = [
    "task_key",
    "state_id",
    "window_start",
    "window_end",
    "source_batch",
    "candidate_role",
    "phase_bin_proxy",
    "label_status",
    "label_vulnerability_ready",
    "gripper_qpos_mujoco",
    "gripper_qpos_obs",
    "gripper_qpos_used",
    "gripper_qpos_source_priority",
    "true_closed",
    "natural_open",
    "phase_proxy_mismatch",
    "qpos_phase_status",
    "reason",
]

QPOS_MUJOCO_FIELDS = [
    "gripper_qpos_mujoco",
    "qpos_mujoco",
    "mujoco_gripper_qpos",
    "qpos_pre_mujoco",
    "qpos_pre_step_mujoco",
]
QPOS_OBS_FIELDS = [
    "gripper_qpos_obs",
    "robot0_gripper_qpos",
    "obs_robot0_gripper_qpos",
    "qpos_obs",
]
QPOS_USED_FIELDS = [
    "gripper_qpos_used",
    "qpos_pre_step",
    "qpos_pre",
    "qpos_before_attack",
    "qpos_start",
    "qpos_initial",
]
OPEN_HINT_FIELDS = [
    "clean_open_count",
    "natural_open_count",
    "open_count_full_window",
    "VIS_OPEN",
    "vis_open_count",
]
CLOSED_PHASE_TOKENS = ["grasp", "contact", "lock", "carry", "lift"]
OPEN_PHASE_TOKENS = ["release", "place", "post_lock", "open"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--output-csv", default="tables/labels_v2_mujoco_qpos_phase_audit.csv")
    ap.add_argument("--output-report", default="reports/LABELS_V2_MUJOCO_QPOS_PHASE_AUDIT.md")
    ap.add_argument("--closed-threshold", type=float, default=0.03)
    ap.add_argument("--open-threshold", type=float, default=0.008)
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def parse_float(value):
    try:
        text = norm(value)
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_count(value):
    text = norm(value)
    if not text:
        return None
    if "/" in text:
        text = text.split("/", 1)[0]
    return parse_float(text)


def first_float(row, fields):
    for field in fields:
        value = parse_float(row.get(field))
        if value is not None:
            return value
    return None


def first_count(row, fields):
    for field in fields:
        value = parse_count(row.get(field))
        if value is not None:
            return value
    return None


def bool_text(value):
    return "true" if value else "false"


def phase_class(row):
    text = " ".join(lower(row.get(field)) for field in ["phase_bin_proxy", "candidate_role", "control_type"])
    closed = any(token in text for token in CLOSED_PHASE_TOKENS)
    open_phase = any(token in text for token in OPEN_PHASE_TOKENS)
    if closed and not open_phase:
        return "closed_expected"
    if open_phase and not closed:
        return "open_expected"
    if closed and open_phase:
        return "mixed"
    return "unknown"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(field).lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({norm(k).lstrip("\ufeff"): v for k, v in row.items()})
        return fields, rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, status, rows, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    counts = Counter(row["qpos_phase_status"] for row in rows)
    lines = [
        "# Labels V2 MuJoCo Qpos Phase Audit",
        "",
        f"**Status**: {status}",
        f"**Input**: `{args.labels_csv}`",
        f"**Rows audited**: {len(rows)}",
        f"**true_closed rows**: {sum(1 for r in rows if r['true_closed'] == 'true')}",
        f"**natural_open rows**: {sum(1 for r in rows if r['natural_open'] == 'true')}",
        f"**phase_proxy_mismatch rows**: {sum(1 for r in rows if r['phase_proxy_mismatch'] == 'true')}",
        "",
        "This is a CPU-only CSV audit. It does not run rollout, VIS, GPU work, or detector training.",
        "",
        "## Status Counts",
        "",
    ]
    if counts:
        for key, count in sorted(counts.items()):
            lines.append(f"- `{key}`: {count}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Notes", ""])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `true_closed` is an input-quality marker for Batch4 candidate selection, not evidence of attack success.",
            "- `natural_open` windows are poor hard-negative candidates unless explicitly intended as controls.",
            "- `phase_proxy_mismatch` rows require manual review before entering detector training or Batch4 execution.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    if not os.path.exists(args.labels_csv):
        write_csv(args.output_csv, [])
        write_report(args.output_report, args, "BLOCKED_MISSING_LABELS_V2", [], [f"labels CSV not found: {args.labels_csv}"])
        return 0

    _, rows = read_csv(args.labels_csv)
    audited = []
    for row in rows:
        qpos_mujoco = first_float(row, QPOS_MUJOCO_FIELDS)
        qpos_obs = first_float(row, QPOS_OBS_FIELDS)
        qpos_used = first_float(row, QPOS_USED_FIELDS)
        if qpos_mujoco is not None:
            qpos_used = qpos_mujoco
            source_priority = "mujoco_primary"
        elif qpos_used is not None:
            source_priority = "explicit_used_no_mujoco"
        elif qpos_obs is not None:
            qpos_used = qpos_obs
            source_priority = "obs_fallback"
        else:
            source_priority = "missing_qpos"

        open_count = first_count(row, OPEN_HINT_FIELDS)
        true_closed = qpos_used is not None and qpos_used >= args.closed_threshold
        natural_open = (qpos_used is not None and qpos_used <= args.open_threshold) or (open_count is not None and open_count > 0)
        klass = phase_class(row)
        mismatch = (klass == "closed_expected" and natural_open) or (klass == "open_expected" and true_closed)
        if source_priority == "missing_qpos":
            status = "missing_qpos"
            reason = "no MuJoCo/obs/used gripper qpos field available"
        elif mismatch:
            status = "phase_proxy_mismatch"
            reason = f"phase={klass}, true_closed={true_closed}, natural_open={natural_open}"
        elif true_closed:
            status = "qpos_verified_true_closed"
            reason = f"qpos_used >= {args.closed_threshold}"
        elif natural_open:
            status = "natural_open"
            reason = f"qpos_used <= {args.open_threshold} or open-count hint present"
        else:
            status = "qpos_ambiguous"
            reason = "qpos is neither closed nor natural-open by threshold"

        audited.append(
            {
                "task_key": norm(row.get("task_key")),
                "state_id": norm(row.get("state_id")),
                "window_start": norm(row.get("window_start")),
                "window_end": norm(row.get("window_end")),
                "source_batch": norm(row.get("source_batch")),
                "candidate_role": norm(row.get("candidate_role")),
                "phase_bin_proxy": norm(row.get("phase_bin_proxy")),
                "label_status": norm(row.get("label_status")),
                "label_vulnerability_ready": norm(row.get("label_vulnerability_ready")),
                "gripper_qpos_mujoco": "" if qpos_mujoco is None else f"{qpos_mujoco:.6g}",
                "gripper_qpos_obs": "" if qpos_obs is None else f"{qpos_obs:.6g}",
                "gripper_qpos_used": "" if qpos_used is None else f"{qpos_used:.6g}",
                "gripper_qpos_source_priority": source_priority,
                "true_closed": bool_text(true_closed),
                "natural_open": bool_text(natural_open),
                "phase_proxy_mismatch": bool_text(mismatch),
                "qpos_phase_status": status,
                "reason": reason,
            }
        )

    status = "PASS_WITH_REVIEW_FLAGS"
    if any(row["qpos_phase_status"] == "missing_qpos" for row in audited):
        notes.append("Some rows lack MuJoCo/obs qpos; do not use them as qpos-verified Batch4 hard negatives.")
    if any(row["phase_proxy_mismatch"] == "true" for row in audited):
        notes.append("Some rows have phase proxy mismatch and need manual review.")
    write_csv(args.output_csv, audited)
    write_report(args.output_report, args, status, audited, notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
