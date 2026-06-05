#!/usr/bin/env python3
"""Audit Phase E MuJoCo qpos cache before aligned-window generation.

CPU-only. Does not run rollout, VIS, watcher jobs, GPU work, or training.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


REQUIRED_COLUMNS = {
    "step",
    "task_key",
    "state_id",
    "gripper_qpos_mujoco",
    "robot0_gripper_qpos_obs",
    "qpos_source",
    "clean_raw_gripper",
    "clean_env_gripper_after_transform",
    "clean_open_flag",
    "done",
    "provenance_status",
}
OUTPUT_FIELDS = [
    "task_key",
    "state_id",
    "cache_path",
    "file_exists",
    "row_count",
    "step_min",
    "step_max",
    "parent_window_start",
    "parent_window_end",
    "covers_parent_window",
    "has_mujoco_qpos",
    "mujoco_qpos_nonempty_count",
    "mujoco_qpos_min",
    "mujoco_qpos_max",
    "mujoco_qpos_mean",
    "obs_qpos_nonempty_count",
    "obs_all_zero",
    "mujoco_obs_mismatch_count",
    "true_closed_count",
    "transitional_count",
    "natural_open_count",
    "provenance_status",
    "cache_status",
    "reason",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="/data/liuyu/outputs/phaseE_mujoco_qpos_cache_20260605")
    ap.add_argument("--candidates", default="tables/fast_vis_calibration_candidates_v0.csv")
    ap.add_argument("--output-csv", default="tables/phaseE_qpos_cache_audit_v0.csv")
    ap.add_argument("--output-report", default="reports/PHASE_E_QPOS_CACHE_AUDIT_V0.md")
    ap.add_argument("--closed-threshold", type=float, default=0.015)
    ap.add_argument("--open-threshold", type=float, default=0.005)
    ap.add_argument("--dry-run", action="store_true")
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


def parse_int(value):
    value = parse_float(value)
    return None if value is None else int(value)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(field).lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = [{norm(k).lstrip("\ufeff"): v for k, v in row.items()} for row in reader]
    return fields, rows


def mean(values):
    return "" if not values else f"{sum(values) / len(values):.6g}"


def phase_counts(values, closed_threshold, open_threshold):
    true_closed = sum(1 for v in values if v >= closed_threshold)
    natural_open = sum(1 for v in values if v <= open_threshold)
    transitional = sum(1 for v in values if open_threshold < v < closed_threshold)
    return true_closed, transitional, natural_open


def classify_cache(path, candidate, args):
    base = {
        "task_key": norm(candidate.get("task_key")),
        "state_id": norm(candidate.get("state_id")),
        "cache_path": str(path),
        "file_exists": "false",
        "row_count": "0",
        "step_min": "",
        "step_max": "",
        "parent_window_start": norm(candidate.get("parent_window_start")),
        "parent_window_end": norm(candidate.get("parent_window_end")),
        "covers_parent_window": "false",
        "has_mujoco_qpos": "false",
        "mujoco_qpos_nonempty_count": "0",
        "mujoco_qpos_min": "",
        "mujoco_qpos_max": "",
        "mujoco_qpos_mean": "",
        "obs_qpos_nonempty_count": "0",
        "obs_all_zero": "false",
        "mujoco_obs_mismatch_count": "0",
        "true_closed_count": "0",
        "transitional_count": "0",
        "natural_open_count": "0",
        "provenance_status": "missing_file",
        "cache_status": "missing_file",
        "reason": "qpos_trace.csv not found",
    }
    if not path.exists():
        return base

    base["file_exists"] = "true"
    fields, rows = read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(fields))
    base["row_count"] = str(len(rows))
    if missing:
        base["cache_status"] = "manual_review"
        base["reason"] = "missing_required_columns:" + ",".join(missing)
        return base

    steps = [parse_int(row.get("step")) for row in rows]
    steps = [s for s in steps if s is not None]
    if steps:
        base["step_min"] = str(min(steps))
        base["step_max"] = str(max(steps))
    parent_start = parse_int(candidate.get("parent_window_start"))
    parent_end = parse_int(candidate.get("parent_window_end"))
    if parent_start is not None and parent_end is not None and steps:
        base["covers_parent_window"] = "true" if min(steps) <= parent_start and max(steps) >= parent_end else "false"

    mujoco_vals = [parse_float(row.get("gripper_qpos_mujoco")) for row in rows]
    mujoco_vals = [v for v in mujoco_vals if v is not None]
    obs_vals = [parse_float(row.get("robot0_gripper_qpos_obs")) for row in rows]
    obs_vals = [v for v in obs_vals if v is not None]
    base["mujoco_qpos_nonempty_count"] = str(len(mujoco_vals))
    base["obs_qpos_nonempty_count"] = str(len(obs_vals))
    base["has_mujoco_qpos"] = "true" if mujoco_vals else "false"
    if mujoco_vals:
        base["mujoco_qpos_min"] = f"{min(mujoco_vals):.6g}"
        base["mujoco_qpos_max"] = f"{max(mujoco_vals):.6g}"
        base["mujoco_qpos_mean"] = mean(mujoco_vals)
        t, tr, n = phase_counts(mujoco_vals, args.closed_threshold, args.open_threshold)
        base["true_closed_count"] = str(t)
        base["transitional_count"] = str(tr)
        base["natural_open_count"] = str(n)
    if obs_vals:
        base["obs_all_zero"] = "true" if all(abs(v) <= 1e-8 for v in obs_vals) else "false"
    mismatch = 0
    for row in rows:
        m = parse_float(row.get("gripper_qpos_mujoco"))
        o = parse_float(row.get("robot0_gripper_qpos_obs"))
        if m is not None and o is not None and abs(m - o) > 1e-3:
            mismatch += 1
    base["mujoco_obs_mismatch_count"] = str(mismatch)
    provenance_values = sorted({lower(row.get("provenance_status")) for row in rows if norm(row.get("provenance_status"))})
    provenance = "|".join(provenance_values) if provenance_values else "missing"
    base["provenance_status"] = provenance

    if any("infra_failed" in p or "xid" in p or "oom" in p for p in provenance_values):
        base["cache_status"] = "infra_failed"
        base["reason"] = "cache provenance has infra failure"
    elif not mujoco_vals and obs_vals:
        base["cache_status"] = "obs_only_untrusted"
        base["reason"] = "obs-only qpos is audit-only and cannot recommend Phase E"
    elif not mujoco_vals:
        base["cache_status"] = "missing_mujoco_qpos"
        base["reason"] = "missing MuJoCo gripper qpos"
    elif base["covers_parent_window"] != "true":
        base["cache_status"] = "insufficient_steps"
        base["reason"] = "cache does not cover parent window"
    else:
        base["cache_status"] = "ok"
        base["reason"] = "usable MuJoCo qpos cache"
    return base


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows, status):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    usable = [r for r in rows if r["cache_status"] == "ok"]
    obs_only = [r for r in rows if r["cache_status"] == "obs_only_untrusted"]
    missing = [r for r in rows if r["cache_status"] in {"missing_file", "missing_mujoco_qpos"}]
    lines = [
        "# Phase E Qpos Cache Audit V0",
        "",
        f"**Status**: {status}",
        f"**Cache root**: `{args.cache_root}`",
        f"**Candidates**: `{args.candidates}`",
        f"**Rows audited**: {len(rows)}",
        f"**Cache usable?**: {'yes' if rows and len(usable) == len(rows) else 'no'}",
        f"**Usable MuJoCo task/states**: {len(usable)}",
        f"**Obs-only task/states**: {len(obs_only)}",
        f"**Missing task/states**: {len(missing)}",
        f"**Phase E generator can safely rerun?**: {'yes' if usable else 'no'}",
        "",
        "This is CPU-only. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.",
        "",
        "## Rules",
        "",
        "- Obs-only qpos is not acceptable for Phase E recommendation.",
        "- All-zero obs qpos is flagged as untrusted.",
        "- Missing MuJoCo qpos is `missing_mujoco_qpos`.",
        "- Cache must cover the parent calibration windows.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    if not os.path.exists(args.candidates):
        write_csv(args.output_csv, [])
        write_report(args.output_report, args, [], "BLOCKED_MISSING_CANDIDATES")
        return 0
    _, candidates = read_csv(args.candidates)
    cache_root = Path(args.cache_root)
    status = "OK"
    if not cache_root.exists():
        status = "BLOCKED_MISSING_QPOS_CACHE"
    rows = []
    for cand in candidates:
        path = cache_root / f"{norm(cand.get('task_key'))}_s{norm(cand.get('state_id'))}" / "qpos_trace.csv"
        rows.append(classify_cache(path, cand, args))
    write_csv(args.output_csv, rows)
    write_report(args.output_report, args, rows, status)
    if args.dry_run:
        print(f"DRY RUN: audited {len(rows)} Phase E qpos cache entries; status={status}")
        for row in rows[:10]:
            print(f"  {row['task_key']}_s{row['state_id']}: {row['cache_status']} {row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
