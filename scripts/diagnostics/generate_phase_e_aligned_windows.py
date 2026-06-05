#!/usr/bin/env python3
"""Generate qpos-aware Phase E low-budget VIS candidate windows.

CPU-only. This script enumerates compressed subwindows from full-VIS parent
windows and recommends only phase-aligned windows with explicit qpos support.
It does not run rollout, VIS, watcher jobs, GPU work, or detector training.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from collections import Counter


LENGTHS = [8, 10, 12]
POSITIONS = [
    "parent_start_aligned",
    "parent_start_plus_2",
    "parent_start_plus_4",
    "centered",
    "parent_end_aligned",
]
OUTPUT_FIELDS = [
    "task_key",
    "state_id",
    "parent_window_start",
    "parent_window_end",
    "subwindow_start",
    "subwindow_end",
    "window_start",
    "window_end",
    "compressed_len",
    "position_rule",
    "full_vis_label",
    "source_batch",
    "phase_bin_proxy",
    "qpos_source",
    "mujoco_qpos_mean",
    "mujoco_qpos_min",
    "mujoco_qpos_max",
    "obs_qpos_mean",
    "qpos_source_warning",
    "qpos_phase_class",
    "true_closed_score",
    "natural_open_score",
    "phase_proxy_mismatch",
    "clean_open_ratio",
    "clean_open_streak",
    "clean_done",
    "random_open_ratio",
    "random_done",
    "denominator_status",
    "provenance_status",
    "phase_alignment_source",
    "recommended_for_phaseE",
    "reason",
]

QPOS_MUJOCO_FIELDS = [
    "gripper_qpos_mujoco",
    "mujoco_qpos_mean",
    "qpos_mujoco",
    "qpos_pre_mujoco",
    "qpos_pre_step_mujoco",
]
QPOS_OBS_FIELDS = [
    "gripper_qpos_obs",
    "obs_qpos_mean",
    "robot0_gripper_qpos",
    "qpos_obs",
]
QPOS_USED_FIELDS = [
    "gripper_qpos_used",
    "qpos_pre_step",
    "qpos_pre",
    "qpos_before_attack",
    "qpos_start",
]
STEP_FIELDS = ["step", "timestep", "time_step", "env_step"]
DENOMINATOR_FIELDS = ["denominator_status", "denominator_type", "denominator_clean"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="tables/fast_vis_calibration_candidates_v0.csv")
    ap.add_argument("--labels-csv", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--trace-root", action="append", default=["/data/liuyu/outputs"])
    ap.add_argument("--output-csv", default="tables/phaseE_aligned_windows_v0.csv")
    ap.add_argument("--output-report", default="reports/PHASE_E_ALIGNED_WINDOWS_V0.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--closed-threshold", type=float, default=0.015)
    ap.add_argument("--open-threshold", type=float, default=0.005)
    return ap.parse_args()


def norm(value):
    return str(value if value is not None else "").strip()


def lower(value):
    return norm(value).lower()


def parse_int(value, default=0):
    try:
        text = norm(value)
        if text == "":
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_float(value):
    try:
        text = norm(value)
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    v = lower(value)
    if v in {"true", "1", "yes", "y", "clean"}:
        return True
    if v in {"false", "0", "no", "n", "polluted", "failed"}:
        return False
    return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(field).lstrip("\ufeff") for field in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({norm(k).lstrip("\ufeff"): v for k, v in row.items()})
        return fields, rows


def key(row, start_field="window_start", end_field="window_end"):
    return (
        norm(row.get("task_key")),
        norm(row.get("state_id")),
        norm(row.get(start_field)),
        norm(row.get(end_field)),
    )


def candidate_key(row):
    return (
        norm(row.get("task_key")),
        norm(row.get("state_id")),
        norm(row.get("parent_window_start")),
        norm(row.get("parent_window_end")),
    )


def first_float(row, fields):
    for field in fields:
        value = parse_float(row.get(field))
        if value is not None:
            return value
    return None


def first_int(row, fields):
    for field in fields:
        value = parse_float(row.get(field))
        if value is not None:
            return int(value)
    return None


def first_value(row, fields):
    for field in fields:
        value = norm(row.get(field))
        if value != "":
            return value
    return ""


def subwindow(parent_start, parent_end, length, position):
    parent_start = int(parent_start)
    parent_end = int(parent_end)
    length = int(length)
    if position == "parent_start_aligned":
        start = parent_start
    elif position == "parent_start_plus_2":
        start = parent_start + 2
    elif position == "parent_start_plus_4":
        start = parent_start + 4
    elif position == "parent_end_aligned":
        start = parent_end - length
    else:
        center = (parent_start + parent_end) // 2
        start = center - length // 2
    start = max(0, start)
    end = start + length
    if end > parent_end:
        end = parent_end
        start = max(0, end - length)
    return start, end


def denominator_status(row):
    status = lower(first_value(row, DENOMINATOR_FIELDS))
    if status in {"clean", "true", "1", "gold", "matched_random_clean_required"}:
        return "clean"
    if any(token in status for token in ["polluted", "failed", "random_failed", "denominator_failed"]):
        return "polluted"
    if status:
        return "manual_review"
    return "missing"


def classify_phase(row, args):
    mujoco = first_float(row, QPOS_MUJOCO_FIELDS)
    mujoco_min = parse_float(row.get("mujoco_qpos_min"))
    mujoco_max = parse_float(row.get("mujoco_qpos_max"))
    obs = first_float(row, QPOS_OBS_FIELDS)
    used = first_float(row, QPOS_USED_FIELDS)
    qpos_source = "missing"
    qpos_warning = ""
    qpos_value = None
    if mujoco is not None:
        qpos_value = mujoco
        qpos_source = "mujoco_trace"
        if obs is not None and abs(mujoco - obs) > 1e-3:
            qpos_warning = "mujoco_obs_qpos_mismatch"
    elif obs is not None:
        qpos_value = obs
        qpos_source = "obs_trace"
    elif used is not None:
        qpos_value = used
        qpos_source = "label_qpos_used"

    if qpos_value is None:
        return {
            "qpos_source": "missing",
            "mujoco_qpos_mean": "",
            "mujoco_qpos_min": "" if mujoco_min is None else f"{mujoco_min:.6g}",
            "mujoco_qpos_max": "" if mujoco_max is None else f"{mujoco_max:.6g}",
            "obs_qpos_mean": "" if obs is None else f"{obs:.6g}",
            "qpos_source_warning": "MISSING_QPOS_TRACE",
            "qpos_phase_class": "missing",
            "true_closed_score": "0",
            "natural_open_score": "0",
            "phase_alignment_source": "missing_trace",
        }

    true_closed_score = max(0.0, min(1.0, (qpos_value - args.open_threshold) / max(args.closed_threshold - args.open_threshold, 1e-6)))
    natural_open_score = max(0.0, min(1.0, (args.closed_threshold - qpos_value) / max(args.closed_threshold - args.open_threshold, 1e-6)))
    if qpos_value >= args.closed_threshold:
        phase_class = "true_closed"
    elif qpos_value <= args.open_threshold:
        phase_class = "natural_open"
    else:
        phase_class = "transitional-pre-open"
    return {
        "qpos_source": qpos_source,
        "mujoco_qpos_mean": "" if mujoco is None else f"{mujoco:.6g}",
        "mujoco_qpos_min": "" if (mujoco_min is None and mujoco is None) else f"{(mujoco_min if mujoco_min is not None else mujoco):.6g}",
        "mujoco_qpos_max": "" if (mujoco_max is None and mujoco is None) else f"{(mujoco_max if mujoco_max is not None else mujoco):.6g}",
        "obs_qpos_mean": "" if obs is None else f"{obs:.6g}",
        "qpos_source_warning": qpos_warning,
        "qpos_phase_class": phase_class,
        "true_closed_score": f"{true_closed_score:.6g}",
        "natural_open_score": f"{natural_open_score:.6g}",
        "phase_alignment_source": "mujoco_trace" if qpos_source == "mujoco_trace" else "obs_trace",
    }


def discover_trace_rows(task_key, state_id, trace_roots, notes, max_files=2000):
    task_key = norm(task_key)
    state_text = norm(state_id)
    state_tokens = {f"s{state_text}", f"state{state_text}", f"state_{state_text}"}
    scanned = 0
    for root in trace_roots or []:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.csv"):
            scanned += 1
            if scanned > max_files:
                notes.append(f"trace scan limit reached under {root_path}; qpos may be incomplete")
                return []
            low_path = str(path).lower()
            if task_key.lower() not in low_path:
                continue
            if not any(token in low_path for token in state_tokens):
                continue
            try:
                _, rows = read_csv(path)
            except Exception as exc:
                notes.append(f"trace CSV read failed: {path}: {str(exc)[:80]}")
                continue
            has_step = any(any(field in row for field in STEP_FIELDS) for row in rows[:5])
            has_qpos = any(
                any(field in row for field in QPOS_MUJOCO_FIELDS + QPOS_OBS_FIELDS + QPOS_USED_FIELDS)
                for row in rows[:5]
            )
            if has_step and has_qpos:
                notes.append(f"trace qpos source for {task_key}_s{state_id}: {path}")
                return rows
    return []


def qpos_overlay_from_trace(trace_rows, start, end):
    if not trace_rows:
        return {}
    mujoco_vals = []
    obs_vals = []
    used_vals = []
    for row in trace_rows:
        step = first_int(row, STEP_FIELDS)
        if step is None or step < start or step >= end:
            continue
        mujoco = first_float(row, QPOS_MUJOCO_FIELDS)
        obs = first_float(row, QPOS_OBS_FIELDS)
        used = first_float(row, QPOS_USED_FIELDS)
        if mujoco is not None:
            mujoco_vals.append(mujoco)
        if obs is not None:
            obs_vals.append(obs)
        if used is not None:
            used_vals.append(used)
    overlay = {}
    if mujoco_vals:
        overlay["gripper_qpos_mujoco"] = str(sum(mujoco_vals) / len(mujoco_vals))
        overlay["mujoco_qpos_mean"] = str(sum(mujoco_vals) / len(mujoco_vals))
        overlay["mujoco_qpos_min"] = str(min(mujoco_vals))
        overlay["mujoco_qpos_max"] = str(max(mujoco_vals))
    if obs_vals:
        overlay["gripper_qpos_obs"] = str(sum(obs_vals) / len(obs_vals))
        overlay["obs_qpos_mean"] = str(sum(obs_vals) / len(obs_vals))
    if used_vals:
        overlay["gripper_qpos_used"] = str(sum(used_vals) / len(used_vals))
    return overlay


def phase_proxy_mismatch(row, phase_class):
    proxy = lower(row.get("phase_bin_proxy"))
    if phase_class == "natural_open" and any(token in proxy for token in ["closed", "grasp", "contact", "lock"]):
        return True
    if phase_class == "true_closed" and any(token in proxy for token in ["release", "post", "open"]):
        return True
    return False


def recommend(row):
    if row["qpos_phase_class"] == "missing":
        return False, "rejected_missing_qpos"
    if row["qpos_phase_class"] == "natural_open":
        return False, "rejected_natural_open"
    if row["qpos_phase_class"] not in {"true_closed", "transitional-pre-open"}:
        return False, "rejected_not_true_closed_or_transitional"
    if row["denominator_status"] == "polluted":
        return False, "rejected_denominator_polluted"
    if row["phase_proxy_mismatch"] == "true":
        return False, "rejected_phase_proxy_mismatch"
    if "infra_failed" in lower(row.get("provenance_status")):
        return False, "rejected_infra_failed_provenance"
    if row["qpos_phase_class"] == "true_closed":
        return True, "phase_aligned_true_closed"
    true_closed_score = parse_float(row.get("true_closed_score"))
    if true_closed_score is None:
        true_closed_score = 0.0
    if true_closed_score >= 0.35:
        return True, "phase_aligned_transitional_pre_open"
    return False, "rejected_transitional_low_true_closed_score"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows, notes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    recommended = [row for row in rows if row["recommended_for_phaseE"] == "true"]
    missing = [row for row in rows if row["qpos_phase_class"] == "missing"]
    phase_counts = Counter(row.get("qpos_phase_class", "missing") for row in rows)
    lines = [
        "# Phase E Aligned Windows V0",
        "",
        f"**Candidates source**: `{args.candidates}`",
        f"**Labels source**: `{args.labels_csv}`",
        f"**closed_threshold**: {args.closed_threshold}",
        f"**open_threshold**: {args.open_threshold}",
        f"**Rows generated**: {len(rows)}",
        f"**Recommended for Phase E**: {len(recommended)}",
        f"**Missing qpos rows**: {len(missing)}",
        f"**Dry run**: {args.dry_run}",
        "",
        "This is a CPU-only candidate audit. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.",
        "",
        "## Notes",
        "",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Qpos Phase Rule",
            "",
            f"- `qpos >= {args.closed_threshold}`: `true_closed`.",
            f"- `qpos <= {args.open_threshold}`: `natural_open`.",
            f"- Otherwise: `transitional-pre-open`.",
            "- `true_closed` may be recommended when denominator/provenance/mismatch gates pass.",
            "- `transitional-pre-open` may be recommended when `true_closed_score >= 0.35` and gates pass.",
            "- `natural_open` and missing-qpos rows are rejected.",
            "",
            "## Qpos Phase Counts",
            "",
            f"- `true_closed`: {phase_counts.get('true_closed', 0)}",
            f"- `transitional-pre-open`: {phase_counts.get('transitional-pre-open', 0)}",
            f"- `natural_open`: {phase_counts.get('natural_open', 0)}",
            f"- `missing`: {phase_counts.get('missing', 0)}",
            "",
            "## Selection Rule",
            "",
            "- Do not assume centered L10 is valid.",
            "- Recommend true_closed windows directly after denominator/provenance/mismatch gates.",
            "- Recommend transitional-pre-open windows only when true_closed_score is at least 0.35.",
            "- MuJoCo qpos is preferred; obs qpos is fallback; missing qpos is never auto-recommended.",
            "- Polluted denominators, severe phase proxy mismatch, and infra-failed provenance block recommendation.",
            "",
            "## Trace Root Guidance",
            "",
            "- Broad `/data/liuyu/outputs` scans may miss traces because the script caps CSV scanning for safety.",
            "- Prefer specific trace roots when available:",
            "  - `/data/liuyu/outputs/nightly_object_batch3_20260604`",
            "  - `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604`",
            "  - `/data/liuyu/outputs/object_phase_response_batch4_...`",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    notes = []
    if not os.path.exists(args.candidates):
        write_csv(args.output_csv, [])
        write_report(args.output_report, args, [], [f"candidate CSV not found: {args.candidates}"])
        return 0

    _, candidate_rows = read_csv(args.candidates)
    labels_by_parent = {}
    if os.path.exists(args.labels_csv):
        _, label_rows = read_csv(args.labels_csv)
        labels_by_parent = {key(row): row for row in label_rows}
    else:
        notes.append(f"labels CSV not found: {args.labels_csv}; qpos/denominator fields may be missing")

    out = []
    seen = set()
    trace_cache = {}
    for candidate in candidate_rows:
        parent_start = parse_int(candidate.get("parent_window_start"))
        parent_end = parse_int(candidate.get("parent_window_end"))
        if parent_end <= parent_start:
            notes.append(f"invalid parent window for {candidate_key(candidate)}")
            continue
        label_row = labels_by_parent.get(candidate_key(candidate), {})
        merged = dict(candidate)
        for k, v in label_row.items():
            if norm(merged.get(k)) == "" and norm(v) != "":
                merged[k] = v
        trace_key = (norm(merged.get("task_key")), norm(merged.get("state_id")))
        if trace_key not in trace_cache:
            trace_cache[trace_key] = discover_trace_rows(trace_key[0], trace_key[1], args.trace_root, notes)
        for length in LENGTHS:
            for position in POSITIONS:
                start, end = subwindow(parent_start, parent_end, length, position)
                row_key = (norm(merged.get("task_key")), norm(merged.get("state_id")), str(start), str(end), str(length), position)
                if row_key in seen:
                    continue
                seen.add(row_key)
                phase_row = dict(merged)
                phase_row.update(qpos_overlay_from_trace(trace_cache.get(trace_key, []), start, end))
                phase = classify_phase(phase_row, args)
                denom = denominator_status(merged)
                mismatch = phase_proxy_mismatch(merged, phase["qpos_phase_class"])
                row = {
                    "task_key": norm(merged.get("task_key")),
                    "state_id": norm(merged.get("state_id")),
                    "parent_window_start": str(parent_start),
                    "parent_window_end": str(parent_end),
                    "subwindow_start": str(start),
                    "subwindow_end": str(end),
                    "window_start": str(start),
                    "window_end": str(end),
                    "compressed_len": str(end - start),
                    "position_rule": position,
                    "full_vis_label": norm(merged.get("full_vis_label") or merged.get("label_vulnerability_ready")),
                    "source_batch": norm(merged.get("source_batch")),
                    "phase_bin_proxy": norm(merged.get("phase_bin_proxy")),
                    **phase,
                    "phase_proxy_mismatch": "true" if mismatch else "false",
                    "clean_open_ratio": norm(merged.get("clean_open_ratio")),
                    "clean_open_streak": norm(merged.get("clean_open_streak")),
                    "clean_done": norm(merged.get("clean_done")),
                    "random_open_ratio": norm(merged.get("random_open_ratio")),
                    "random_done": norm(merged.get("random_done")),
                    "denominator_status": denom,
                    "provenance_status": norm(merged.get("provenance_status")) or "missing",
                }
                ok, reason = recommend(row)
                row["recommended_for_phaseE"] = "true" if ok else "false"
                row["reason"] = reason
                out.append(row)

    write_csv(args.output_csv, out)
    write_report(args.output_report, args, out, notes)
    if args.dry_run:
        print(f"DRY RUN: generated {len(out)} Phase E aligned-window rows")
        for row in out[:12]:
            print(
                f"  {row['task_key']}_s{row['state_id']} "
                f"{row['position_rule']} L{row['compressed_len']} "
                f"[{row['subwindow_start']},{row['subwindow_end']}] "
                f"qpos={row['qpos_phase_class']} recommend={row['recommended_for_phaseE']} reason={row['reason']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
