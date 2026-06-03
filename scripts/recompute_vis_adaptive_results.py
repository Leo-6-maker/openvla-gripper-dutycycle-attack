# -*- coding: utf-8 -*-
"""Recompute VIS adaptive-controller trace summaries from completed CSV traces."""
import argparse
import csv
import math
import re
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from gripper_attack.gripper_semantics import raw_gripper_is_open


REMOTE_RUN_ROOT = "/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs"
REMOTE_LOG_ROOT = "/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/logs"
REQUIRED_P0_FIELDS = {
    "attack_attempted",
    "pgd_applied",
    "controller_active",
    "controller_stopped",
    "effective_attack_step_idx",
    "qpos_pre_step",
    "qpos_post_step",
}


NAME_RE = re.compile(
    r"^vis_(?P<task>.+?)_s0_(?P<condition>.+?)_(?P<strategy>full|sparse)_d(?P<duration>\d+)_"
    r"w(?P<window_start>\d+)_(?P<window_end>\d+)_seed(?P<seed>\d+)_"
    r"(?P<controller>open_streak_stop|min_hold_qpos_cap)_K(?P<K>[^_]+)_Q(?P<Q>[^_]+)_"
    r"md(?P<max_duration>[^_]+)_(?P<timestamp>\d+)_trace\.csv$"
)


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        f = float(value)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


def parse_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def longest_open_streak(rows):
    cur = 0
    best = 0
    for row in rows:
        if raw_gripper_is_open(parse_float(row.get("adv_grip"))):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def read_trace(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def infer_attacked_rows(window_rows, has_pgd_field):
    if has_pgd_field:
        return [r for r in window_rows if parse_bool(r.get("pgd_applied"))]
    attacked = []
    prev_attacks = 0
    for row in window_rows:
        ctrl_attacks = parse_int(row.get("ctrl_attacks"), 0)
        arm_l2 = parse_float(row.get("arm_l2"), 0.0)
        linf = parse_float(row.get("linf"), 0.0)
        if ctrl_attacks > prev_attacks or arm_l2 > 0.0 or linf > 0.0:
            attacked.append(row)
        prev_attacks = max(prev_attacks, ctrl_attacks)
    return attacked


def classify_trace(path, fieldnames, rows):
    if not rows:
        return "crashed", "empty_trace"
    missing_p0 = sorted(REQUIRED_P0_FIELDS.difference(fieldnames))
    if missing_p0:
        return "schema_incomplete", "missing_fields=" + "|".join(missing_p0)
    return "valid", ""


def summarize_trace(path):
    meta = NAME_RE.match(path.name)
    fieldnames, rows = read_trace(path)
    validity_status, validity_note = classify_trace(path, fieldnames, rows)
    if meta is None:
        validity_status = "needs_manual_audit"
        validity_note = (validity_note + "; " if validity_note else "") + "filename_parse_failed"
        groups = {}
    else:
        groups = meta.groupdict()

    window_rows = [r for r in rows if parse_bool(r.get("in_window"))]
    attacked_rows = infer_attacked_rows(window_rows, "pgd_applied" in fieldnames)
    full_flips = sum(1 for r in window_rows if parse_bool(r.get("token_flip")))
    attacked_flips = sum(1 for r in attacked_rows if parse_bool(r.get("token_flip")))
    open_full = sum(1 for r in window_rows if parse_float(r.get("adv_grip")) > 0.5)
    open_attacked = sum(1 for r in attacked_rows if parse_float(r.get("adv_grip")) > 0.5)
    final = rows[-1] if rows else {}
    success = any(parse_bool(r.get("done")) and parse_float(r.get("reward")) > 0 for r in rows)

    qpos_pre_values = [parse_float(r.get("qpos_pre_step", r.get("gripper_qpos"))) for r in attacked_rows]
    qpos_post_values = [parse_float(r.get("qpos_post_step")) for r in attacked_rows if r.get("qpos_post_step") not in (None, "")]
    qpos_delta_pre = max(qpos_pre_values) - min(qpos_pre_values) if qpos_pre_values else 0.0
    qpos_delta_post = max(qpos_post_values) - min(qpos_post_values) if qpos_post_values else ""
    arm_values = [parse_float(r.get("arm_l2")) for r in attacked_rows]
    arm_l2 = sum(arm_values) / len(arm_values) if arm_values else 0.0
    stop_reason = final.get("ctrl_stop_reason", "")
    if stop_reason == "none" and attacked_rows:
        stop_reason = "force_window_end"

    remote_path = f"{REMOTE_RUN_ROOT}/{path.name}"
    summary = {
        "task": groups.get("task", rows[0].get("task", "") if rows else ""),
        "condition": groups.get("condition", rows[0].get("condition", "") if rows else ""),
        "seed": groups.get("seed", rows[0].get("seed", "") if rows else ""),
        "controller": groups.get("controller", final.get("ctrl_mode", "")),
        "K": groups.get("K", ""),
        "Q": groups.get("Q", ""),
        "max_duration": groups.get("max_duration", ""),
        "duration": groups.get("duration", ""),
        "window_start": groups.get("window_start", ""),
        "window_end": groups.get("window_end", ""),
        "trace_timestamp": groups.get("timestamp", ""),
        "success": success,
        "window_steps": len(window_rows),
        "attacks_applied": len(attacked_rows),
        "token_flips_full_window": full_flips,
        "token_flips_attacked_steps": attacked_flips,
        "open_count": open_full,
        "open_count_full_window": open_full,
        "open_count_attacked_steps": open_attacked,
        "longest_open_streak": longest_open_streak(attacked_rows),
        "qpos_delta_pre": round(qpos_delta_pre, 6),
        "qpos_delta_post": qpos_delta_post if qpos_delta_post == "" else round(qpos_delta_post, 6),
        "arm_l2": round(arm_l2, 6),
        "stop_reason": stop_reason,
        "validity_status": validity_status,
        "validity_note": validity_note,
        "trace_path": remote_path,
    }
    audit = {
        "trace_file": path.name,
        "trace_path": remote_path,
        "log_file": "",
        "log_path": "",
        "rows": len(rows),
        "has_trace": bool(rows),
        "has_p0_denominator_fields": REQUIRED_P0_FIELDS.issubset(set(fieldnames)),
        "field_count": len(fieldnames),
        "missing_fields": "|".join(sorted(REQUIRED_P0_FIELDS.difference(fieldnames))),
        "validity_status": validity_status,
        "validity_note": validity_note,
        "needs_manual_audit": validity_status != "valid" or qpos_delta_post == "",
        "random_denominator_status": "random_denominator_missing",
        "crashed": validity_status == "crashed",
    }
    return summary, audit


def crashed_log_audits(log_dir):
    if not log_dir:
        return []
    audits = []
    for path in sorted(Path(log_dir).glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        crashed = any(marker in text for marker in ("Traceback", "RuntimeError", "CUDA error", "Xid"))
        completed = "Episode finished:" in text
        if not crashed or completed:
            continue
        audits.append({
            "trace_file": "",
            "trace_path": "",
            "log_file": path.name,
            "log_path": f"{REMOTE_LOG_ROOT}/{path.name}",
            "rows": 0,
            "has_trace": False,
            "has_p0_denominator_fields": False,
            "field_count": 0,
            "missing_fields": "|".join(sorted(REQUIRED_P0_FIELDS)),
            "validity_status": "crashed",
            "validity_note": "crash_log_without_completed_episode; missing_trace",
            "needs_manual_audit": True,
            "random_denominator_status": "random_denominator_missing",
            "crashed": True,
        })
    return audits


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--out-summary", default="tables/codex_recomputed_adaptive_result_summary.csv", type=Path)
    parser.add_argument("--out-audit", default="tables/codex_recomputed_trace_validity_audit.csv", type=Path)
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("*open_streak*trace.csv")) + sorted(args.trace_dir.glob("*min_hold*trace.csv"))
    summaries = []
    audits = []
    for path in sorted(paths):
        summary, audit = summarize_trace(path)
        summaries.append(summary)
        audits.append(audit)
    audits.extend(crashed_log_audits(args.log_dir))
    write_csv(args.out_summary, summaries)
    write_csv(args.out_audit, audits)
    print(f"wrote {len(summaries)} summaries to {args.out_summary}")
    print(f"wrote {len(audits)} audit rows to {args.out_audit}")


if __name__ == "__main__":
    main()
