#!/usr/bin/env python3
"""Select or validate saved frames for no-rollout VIS diagnostics.

This script is read-only. It inspects existing ``step_records.jsonl`` files and
scores timesteps for contact/carry/pre-place relevance using clean rollout
signals already present in the artifacts. It also checks whether a matching
saved frame exists. It does not run LIBERO, OpenVLA, attacks, or training.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = [
    "run_id",
    "suite",
    "task_id",
    "task_name",
    "state_id",
    "seed",
    "step_records",
    "step_idx",
    "policy_step_idx",
    "phase",
    "score",
    "selector_status",
    "reason",
    "frame_path",
    "frame_available",
    "image_path",
    "image_path_available",
    "proxy_lift_carry_gate_active",
    "priv_lift_carry_gate_active",
    "grasp_gate_active",
    "trigger_request_active",
    "eef_z",
    "eef_z_delta_from_min",
    "gripper_qpos",
    "gripper_width",
    "action_gripper",
    "clean_gripper_token",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_frame_path(step_records: Path, row: dict) -> tuple[str, bool, str, bool]:
    image_path = str(row.get("image_path") or row.get("agentview_image_path") or row.get("frame_path") or row.get("rgb_path") or "")
    image_available = bool(image_path and Path(image_path).exists())
    candidates = []
    if image_path:
        candidates.append(Path(image_path))
    step_idx = int(_float(row.get("step_idx"), -1))
    if step_idx >= 0:
        candidates.append(step_records.parent / "frames" / f"step_{step_idx:04d}.png")
        candidates.append(step_records.parent / "frames" / f"step_{step_idx:05d}.png")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), True, image_path, image_available
    return (str(candidates[0]) if candidates else ""), False, image_path, image_available


def score_row(row: dict) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    phase = str(row.get("phase") or "").lower()
    policy_step = int(_float(row.get("policy_step_idx"), row.get("step_idx", -1)))
    if phase == "wait" or policy_step < 0:
        score -= 8.0
        reasons.append("wait_or_prepolicy")
    else:
        score += 1.0
        reasons.append("policy_step")
    for key, weight in (
        ("proxy_lift_carry_gate_active", 5.0),
        ("priv_lift_carry_gate_active", 5.0),
        ("grasp_gate_active", 3.0),
        ("proxy_grasp_gate_active", 3.0),
        ("trigger_request_active", 2.0),
        ("proxy_lift_carry_closed", 2.0),
    ):
        if _truthy(row.get(key)):
            score += weight
            reasons.append(key)
    if any(token in phase for token in ("contact", "carry", "place", "lift", "pre")):
        score += 3.0
        reasons.append(f"phase={phase}")
    eef_delta = _float(row.get("proxy_lift_carry_eef_z_delta_from_min"), _float(row.get("eef_z_delta_from_min"), 0.0))
    if eef_delta >= 0.04:
        score += 2.0
        reasons.append("eef_z_delta>=0.04")
    if _float(row.get("proxy_lift_carry_z_up_streak"), 0.0) >= 4:
        score += 1.0
        reasons.append("z_up_streak")
    if _truthy(row.get("success_done")) or _truthy(row.get("done")):
        score -= 2.0
        reasons.append("done_or_success")
    return score, reasons


def best_for_file(path: Path) -> dict:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty step_records: {path}")
    scored = []
    for row in rows:
        score, reasons = score_row(row)
        frame_path, frame_available, image_path, image_available = resolve_frame_path(path, row)
        scored.append((score, frame_available, row, reasons, frame_path, image_path, image_available))
    scored.sort(key=lambda item: (item[0], item[1], int(_float(item[2].get("step_idx"), -1))), reverse=True)
    score, frame_available, row, reasons, frame_path, image_path, image_available = scored[0]
    status = "candidate_frame_available" if frame_available and score > 0 else "candidate_missing_frame" if score > 0 else "no_contact_candidate"
    if status == "no_contact_candidate" and frame_available:
        status = "frame_available_but_not_contact"
    return {
        "run_id": row.get("run_id", path.parent.name),
        "suite": row.get("suite", ""),
        "task_id": row.get("task_id", ""),
        "task_name": row.get("task_name", row.get("base_instruction", "")),
        "state_id": row.get("state_id", ""),
        "seed": row.get("seed", ""),
        "step_records": str(path),
        "step_idx": row.get("step_idx", ""),
        "policy_step_idx": row.get("policy_step_idx", ""),
        "phase": row.get("phase", ""),
        "score": score,
        "selector_status": status,
        "reason": ";".join(reasons),
        "frame_path": frame_path,
        "frame_available": str(bool(frame_available)).lower(),
        "image_path": image_path,
        "image_path_available": str(bool(image_available)).lower(),
        "proxy_lift_carry_gate_active": row.get("proxy_lift_carry_gate_active", ""),
        "priv_lift_carry_gate_active": row.get("priv_lift_carry_gate_active", ""),
        "grasp_gate_active": row.get("grasp_gate_active", row.get("proxy_grasp_gate_active", "")),
        "trigger_request_active": row.get("trigger_request_active", ""),
        "eef_z": row.get("eef_z", row.get("proxy_lift_carry_eef_z", "")),
        "eef_z_delta_from_min": row.get("proxy_lift_carry_eef_z_delta_from_min", row.get("eef_z_delta_from_min", "")),
        "gripper_qpos": row.get("gripper_qpos", row.get("gripper_qpos_abs_sum_after", "")),
        "gripper_width": row.get("gripper_width", ""),
        "action_gripper": row.get("action_gripper", row.get("clean_gripper_raw", "")),
        "clean_gripper_token": row.get("clean_gripper_token", ""),
    }


def expand_inputs(paths: Iterable[str], globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        out.append(Path(item))
    for pattern in globs:
        out.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    seen = set()
    unique = []
    for path in out:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step_records", action="append", default=[])
    parser.add_argument("--step_records_glob", action="append", default=[])
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()
    if args.print_schema:
        for field in CSV_FIELDS:
            print(field)
    paths = expand_inputs(args.step_records, args.step_records_glob)
    rows = []
    for path in paths:
        try:
            rows.append(best_for_file(path))
        except Exception as exc:
            rows.append({"step_records": str(path), "selector_status": "error", "reason": str(exc)})
    write_rows(Path(args.output_csv), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
