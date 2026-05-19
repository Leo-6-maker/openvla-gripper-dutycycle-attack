#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


OPEN_THRESHOLD = -0.5
CQ_RULE_VERSION = "cq_v2_20260517"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def infer_condition(run_id: str) -> str:
    for condition in ("random_gripper_clean", "vis_arm_clean", "clean"):
        if condition in run_id:
            return condition
    return ""


def official_success(run_dir: Path) -> bool | str:
    episodes = read_jsonl(run_dir / "episode_records.jsonl")
    return bool(episodes[-1].get("success", False)) if episodes else ""


def env_open(row: dict) -> bool | None:
    if "executed_gripper_open_ok" in row:
        value = row.get("executed_gripper_open_ok")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
    for key in ("executed_gripper_env", "clean_gripper_env", "env_action_gripper"):
        value = to_float(row.get(key))
        if value is not None:
            return value < OPEN_THRESHOLD
    return None


def qpos(row: dict) -> float | None:
    for key in ("qpos_abs_after_max", "gripper_qpos_abs_sum_after", "qpos_abs_sum"):
        value = to_float(row.get(key))
        if value is not None:
            return abs(value)
    return None


def bowl_z(row: dict) -> float | None:
    for key in ("bowl_z_after", "object_z_after", "target_object_z_after", "bowl_z"):
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def object_pose_available(rows: list[dict]) -> bool:
    keys = ("bowl_z_after", "object_z_after", "target_object_z_after", "grasp_bowl_z_delta", "bowl_z")
    return any(row.get(key) not in (None, "") for row in rows for key in keys)


def max_open_streak(rows: list[dict]) -> int:
    best = cur = 0
    for row in rows:
        if env_open(row) is True:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def first_step(rows: list[dict], predicate) -> int | str:
    for idx, row in enumerate(rows):
        if predicate(row):
            value = row.get("step_idx", row.get("step", idx))
            try:
                return int(value)
            except Exception:
                return idx
    return ""


def contact_quality_v2(
    *,
    success: bool | str,
    pose_available: bool,
    object_lifted: bool | str,
    premature_release: bool,
    drop_after_lift: bool | str,
    unstable_transport: bool,
) -> dict:
    if not pose_available:
        return {
            "contact_quality_failure": "NA",
            "contact_quality_success": "NA",
            "sr_cq_mismatch": "NA",
            "failure_phase_auto": "",
            "failure_reason": "missing_object_pose_or_contact_proxy",
            "uncontrolled_final_drop": "NA",
            "stable_controlled_place": "NA",
            "cq_failure_reason_v2": "missing_pose_low_confidence",
            "cq_rule_version": CQ_RULE_VERSION,
            "cq_confidence_v2": "low",
        }

    uncontrolled_final_drop = bool(drop_after_lift is True and (premature_release or success is False))
    stable_controlled_place = bool(success and object_lifted is True and not uncontrolled_final_drop)
    cq_failure = bool(uncontrolled_final_drop or (premature_release and unstable_transport))
    cq_success = bool(object_lifted is True and stable_controlled_place and not cq_failure)
    mismatch = bool(success and cq_failure)

    reason = ""
    failure_phase = ""
    if cq_failure:
        failure_phase = "premature_release"
        if success is True:
            reason = "uncontrolled_final_drop_to_target"
        elif drop_after_lift is True:
            reason = "premature_release_plus_drop"
        elif unstable_transport:
            reason = "premature_release_plus_unstable_transport"
        else:
            reason = "uncontrolled_final_drop_to_target"
    elif stable_controlled_place:
        reason = "stable_controlled_place"
    elif success is False:
        reason = "official_failure_without_clear_contact_proxy"
        failure_phase = "unknown_failure"

    return {
        "contact_quality_failure": cq_failure,
        "contact_quality_success": cq_success,
        "sr_cq_mismatch": mismatch,
        "failure_phase_auto": failure_phase,
        "failure_reason": "" if reason == "stable_controlled_place" else reason,
        "uncontrolled_final_drop": uncontrolled_final_drop,
        "stable_controlled_place": stable_controlled_place,
        "cq_failure_reason_v2": reason,
        "cq_rule_version": CQ_RULE_VERSION,
        "cq_confidence_v2": "medium",
    }


def summarize_run(run_dir: Path) -> dict:
    run_id = run_dir.name
    manifest = read_json(run_dir / "run_manifest.json")
    rows = read_jsonl(run_dir / "step_records.jsonl")
    success = official_success(run_dir)
    opens = [env_open(row) for row in rows]
    opens = [value for value in opens if value is not None]
    qpos_values = [qpos(row) for row in rows]
    qpos_values = [value for value in qpos_values if value is not None]
    active_rows = [row for row in rows if row.get("attack_active")]
    active_opens = [env_open(row) for row in active_rows]
    active_opens = [value for value in active_opens if value is not None]

    pose_available = object_pose_available(rows)
    base_z = next((bowl_z(row) for row in rows if bowl_z(row) is not None), None)
    lifted_steps = []
    dropped_steps = []
    for row in rows:
        dz = to_float(row.get("grasp_bowl_z_delta"))
        z = bowl_z(row)
        lifted = (dz is not None and dz >= 0.018) or (z is not None and base_z is not None and z - base_z >= 0.018)
        if lifted:
            lifted_steps.append(row)
        if lifted_steps and z is not None and base_z is not None and z - base_z < 0.010:
            dropped_steps.append(row)

    object_lifted = bool(lifted_steps) if pose_available else "NA"
    premature_release = bool(active_rows and active_opens and max_open_streak(active_rows) >= 3)
    drop_after_lift = bool(dropped_steps) if pose_available else "NA"
    unstable_transport = bool(drop_after_lift is True or (object_lifted is True and not success and max_open_streak(rows) >= 3))

    cq = contact_quality_v2(
        success=success,
        pose_available=pose_available,
        object_lifted=object_lifted,
        premature_release=premature_release,
        drop_after_lift=drop_after_lift,
        unstable_transport=unstable_transport,
    )
    cq_failure = cq["contact_quality_failure"]
    first_failure = first_step(rows, lambda row: env_open(row) is True) if cq_failure is True else ""

    state_match = re.search(r"s(?:tate)?(\d+)", run_id)
    return {
        "run_id": run_id,
        "task_id": manifest.get("task_id", ""),
        "state": state_match.group(1) if state_match else "",
        "condition": infer_condition(run_id),
        "official_success": success,
        "object_lifted": object_lifted,
        "premature_release": premature_release,
        "drop_after_lift": drop_after_lift,
        "unstable_transport": unstable_transport,
        "contact_quality_failure": cq_failure,
        "contact_quality_success": cq["contact_quality_success"],
        "sr_cq_mismatch": cq["sr_cq_mismatch"],
        "failure_phase_auto": cq["failure_phase_auto"],
        "first_failure_step": first_failure,
        "exec_open_rate": "" if not opens else sum(opens) / len(opens),
        "executed_open_rate_on_active_steps": "" if not active_opens else sum(active_opens) / len(active_opens),
        "max_open_streak": max_open_streak(rows),
        "qpos_abs_after_max": "" if not qpos_values else max(qpos_values),
        "confidence": cq["cq_confidence_v2"],
        "failure_reason": cq["failure_reason"],
        "uncontrolled_final_drop": cq["uncontrolled_final_drop"],
        "stable_controlled_place": cq["stable_controlled_place"],
        "cq_failure_reason_v2": cq["cq_failure_reason_v2"],
        "cq_rule_version": cq["cq_rule_version"],
        "cq_confidence_v2": cq["cq_confidence_v2"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract contact-quality metrics from completed runs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    root = Path(args.input_root)
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "run_manifest.json").exists():
            continue
        rows.append(summarize_run(run_dir))
    write_csv(Path(args.output_csv), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
