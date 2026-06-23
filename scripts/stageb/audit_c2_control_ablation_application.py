#!/usr/bin/env python3
"""Post-hoc audit for C2 control-state ablation application.

This script reads existing C2 artifacts only. It does not load OpenVLA, LIBERO,
or any GPU resource. Its purpose is to distinguish a requested ablation from a
state group that was actually present, written, and matched after apply.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ABLATION_REQUESTS: dict[str, list[str]] = {
    "A0_BASELINE": [],
    "A1_DERIVED_RECOMPUTE": ["derived_refresh"],
    "A2_GOAL_STATE": ["goal"],
    "A3_GOAL_INTERPOLATOR_STATE": ["goal", "interpolator"],
    "A4_GOAL_INTERPOLATOR_ACTION_HISTORY": ["goal", "interpolator", "action_history"],
    "A5_QACC_ABLATION": ["goal", "interpolator", "action_history", "qacc"],
}

GROUP_ACTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "derived_refresh": ("robot0.controller.update(force=True)",),
    "goal": ("robot0.controller.goal_",),
    "interpolator": ("robot0.controller.interpolator_",),
    "action_history": ("robot0.recent_actions.", "env_inner.timestep", "env_inner.cur_time", "env_inner._"),
    "qacc": ("mujoco.qacc",),
}

GROUP_DIFF_PREFIXES: dict[str, tuple[str, ...]] = {
    "goal": ("robots[0].controller.attrs.goal_",),
    "interpolator": ("robots[0].controller_selected_attrs.interpolator_",),
    "action_history": (
        "robots[0].attrs.recent_actions.",
        "env_inner.attrs.timestep",
        "env_inner.attrs.cur_time",
        "env_inner.attrs._",
    ),
    "qacc": ("mujoco.qacc",),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ablation",
        "requested_group",
        "reference_field_present",
        "applied_action_count",
        "actual_applied_actions",
        "target_state_match_after_apply",
        "was_effective_intervention",
        "status",
        "evidence_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def reference_group_present(reference_state: Mapping[str, Any], group: str) -> bool:
    robots = list(reference_state.get("robots") or [])
    robot0 = robots[0] if robots else {}
    if group == "derived_refresh":
        return True
    if group == "goal":
        goal = robot0.get("controller_goal") or {}
        return any(goal.get(name) is not None for name in ("goal_pos", "goal_ori", "goal_orientation", "goal_qpos"))
    if group == "interpolator":
        return any(robot0.get(name) is not None for name in ("interpolator_pos", "interpolator_ori"))
    if group == "action_history":
        history = robot0.get("action_history") or {}
        counters = reference_state.get("env_counters") or {}
        return bool(history) or bool(counters)
    if group == "qacc":
        return reference_state.get("qacc") is not None
    return False


def matching_actions(actions: list[str], group: str) -> list[str]:
    prefixes = GROUP_ACTION_PREFIXES.get(group, ())
    return [action for action in actions if any(action.startswith(prefix) for prefix in prefixes)]


def group_diff_fields(pre_diff_rows: list[Mapping[str, str]], group: str) -> list[str]:
    prefixes = GROUP_DIFF_PREFIXES.get(group, ())
    if not prefixes:
        return []
    fields = []
    for row in pre_diff_rows:
        field = str(row.get("field", ""))
        if any(field.startswith(prefix) for prefix in prefixes):
            fields.append(field)
    return fields


def audit_ablation_group(
    *,
    ablation: str,
    group: str,
    reference_state: Mapping[str, Any],
    applied_actions: list[str],
    pre_diff_rows: list[Mapping[str, str]],
) -> dict[str, Any]:
    present = reference_group_present(reference_state, group)
    actions = matching_actions(applied_actions, group)
    diff_fields = group_diff_fields(pre_diff_rows, group)
    target_match = present and len(diff_fields) == 0
    if not present:
        status = f"NO_APPLICABLE_{group.upper()}_STATE"
        effective = False
        note = "requested group absent from reference snapshot"
    elif not actions and group != "derived_refresh":
        status = "REQUESTED_BUT_NO_ACTION_APPLIED"
        effective = False
        note = "reference state existed but no matching applied action was recorded"
    elif target_match:
        status = "APPLIED_AND_MATCHED"
        effective = True
        note = "matching applied action recorded and no post-apply pre-step diff for this group"
    else:
        status = "APPLIED_BUT_STILL_DIFFERS"
        effective = False
        note = "matching applied action recorded but pre-step diff still contains this group"
    return {
        "ablation": ablation,
        "requested_group": group,
        "reference_field_present": bool(present),
        "applied_action_count": len(actions),
        "actual_applied_actions": ";".join(actions),
        "target_state_match_after_apply": bool(target_match),
        "was_effective_intervention": bool(effective),
        "status": status,
        "evidence_note": note,
    }


def build_rows(c2_dir: Path) -> list[dict[str, Any]]:
    reference_state = read_json(c2_dir / "reference_mutable_control_state.json")
    rows: list[dict[str, Any]] = []
    for ablation, groups in ABLATION_REQUESTS.items():
        ablation_dir = c2_dir / ablation
        if not groups:
            continue
        applied = read_json(ablation_dir / "applied_ablation.json")
        actions = [str(action) for action in applied.get("actions", [])]
        pre_diff = read_csv_rows(ablation_dir / "pre_diff.csv")
        for group in groups:
            rows.append(
                audit_ablation_group(
                    ablation=ablation,
                    group=group,
                    reference_state=reference_state,
                    applied_actions=actions,
                    pre_diff_rows=pre_diff,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    c2_dir = args.c2_dir
    required = [
        c2_dir / "reference_mutable_control_state.json",
        c2_dir / "c2_control_state_ablation_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required C2 artifacts: " + ", ".join(missing))
    rows = build_rows(c2_dir)
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
