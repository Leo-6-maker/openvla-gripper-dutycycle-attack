from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gripper_attack.stage_v_m3_5_phase_classifier import (  # noqa: E402
    PHASES,
    UNKNOWN,
    classify_trajectory,
)
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import (  # noqa: E402
    ProbePlanError,
    select_probe_steps,
)


def _row(step: int, *, z: float = 0.0, contact: bool = False, distance: float = 0.3, remaining: int = 40) -> dict:
    return {
        "step": step,
        "clean_record_valid": True,
        "clean_terminal": False,
        "remaining_horizon": remaining,
        "object_identity": "cube_1",
        "object_position": [0.0, 0.0, z],
        "eef_position": [0.0, 0.0, z + distance],
        "object_eef_distance_m": distance,
        "object_gripper_contact": contact,
        "object_support_contact": not contact,
        "contact_telemetry_valid": True,
        "raw_gripper": 0.0,
        "env_gripper": 1.0,
    }


def test_classifier_emits_registered_phase_sequence_and_fail_closed_unknown():
    rows = [_row(0), _row(1, contact=True, distance=0.08), _row(2, z=0.02, contact=True, distance=0.03)]
    rows.extend(_row(3 + i, z=0.02, contact=True, distance=0.03) for i in range(3))
    labels = classify_trajectory(rows)
    assert labels[0]["clean_only_phase_label"] == "PRE_CONTACT"
    assert labels[1]["clean_only_phase_label"] == "CONTACT_MANIPULATION"
    assert labels[2]["clean_only_phase_label"] == "ENGAGED_LIFT"
    assert labels[-1]["clean_only_phase_label"] == "CARRY"
    assert all(label["clean_only_phase_label"] in (*PHASES, UNKNOWN) for label in labels)
    assert classify_trajectory([_row(0, remaining=19)])[0]["clean_only_phase_label"] == UNKNOWN


def test_probe_plan_requires_six_clean_candidates_per_phase_and_is_deterministic():
    rows = []
    step = 0
    for phase in PHASES:
        for _ in range(8):
            rows.append({
                "step": step,
                "clean_record_valid": True,
                "phase_eligible": True,
                "clean_only_phase_label": phase,
                "remaining_horizon": 30,
            })
            step += 1
    first = select_probe_steps(rows, "libero_goal/task_00/state_00")
    second = select_probe_steps(rows, "libero_goal/task_00/state_00")
    assert first == second
    assert first["probe_count"] == 24
    assert {phase: sum(item["phase_label"] == phase for item in first["probe_steps"]) for phase in PHASES} == {phase: 6 for phase in PHASES}
    assert first["outcomes_read"] is False


def test_probe_plan_does_not_backfill_missing_phase():
    rows = []
    for step in range(24):
        rows.append({
            "step": step,
            "clean_record_valid": True,
            "phase_eligible": True,
            "clean_only_phase_label": PHASES[0] if step < 18 else PHASES[1],
            "remaining_horizon": 30,
        })
    try:
        select_probe_steps(rows, "libero_goal/task_00/state_00")
    except ProbePlanError as exc:
        assert "INSUFFICIENT_PHASE_COVERAGE" in str(exc)
    else:
        raise AssertionError("missing phase coverage must fail closed")
