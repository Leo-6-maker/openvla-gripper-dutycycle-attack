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
    PREFIX_SELECTION_VERSION,
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


def _corridor_row(step: int, phase: str = "CONTACT_MANIPULATION", *, remaining: int = 30) -> dict:
    return {
        "step": step,
        "clean_record_valid": True,
        "clean_terminal": False,
        "phase_eligible": True,
        "clean_only_phase_label": phase,
        "remaining_horizon": remaining,
        "contact_telemetry_valid": True,
        "object_identity": "cube_1",
        "object_position": [0.0, 0.0, 0.1],
        "eef_position": [0.0, 0.0, 0.11],
        "object_eef_distance_m": 0.01,
        "object_gripper_contact": True,
        "object_support_contact": False,
    }


def test_probe_plan_uses_deterministic_corridor_quantiles_without_phase_quota():
    rows = [_corridor_row(step, PHASES[step % len(PHASES)]) for step in range(48)]
    first = select_probe_steps(rows, "libero_goal/task_00/state_00")
    second = select_probe_steps(rows, "libero_goal/task_00/state_00")
    assert first == second
    assert first["probe_count"] == 24
    assert [item["quantile_ordinal"] for item in first["probe_steps"]] == list(range(24))
    assert len({item["step"] for item in first["probe_steps"]}) == 24
    assert first["selected_phase_distribution_descriptive_only"] == {
        phase: sum(item["phase_label"] == phase for item in first["probe_steps"])
        for phase in PHASES
    }
    assert first["outcomes_read"] is False


def test_probe_plan_prefix_selection_is_outcome_blind_and_early():
    rows = [_corridor_row(step, PHASES[step % len(PHASES)]) for step in range(48)]
    plan = select_probe_steps(rows, "libero_goal/task_00/state_00", selection_version=PREFIX_SELECTION_VERSION)
    assert [item["step"] for item in plan["probe_steps"]] == list(range(24))
    assert plan["selection_algorithm"].startswith("sort eligible corridor by timestep; choose the first 24")
    assert plan["outcomes_read"] is False


def test_probe_plan_fails_deterministically_below_24_corridor_states():
    rows = [_corridor_row(step) for step in range(42)]
    for row in rows[23:]:
        row["object_gripper_contact"] = False
    try:
        select_probe_steps(rows, "libero_goal/task_00/state_00")
    except ProbePlanError as exc:
        assert str(exc) == "PROBE_PLAN_INSUFFICIENT_CORRIDOR:23/24"
    else:
        raise AssertionError("short corridor must fail closed")


def test_probe_plan_rejects_gapped_step_horizon():
    rows = [_corridor_row(step) for step in range(44)]
    rows[20]["step"] = 100
    try:
        select_probe_steps(rows, "libero_goal/task_00/state_00")
    except ProbePlanError as exc:
        assert str(exc) == "CLEAN_TRAJECTORY_STEPS_NOT_CONTIGUOUS_FROM_ZERO"
    else:
        raise AssertionError("row count must not substitute for an environment-step horizon")
