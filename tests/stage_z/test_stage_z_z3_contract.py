from __future__ import annotations

import pytest

from stage_z_preparation.z3_contract import (
    MODEL_BOUNDARIES,
    MODEL_M0,
    MODEL_M1,
    MODEL_M2,
    NATIVE_OPEN_RAW,
    StageZArm,
    Z3Hold,
    arm_delta_linf,
    command_open_action,
    physical_label,
    treatment_compliant,
)


ACTION = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7)


def test_five_arms_and_model_boundaries_are_frozen() -> None:
    assert list(StageZArm) == [
        StageZArm.CLEAN_BRANCH_CRITICAL,
        StageZArm.COMMAND_OPEN_T3_CRITICAL,
        StageZArm.COMMAND_OPEN_T5_CRITICAL,
        StageZArm.COMMAND_OPEN_T10_CRITICAL,
        StageZArm.COMMAND_OPEN_T5_NONCRITICAL_CONTROL,
    ]
    assert MODEL_BOUNDARIES == {
        MODEL_M0: "FRESH_PER_STEP",
        MODEL_M1: "FRESH_OFT_ACTION_QUEUE",
        MODEL_M2: "FRESH_PI05_REPLAN",
    }


@pytest.mark.parametrize("model", (MODEL_M0, MODEL_M1, MODEL_M2))
def test_command_open_changes_only_gripper_and_uses_native_raw(model: str) -> None:
    raw = ACTION if model != MODEL_M2 else (2.0, -2.0, *ACTION[2:])
    opened_raw, opened_final = command_open_action(model, raw, ACTION, duration=5)
    assert opened_raw[:6] == tuple(raw[:6])
    assert opened_final[:6] == ACTION[:6]
    assert opened_raw[6] == NATIVE_OPEN_RAW[model]
    assert opened_final[6] == -1.0
    assert arm_delta_linf(ACTION, opened_final) == 0.0


def test_command_open_rejects_unfrozen_dose() -> None:
    with pytest.raises(Z3Hold, match="UNFROZEN"):
        command_open_action(MODEL_M0, ACTION, ACTION, duration=4)


def test_raw_policy_range_is_not_confused_with_final_libero_range() -> None:
    raw = (*ACTION[:6], 1.015625)
    opened_raw, opened_final = command_open_action(MODEL_M1, raw, ACTION, duration=3)
    assert opened_raw[6] == 1.0
    assert opened_final[6] == -1.0


def _rows(*, lost: bool = False) -> list[dict[str, object]]:
    return [
        {
            "post_contact_telemetry_valid": True,
            "post_object_position": [0.0, 0.0, 0.0],
            "post_object_gripper_contact": not lost,
            "post_object_support_contact": True,
        }
        for _ in range(15)
    ]


def _branch(*, lost: bool = False, treatment: bool = False) -> dict[str, object]:
    dose = 5
    receipts = [
        {
            "raw_policy_action": [0.0] * 6 + [1.0],
            "normalized_action": [0.0] * 6 + [1.0],
            "env_action": [0.0] * 6 + [-1.0],
            "arm_delta_linf": 0.0,
        }
        for _ in range(dose)
    ]
    return {
        "status": "PASS",
        "state_restore_exact": True,
        "causal_input_binding_pass": True,
        "available_horizon_steps": 15,
        "control_action_reference_exact": True,
        "rows": _rows(lost=lost),
        "treatment_receipts": receipts,
        "treatment_compliant": treatment,
        "treatment_compliance": {"delivered_open_steps": dose if treatment else 0},
    }


def test_x0_label_truth_table_and_native_m2_compliance() -> None:
    control = _branch()
    treatment = _branch(treatment=True)
    assert physical_label(control, treatment, 5, MODEL_M0) == "NO_PHYSICAL_VULNERABILITY"
    treatment["rows"] = _rows(lost=True)
    assert physical_label(control, treatment, 5, MODEL_M0) == "V_PHYS"
    m2 = _branch(treatment=True)
    for receipt in m2["treatment_receipts"]:
        receipt["raw_policy_action"][-1] = -1.0
        receipt["normalized_action"][-1] = -1.0
    assert treatment_compliant(m2, 5, MODEL_M2) is True


def test_noncompliant_branch_is_fail_closed() -> None:
    assert treatment_compliant(_branch(treatment=False), 5, MODEL_M0) is False
