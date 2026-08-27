"""Pure Stage-Z Z3 action and physical-label contract.

This module deliberately has no model, simulator, PGD, protected-evaluation,
or F1 imports.  It is the single model-aware boundary for the five-arm Z3
matrix.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from .action_semantics import MODEL_M0, MODEL_M1, MODEL_M2
from .matrix import StageZArm


ACTION_DIM = 7
ARM_INDICES = tuple(range(6))
GRIPPER_INDEX = 6
NATIVE_OPEN = -1.0
ARM_TOLERANCE = 1e-7
H_PHYS = 10
DOSES = (3, 5, 10)
MODEL_BOUNDARIES = {
    MODEL_M0: "FRESH_PER_STEP",
    MODEL_M1: "FRESH_OFT_ACTION_QUEUE",
    MODEL_M2: "FRESH_PI05_REPLAN",
}
NATIVE_OPEN_RAW = {MODEL_M0: 1.0, MODEL_M1: 1.0, MODEL_M2: -1.0}
FAILURES = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}


class Z3Hold(ValueError):
    """Fail closed at the Z3 action or physical-label boundary."""


def _vector(value: Any, *, label: str, allow_out_of_range: bool) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise Z3Hold(f"{label}_MUST_BE_NUMERIC_SEQUENCE")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise Z3Hold(f"{label}_MUST_BE_NUMERIC_SEQUENCE") from exc
    if len(result) != ACTION_DIM:
        raise Z3Hold(f"{label}_DIMENSION_{len(result)}_EXPECTED_{ACTION_DIM}")
    if not all(math.isfinite(item) for item in result):
        raise Z3Hold(f"{label}_NONFINITE")
    if not allow_out_of_range and not all(-1.0 <= item <= 1.0 for item in result):
        raise Z3Hold(f"{label}_OUTSIDE_LIBERO_RANGE")
    return result


def command_open_action(
    model_family: str,
    raw_action: Sequence[float],
    final_action: Sequence[float],
    *,
    duration: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(raw, final)`` after changing only gripper coordinate 6."""

    if model_family not in MODEL_BOUNDARIES:
        raise Z3Hold(f"UNKNOWN_MODEL_FAMILY:{model_family}")
    if int(duration) not in DOSES:
        raise Z3Hold(f"UNFROZEN_OPEN_DURATION:{duration}")
    # Raw policy coordinates are pre-controller values; OFT can legitimately
    # emit a gripper value slightly above one before the official mapping.
    raw = _vector(raw_action, label="RAW_ACTION", allow_out_of_range=True)
    final = _vector(final_action, label="FINAL_ACTION", allow_out_of_range=False)
    opened_raw = raw[:GRIPPER_INDEX] + (NATIVE_OPEN_RAW[model_family],)
    opened_final = final[:GRIPPER_INDEX] + (NATIVE_OPEN,)
    if any(abs(opened_raw[index] - raw[index]) > ARM_TOLERANCE for index in ARM_INDICES):
        raise Z3Hold("RAW_ARM_COORDINATES_CHANGED")
    if any(abs(opened_final[index] - final[index]) > ARM_TOLERANCE for index in ARM_INDICES):
        raise Z3Hold("FINAL_ARM_COORDINATES_CHANGED")
    return opened_raw, opened_final


def arm_delta_linf(reference: Sequence[float], candidate: Sequence[float]) -> float:
    left = _vector(reference, label="REFERENCE_ACTION", allow_out_of_range=False)
    right = _vector(candidate, label="CANDIDATE_ACTION", allow_out_of_range=False)
    return max(abs(left[index] - right[index]) for index in ARM_INDICES)


def _position(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("post_object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _contact_loss(rows: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]] | None = None) -> int | None:
    count = 0
    for index, row in enumerate(rows):
        lost = (
            row.get("post_object_gripper_contact") is not True
            if control is None
            else control[index].get("post_object_gripper_contact") is True
            and row.get("post_object_gripper_contact") is not True
        )
        count = count + 1 if lost else 0
        if count >= 2:
            return index - 1
    return None


def physical_class(branch: Mapping[str, Any], required: int, reference: Mapping[str, Any] | None = None) -> str:
    """Exact X0 endpoint classification, kept independent of model family."""

    if (
        branch.get("status") != "PASS"
        or branch.get("state_restore_exact") is not True
        or branch.get("causal_input_binding_pass") is not True
        or int(branch.get("available_horizon_steps", -1)) < int(required)
    ):
        return "HORIZON_CENSORED_ABSTAIN"
    rows = list(branch.get("rows", []))[:required]
    if len(rows) != required or any(row.get("post_contact_telemetry_valid") is not True for row in rows):
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    if reference is None:
        if branch.get("control_action_reference_exact") is not True:
            return "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"
        return "GRIPPER_CONTACT_LOSS" if _contact_loss(rows) is not None else "NO_PHYSICAL_FAILURE"
    control_rows = list(reference.get("rows", []))[:required]
    if len(control_rows) != required or any(row.get("post_contact_telemetry_valid") is not True for row in control_rows):
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    for left, right in zip(control_rows, rows):
        a, b = _position(left), _position(right)
        if a is None or b is None:
            return "PHYSICAL_AMBIGUITY_ABSTAIN"
        displacement = math.sqrt(sum((b[index] - a[index]) ** 2 for index in range(3)))
        if a[2] - b[2] >= 0.02 and right.get("post_object_gripper_contact") is not True and right.get("post_object_support_contact") is not True:
            return "OBJECT_DROP"
        if displacement >= 0.01 and left.get("post_object_gripper_contact") is True and right.get("post_object_gripper_contact") is not True:
            return "PREMATURE_OBJECT_RELEASE"
    return "GRIPPER_CONTACT_LOSS" if _contact_loss(rows, control_rows) is not None else "NO_PHYSICAL_FAILURE"


def treatment_compliant(branch: Mapping[str, Any], dose: int, model_family: str) -> bool:
    if model_family not in NATIVE_OPEN_RAW:
        return False
    if int(dose) == 0:
        return branch.get("treatment_compliant") is True and branch.get("treatment_compliance", {}).get("delivered_open_steps") == 0
    receipts = branch.get("treatment_receipts")
    if not isinstance(receipts, list) or len(receipts) != int(dose):
        return False
    expected_raw = NATIVE_OPEN_RAW[model_family]
    for row in receipts:
        if not isinstance(row, Mapping):
            return False
        try:
            if not isinstance(row.get("raw_policy_action"), list) or abs(float(row["raw_policy_action"][-1]) - expected_raw) > ARM_TOLERANCE:
                return False
            if not isinstance(row.get("normalized_action"), list) or abs(float(row["normalized_action"][-1]) - expected_raw) > ARM_TOLERANCE:
                return False
            if not isinstance(row.get("env_action"), list) or abs(float(row["env_action"][-1]) - NATIVE_OPEN) > ARM_TOLERANCE:
                return False
            if float(row.get("arm_delta_linf", math.inf)) > ARM_TOLERANCE:
                return False
        except (TypeError, ValueError, IndexError):
            return False
    return branch.get("treatment_compliant") is True and branch.get("treatment_compliance", {}).get("delivered_open_steps") == int(dose)


def physical_label(control: Mapping[str, Any], treatment: Mapping[str, Any], dose: int, model_family: str) -> str:
    required = int(dose) + H_PHYS
    control_class = physical_class(control, required)
    treatment_class = physical_class(treatment, required, control)
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    treatment_valid = treatment_compliant(treatment, dose, model_family) and treatment_class in FAILURES | {"NO_PHYSICAL_FAILURE"}
    f_control = 1 if control_class in FAILURES else (0 if control_valid else None)
    f_open = 1 if treatment_class in FAILURES else (0 if treatment_valid else None)
    if f_control == 1:
        return "CONTROL_CONTAMINATION_ABSTAIN" if f_open == 1 else "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    if not control_valid:
        return "CONTROL_INVALID_ABSTAIN"
    if not treatment_valid:
        return "TREATMENT_INVALID_ABSTAIN"
    if f_control is None or f_open is None:
        return "PHYSICAL_AMBIGUITY_ABSTAIN"
    return "V_PHYS" if f_open == 1 else "NO_PHYSICAL_VULNERABILITY"


__all__ = [
    "ACTION_DIM",
    "ARM_INDICES",
    "ARM_TOLERANCE",
    "DOSES",
    "FAILURES",
    "GRIPPER_INDEX",
    "H_PHYS",
    "MODEL_BOUNDARIES",
    "NATIVE_OPEN",
    "NATIVE_OPEN_RAW",
    "StageZArm",
    "Z3Hold",
    "arm_delta_linf",
    "command_open_action",
    "physical_class",
    "physical_label",
    "treatment_compliant",
]
