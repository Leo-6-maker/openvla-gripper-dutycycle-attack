"""Clean-only, fail-closed phase classifier for the M3.5 probe plan.

The classifier consumes only per-step clean privileged telemetry.  It never
reads branch outcomes, labels, or task success, and emits ``UNKNOWN`` when the
telemetry cannot support one of the four registered strata.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "STAGE_V_M3_5_PHASE_CLASSIFIER_V1"
VERSION = "V1"
PHASES = ("PRE_CONTACT", "CONTACT_MANIPULATION", "ENGAGED_LIFT", "CARRY")
UNKNOWN = "UNKNOWN"
REGISTERED_LABELS = PHASES + (UNKNOWN,)
MIN_REMAINING_STEPS = 20  # T10 + frozen H_phys(10)
DEFAULT_THRESHOLDS = {
    "raw_open_threshold": 0.5,
    "env_open_value": -1.0,
    "env_close_value": 1.0,
    "action_semantics_tolerance": 1e-6,
    "lift_delta_m": 0.015,
    "object_eef_distance_m": 0.12,
    "carry_relative_distance_m": 0.04,
    "carry_stability_steps": 3,
}
REQUIRED_FIELDS = (
    "step",
    "clean_record_valid",
    "clean_terminal",
    "remaining_horizon",
    "object_identity",
    "object_position",
    "eef_position",
    "object_eef_distance_m",
    "object_gripper_contact",
    "object_support_contact",
    "contact_telemetry_valid",
    "raw_gripper",
    "env_gripper",
)


class PhaseClassifierError(ValueError):
    """Raised when clean telemetry violates the classifier input contract."""


def specification() -> dict[str, Any]:
    """Return the frozen, machine-readable classifier specification."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "registered_phase_labels": list(PHASES),
        "unknown_label": UNKNOWN,
        "required_fields": list(REQUIRED_FIELDS),
        "clean_only": True,
        "outcome_blind": True,
        "minimum_remaining_steps": MIN_REMAINING_STEPS,
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "history_rule": "only current row and clean rows at or before current step",
        "state_rule": "object identity and baseline are explicit; missing telemetry yields UNKNOWN",
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _vector(value: Any, width: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    result = tuple(float(item) for item in value)
    return result if all(math.isfinite(item) for item in result) else None


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _unknown(reason: str, *, step: Any = None, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "step": None if step is None else int(step),
        "clean_only_phase_label": UNKNOWN,
        "phase_eligible": False,
        "phase_confidence": 0.0,
        "abstain_reason": str(reason),
        "evidence": dict(evidence or {}),
    }


def _semantics_ok(raw: Any, env: Any, thresholds: Mapping[str, Any]) -> bool:
    if not _finite(raw) or not _finite(env):
        return False
    raw_value = float(raw)
    env_value = float(env)
    threshold = float(thresholds["raw_open_threshold"])
    tolerance = float(thresholds["action_semantics_tolerance"])
    if math.isclose(raw_value, threshold, abs_tol=tolerance):
        return False
    expected = float(thresholds["env_open_value"] if raw_value > threshold else thresholds["env_close_value"])
    return math.isclose(env_value, expected, abs_tol=tolerance)


def _stable_carry_count(
    rows: Sequence[Mapping[str, Any]], *, baseline_z: float, thresholds: Mapping[str, Any]
) -> int:
    required = int(thresholds["carry_stability_steps"])
    count = 0
    for row in reversed(rows):
        position = _vector(row.get("object_position"), 3)
        eef = _vector(row.get("eef_position"), 3)
        distance = row.get("object_eef_distance_m")
        if (
            row.get("clean_record_valid") is not True
            or row.get("object_gripper_contact") is not True
            or position is None
            or eef is None
            or not _finite(distance)
            or position[2] - baseline_z < float(thresholds["lift_delta_m"])
            or float(distance) > float(thresholds["carry_relative_distance_m"])
        ):
            break
        count += 1
        if count >= required:
            break
    return count


def classify_phase(
    row: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
    *,
    thresholds: Mapping[str, Any] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Classify one clean row from explicit clean telemetry.

    ``history`` must contain only rows at or before ``row['step']``.  The
    function does not infer missing object identity, contact, or geometry.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        return _unknown("MISSING_REQUIRED_TELEMETRY", step=row.get("step"), evidence={"missing": missing})
    step = row.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        return _unknown("INVALID_STEP", step=step)
    if row.get("clean_record_valid") is not True or row.get("clean_terminal") is True:
        return _unknown("CLEAN_RECORD_INVALID_OR_TERMINAL", step=step)
    try:
        remaining = int(row["remaining_horizon"])
    except (TypeError, ValueError):
        return _unknown("INVALID_REMAINING_HORIZON", step=step)
    if remaining < MIN_REMAINING_STEPS:
        return _unknown("INSUFFICIENT_REMAINING_HORIZON", step=step, evidence={"remaining_horizon": remaining})
    if not isinstance(row["object_identity"], str) or not row["object_identity"]:
        return _unknown("OBJECT_IDENTITY_UNBOUND", step=step)
    if row.get("contact_telemetry_valid") is not True:
        return _unknown("CONTACT_TELEMETRY_INVALID", step=step)
    object_position = _vector(row.get("object_position"), 3)
    eef_position = _vector(row.get("eef_position"), 3)
    distance = row.get("object_eef_distance_m")
    if object_position is None or eef_position is None or not _finite(distance):
        return _unknown("OBJECT_OR_EEF_GEOMETRY_INVALID", step=step)
    if not isinstance(row.get("object_gripper_contact"), bool) or not isinstance(row.get("object_support_contact"), bool):
        return _unknown("CONTACT_FLAGS_INVALID", step=step)
    if not _semantics_ok(row.get("raw_gripper"), row.get("env_gripper"), thresholds):
        return _unknown("ACTION_SEMANTICS_INVALID", step=step)
    baseline = row.get("object_z_baseline_m")
    if not _finite(baseline):
        return _unknown("OBJECT_BASELINE_MISSING", step=step)
    baseline_z = float(baseline)
    lift_delta = float(object_position[2] - baseline_z)
    contact = bool(row["object_gripper_contact"])
    stable_count = _stable_carry_count((*history, row), baseline_z=baseline_z, thresholds=thresholds)
    evidence = {
        "object_identity": row["object_identity"],
        "object_gripper_contact": contact,
        "object_support_contact": bool(row["object_support_contact"]),
        "object_eef_distance_m": float(distance),
        "object_z_baseline_m": baseline_z,
        "object_lift_delta_m": lift_delta,
        "stable_carry_count": stable_count,
        "remaining_horizon": remaining,
    }
    if lift_delta >= float(thresholds["lift_delta_m"]):
        if not contact:
            return _unknown("POST_CONTACT_LOSS", step=step, evidence=evidence)
        if stable_count >= int(thresholds["carry_stability_steps"]) and float(distance) <= float(thresholds["carry_relative_distance_m"]):
            label, confidence = "CARRY", 1.0
        else:
            label, confidence = "ENGAGED_LIFT", 0.8
    elif contact or float(distance) <= float(thresholds["object_eef_distance_m"]):
        label, confidence = "CONTACT_MANIPULATION", 0.9
    else:
        label, confidence = "PRE_CONTACT", 0.8
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "step": step,
        "clean_only_phase_label": label,
        "phase_eligible": True,
        "phase_confidence": confidence,
        "abstain_reason": "",
        "evidence": evidence,
    }


def classify_trajectory(
    rows: Sequence[Mapping[str, Any]], *, thresholds: Mapping[str, Any] = DEFAULT_THRESHOLDS
) -> list[dict[str, Any]]:
    """Classify a clean trajectory while deriving one explicit z baseline."""
    if not rows:
        raise PhaseClassifierError("EMPTY_CLEAN_TRAJECTORY")
    steps = [row.get("step") for row in rows]
    if any(not isinstance(step, int) or isinstance(step, bool) for step in steps) or steps != sorted(set(steps)):
        raise PhaseClassifierError("CLEAN_TRAJECTORY_STEPS_NOT_UNIQUE_SORTED")
    baseline: float | None = None
    for row in rows:
        position = _vector(row.get("object_position"), 3)
        if row.get("clean_record_valid") is True and position is not None:
            baseline = position[2]
            break
    result: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for raw in rows:
        current = dict(raw)
        if baseline is not None:
            current.setdefault("object_z_baseline_m", baseline)
        label = classify_phase(current, history, thresholds=thresholds)
        result.append(label)
        history.append(current)
    return result


if __name__ == "__main__":
    assert specification()["registered_phase_labels"] == list(PHASES)
    print(specification()["schema"])
