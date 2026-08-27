"""Stage AC0 provisional, model-agnostic clean eligibility evaluator.

This module is deliberately independent of the frozen AA2 scanner.  It is
used only on consumed calibration identities until AC0 is accepted.  A
candidate is judged by its local 20-row continuation; full episode horizon is
not a requirement.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple


STABLE_GRASP_WINDOW_STEPS = 3
CLEAN_CONTINUATION_STEPS = 20
ABSOLUTE_OBJECT_EEF_DISTANCE_MAX_M = 0.12
RELATIVE_CARRY_DISPLACEMENT_MAX_M = 0.04
MIN_LIFT_M = 0.015
DISTANCE_CONSISTENCY_TOLERANCE_M = 1e-6


def _finite_vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == size
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def _relative_offset(row: Dict[str, Any]) -> Tuple[float, float, float]:
    object_position = row["object_position"]
    eef_position = row["eef_position"]
    return tuple(float(object_position[i]) - float(eef_position[i]) for i in range(3))  # type: ignore[return-value]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in value))


def telemetry_valid(row: Dict[str, Any], distance_tolerance_m: float = DISTANCE_CONSISTENCY_TOLERANCE_M) -> bool:
    if row.get("contact_telemetry_valid") is not True:
        return False
    if not isinstance(row.get("object_identity"), str) or not row["object_identity"]:
        return False
    if not _finite_vector(row.get("object_position"), 3) or not _finite_vector(row.get("eef_position"), 3):
        return False
    if not isinstance(row.get("object_eef_distance_m"), (int, float)):
        return False
    distance = float(row["object_eef_distance_m"])
    if not math.isfinite(distance):
        return False
    if abs(distance - _norm(_relative_offset(row))) > distance_tolerance_m:
        return False
    return isinstance(row.get("object_gripper_contact"), bool) and isinstance(row.get("object_support_contact"), bool)


def _reason(result: Dict[str, Any], value: str) -> None:
    reasons = result["reason_codes"]
    if value not in reasons:
        reasons.append(value)


def evaluate_candidate(
    rows: Sequence[Dict[str, Any]],
    actions: Optional[Sequence[Dict[str, Any]]],
    step: int,
    baseline_z: Optional[float],
    anchor_class: str = "CRITICAL",
    max_contact_false_rows: int = 0,
) -> Dict[str, Any]:
    """Evaluate one local candidate without requiring a complete episode."""

    result: Dict[str, Any] = {
        "step": int(step),
        "anchor_class": anchor_class,
        "eligible": False,
        "reason_codes": [],
        "metrics": {},
    }
    if anchor_class not in {"CRITICAL", "NONCRITICAL"}:
        _reason(result, "ANCHOR_CLASS_INVALID")
        return result
    if step < 0 or step >= len(rows):
        _reason(result, "ANCHOR_STEP_INVALID")
        return result
    if actions is not None and (step >= len(actions) or not actions[step].get("boundary")):
        _reason(result, "ANCHOR_NOT_FRESH_BOUNDARY")
    continuation = list(rows[step : step + CLEAN_CONTINUATION_STEPS])
    if len(continuation) != CLEAN_CONTINUATION_STEPS:
        _reason(result, "LOCAL_CONTINUATION_INCOMPLETE")
    if baseline_z is None or not math.isfinite(float(baseline_z)):
        _reason(result, "OBJECT_BASELINE_INVALID")
    if any(row.get("terminal_before") is True for row in continuation):
        _reason(result, "TERMINAL_BEFORE_CONTINUATION_END")
    if any(row.get("terminal_after") is True for row in continuation[:-1]):
        _reason(result, "TERMINAL_BEFORE_CONTINUATION_END")
    valid_rows = [row for row in continuation if telemetry_valid(row)]
    if len(valid_rows) != len(continuation):
        _reason(result, "TELEMETRY_INVALID_IN_LOCAL_WINDOW")
    identities = {row.get("object_identity") for row in valid_rows}
    if len(identities) != 1:
        _reason(result, "OBJECT_IDENTITY_NOT_CONSTANT")

    if anchor_class == "NONCRITICAL":
        current = rows[step]
        if current.get("object_gripper_contact") is not False or current.get("object_support_contact") is not False:
            _reason(result, "NOT_PRE_CONTACT")
        result["eligible"] = not result["reason_codes"]
        return result

    if len(continuation) < CLEAN_CONTINUATION_STEPS or not valid_rows or baseline_z is None:
        return result
    window = continuation[:STABLE_GRASP_WINDOW_STEPS]
    if len(window) != STABLE_GRASP_WINDOW_STEPS or any(row.get("object_gripper_contact") is not True for row in window):
        _reason(result, "STABLE_GRASP_WINDOW_INVALID")
    if any(row.get("object_support_contact") is not False for row in continuation):
        _reason(result, "SUPPORT_CONTACT_PRESENT")

    anchor_relative = _relative_offset(continuation[0])
    relative_drifts = [_norm(tuple(a - b for a, b in zip(_relative_offset(row), anchor_relative))) for row in continuation]
    distances = [float(row["object_eef_distance_m"]) for row in continuation]
    lifts = [float(row["object_position"][2]) - float(baseline_z) for row in continuation]
    false_contact_rows = sum(row.get("object_gripper_contact") is not True for row in continuation)
    result["metrics"] = {
        "relative_drift_max_m": max(relative_drifts),
        "absolute_object_eef_distance_max_m": max(distances),
        "minimum_lift_m": min(lifts),
        "contact_false_rows": false_contact_rows,
    }
    if max(distances) > ABSOLUTE_OBJECT_EEF_DISTANCE_MAX_M:
        _reason(result, "ABSOLUTE_OBJECT_EEF_DISTANCE_EXCEEDED")
    if max(relative_drifts) > RELATIVE_CARRY_DISPLACEMENT_MAX_M:
        _reason(result, "RELATIVE_CARRY_DISPLACEMENT_EXCEEDED")
    if min(lifts) < MIN_LIFT_M:
        _reason(result, "LIFT_THRESHOLD_NOT_MET")
    if false_contact_rows > int(max_contact_false_rows):
        _reason(result, "CONTACT_FLICKER_LIMIT_EXCEEDED")
    result["eligible"] = not result["reason_codes"]
    return result


def rank_candidate(salt: str, model_family: str, parent_key: str, step: int) -> str:
    return hashlib.sha256(f"{salt}|{model_family}|{parent_key}|{step}".encode("utf-8")).hexdigest()


def scan_candidates(
    rows: Sequence[Dict[str, Any]],
    actions: Sequence[Dict[str, Any]],
    model_family: str,
    parent_key: str,
    baseline_z: Optional[float],
    salt: str,
    anchor_class: str = "CRITICAL",
    max_contact_false_rows: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    for step in range(max(0, len(rows) - CLEAN_CONTINUATION_STEPS + 1)):
        result = evaluate_candidate(rows, actions, step, baseline_z, anchor_class, max_contact_false_rows)
        for value in set(result["reason_codes"]):
            reasons[value] += 1
        if result["eligible"]:
            candidates.append(
                {
                    "step": step,
                    "anchor_class": anchor_class,
                    "selection_rank_sha256": rank_candidate(salt, model_family, parent_key, step),
                    "metrics": result["metrics"],
                }
            )
    candidates.sort(key=lambda item: (item["selection_rank_sha256"], item["step"]))
    return candidates, dict(sorted(reasons.items()))


def classify_calibration_control(rows: Sequence[Dict[str, Any]], step: int, baseline_z: Optional[float]) -> str:
    """Classify only controls with directly observable, privileged semantics."""

    if step < 0 or step >= len(rows) or not telemetry_valid(rows[step]):
        return "NOT_IDENTIFIABLE"
    row = rows[step]
    if row.get("object_support_contact") is True:
        return "SUPPORT_CONTACT"
    if row.get("object_gripper_contact") is False:
        prior_stable = any(
            evaluate_candidate(rows, None, previous, baseline_z, "CRITICAL")["eligible"]
            for previous in range(max(0, step - CLEAN_CONTINUATION_STEPS + 1), step)
        )
        return "INTENDED_RELEASE_OR_DROP" if prior_stable else "PRE_CONTACT"
    return "UNKNOWN_CONTROL"


__all__ = [
    "ABSOLUTE_OBJECT_EEF_DISTANCE_MAX_M",
    "CLEAN_CONTINUATION_STEPS",
    "DISTANCE_CONSISTENCY_TOLERANCE_M",
    "MIN_LIFT_M",
    "RELATIVE_CARRY_DISPLACEMENT_MAX_M",
    "STABLE_GRASP_WINDOW_STEPS",
    "classify_calibration_control",
    "evaluate_candidate",
    "rank_candidate",
    "scan_candidates",
    "telemetry_valid",
]
