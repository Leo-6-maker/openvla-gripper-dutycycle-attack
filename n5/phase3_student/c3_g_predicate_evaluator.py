"""Pure geometry-only tri-state In/On/Stack evaluator for C3-G-DEV."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "configs" / "C3_G_PREDICATE_CONTRACT_V1_1.json"
SUPPORTED = frozenset({"In", "On", "Stack"})
OBJECT_ROLE = "MANIPULATED_OBJECT"
OBJECT_TARGET = "OBJECT_TARGET"
REGION_TARGET = "REGION_TARGET"
FORBIDDEN = frozenset({"task_success", "reward", "teacher", "outcome", "attack", "future"})


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "C3_G_PREDICATE_CONTRACT_V1_1" or data.get("status") != "FROZEN":
        raise ValueError("C3-G predicate contract is not frozen")
    if data.get("tri_state") != ["TRUE", "FALSE", "UNKNOWN"]:
        raise ValueError("C3-G tri-state contract mismatch")
    tolerance = data.get("tolerance", {})
    required = ("numerical_epsilon_m", "comparison_epsilon_m", "containment_margin_m", "support_vertical_tolerance_m", "horizontal_overlap_threshold_m")
    if any(key not in tolerance for key in required):
        raise ValueError("C3-G tolerance split is incomplete")
    if any(float(tolerance[key]) < 0 for key in required) or float(tolerance["numerical_epsilon_m"]) <= 0:
        raise ValueError("C3-G tolerance is invalid")
    return data


def _finite_vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _quat(value: Any) -> tuple[float, float, float, float] | None:
    vector = _finite_vector(value, 4)
    if vector is None:
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _quat_inverse(value: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = value
    return (w, -x, -y, -z)


def _quat_mul(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    raw = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
           w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
           w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
           w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    norm = math.sqrt(sum(item * item for item in raw))
    return tuple(item / norm for item in raw)  # type: ignore[return-value]


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def _relative_position(target_pose: Mapping[str, Any], object_pose: Mapping[str, Any]) -> tuple[list[float], tuple[float, float, float, float]] | None:
    target_pos = _finite_vector(target_pose.get("pos"), 3)
    object_pos = _finite_vector(object_pose.get("pos"), 3)
    target_quat = _quat(target_pose.get("quat"))
    object_quat = _quat(object_pose.get("quat"))
    if target_pos is None or object_pos is None or target_quat is None or object_quat is None:
        return None
    delta = [object_pos[index] - target_pos[index] for index in range(3)]
    return list(_rotate(_quat_inverse(target_quat), delta)), _quat_mul(_quat_inverse(target_quat), object_quat)


def _unknown(predicate: Any, reason: str) -> dict[str, Any]:
    return {"value": "UNKNOWN", "predicate": predicate, "reason": reason, "raw_measurements": {}}


def evaluate_case(case: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frozen = contract or load_contract()
    if any(key in case for key in FORBIDDEN):
        return _unknown(case.get("predicate"), "FORBIDDEN_INPUT")
    predicate = case.get("predicate")
    if predicate not in SUPPORTED or predicate not in frozen.get("predicates", {}):
        return _unknown(predicate, "INVALID_PREDICATE")
    object_row = case.get("object")
    target_row = case.get("target")
    if not isinstance(object_row, Mapping) or not isinstance(target_row, Mapping):
        return _unknown(predicate, "MISSING_ENTITY")
    expected = case.get("expected_identity")
    if isinstance(expected, Mapping):
        for field in ("episode_id", "step", "object_id", "target_id"):
            actual = {"episode_id": case.get("episode_id"), "step": case.get("step"), "object_id": object_row.get("id"), "target_id": target_row.get("id")}[field]
            if field in expected and actual != expected[field]:
                return _unknown(predicate, "IDENTITY_MISMATCH")
    object_id = object_row.get("id")
    target_id = target_row.get("id")
    if not isinstance(object_id, str) or not object_id or not isinstance(target_id, str) or not target_id or object_id == target_id:
        return _unknown(predicate, "IDENTITY_MISMATCH")
    if object_row.get("role") != OBJECT_ROLE:
        return _unknown(predicate, "OBJECT_ROLE_MISMATCH")
    target_role = target_row.get("role")
    allowed_roles = frozen["predicates"][predicate]["target_roles"]
    if target_role not in allowed_roles:
        return _unknown(predicate, "TARGET_ROLE_MISMATCH")
    target_extents = _finite_vector(target_row.get("half_extents"), 3)
    object_extents = _finite_vector(object_row.get("half_extents"), 3)
    if target_extents is None or object_extents is None or any(item <= 0 for item in target_extents + object_extents):
        return _unknown(predicate, "MISSING_OR_DEGENERATE_GEOMETRY")
    relative = _relative_position(target_row.get("pose", {}), object_row.get("pose", {}))
    if relative is None:
        return _unknown(predicate, "NON_FINITE_POSE")
    position, _ = relative
    tolerance = frozen["tolerance"]
    epsilon = float(tolerance.get("comparison_epsilon_m", tolerance["numerical_epsilon_m"]))
    if predicate == "In":
        limits = [target_extents[index] - object_extents[index] for index in range(3)]
        if any(limit < 0 for limit in limits):
            return _unknown(predicate, "DEGENERATE_CONTAINMENT")
        margin = float(tolerance["containment_margin_m"])
        value = "TRUE" if all(abs(position[index]) <= limits[index] + margin + epsilon for index in range(3)) else "FALSE"
        raw = {"relative_position": position, "limits": limits, "containment_margin_m": margin, "comparison_epsilon_m": epsilon}
    else:
        limits = [target_extents[0] - object_extents[0], target_extents[1] - object_extents[1]]
        if any(limit < 0 for limit in limits):
            return _unknown(predicate, "DEGENERATE_SUPPORT")
        horizontal_margin = float(tolerance["horizontal_overlap_threshold_m"])
        horizontal = all(abs(position[index]) <= limits[index] + horizontal_margin + epsilon for index in range(2))
        vertical_gap = position[2] - target_extents[2] - object_extents[2]
        vertical_tolerance = float(tolerance["support_vertical_tolerance_m"])
        value = "TRUE" if horizontal and abs(vertical_gap) <= vertical_tolerance + epsilon else "FALSE"
        raw = {"relative_position": position, "limits": limits, "horizontal_overlap_threshold_m": horizontal_margin, "vertical_gap_m": vertical_gap, "support_vertical_tolerance_m": vertical_tolerance, "comparison_epsilon_m": epsilon}
    return {"value": value, "predicate": predicate, "reason": "GEOMETRY_EVALUATED", "raw_measurements": raw, "relative_position": position}
