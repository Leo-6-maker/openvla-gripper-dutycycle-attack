"""Contact-complete, causal V23 Teacher for the R3 development canary.

This module intentionally has no fallback to the historical Fresh40 proxy.
It consumes only the canonical FIT670 V2 contact/geometry row contract and
returns explicit TRUE/FALSE/UNKNOWN labels.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


HEADS = (
    "physical_criticality",
    "k10_feasibility",
    "safe_release",
    "instability",
    "gripper_closing_state",
)
TRUTH_VALUES = ("TRUE", "FALSE", "UNKNOWN")
ENTITY_ROLES = {
    "MANIPULATED_OBJECT",
    "OBJECT_TARGET",
    "REGION_TARGET",
    "SUPPORT_SURFACE",
    "GRIPPER",
    "EEF",
}
FORBIDDEN_FIELDS = {
    "task_success", "terminal", "terminal_state", "reward", "outcome",
    "attack_result", "future", "future_frame", "future_label",
}


class R3ContractError(ValueError):
    """Raised when a canary row cannot be consumed without guessing."""


def _walk_forbidden(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_FIELDS:
                found.append(child)
            found.extend(_walk_forbidden(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_forbidden(item, f"{path}[{index}]"))
    return found


def _finite_vector(value: Any, size: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _label(value: str, reason: str) -> dict[str, Any]:
    if value not in TRUTH_VALUES or not reason:
        raise R3ContractError(f"invalid label: {value!r} {reason!r}")
    return {"value": value, "mask": value != "UNKNOWN", "reason": reason}


def tri_and(values: Iterable[str], *, unknown_reason: str = "COMPONENT_UNKNOWN") -> str:
    vals = list(values)
    if not vals or any(value not in TRUTH_VALUES for value in vals):
        return "UNKNOWN"
    if any(value == "FALSE" for value in vals):
        return "FALSE"
    return "UNKNOWN" if any(value == "UNKNOWN" for value in vals) else "TRUE"


def tri_or(values: Iterable[str]) -> str:
    vals = list(values)
    if not vals or any(value not in TRUTH_VALUES for value in vals):
        return "UNKNOWN"
    if any(value == "TRUE" for value in vals):
        return "TRUE"
    return "UNKNOWN" if any(value == "UNKNOWN" for value in vals) else "FALSE"


def quaternion_geodesic(q1: Sequence[float], q2: Sequence[float]) -> float | None:
    """Return sign-invariant shortest rotation angle in radians."""
    a = _finite_vector(q1, 4)
    b = _finite_vector(q2, 4)
    if a is None or b is None:
        return None
    na = math.sqrt(sum(value * value for value in a))
    nb = math.sqrt(sum(value * value for value in b))
    if na <= 0.0 or nb <= 0.0:
        return None
    dot = abs(sum((x / na) * (y / nb) for x, y in zip(a, b)))
    dot = min(1.0, max(0.0, dot))
    return 2.0 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), dot)


def _required_entity(entity: Mapping[str, Any]) -> None:
    if not isinstance(entity, Mapping):
        raise R3ContractError("entity is not an object")
    for key in ("logical_name", "alias_to", "role", "entity_id", "body_origin", "quat_xyzw"):
        if key not in entity:
            raise R3ContractError(f"entity missing {key}")
    if not str(entity["logical_name"]):
        raise R3ContractError("entity logical_name is empty")
    if entity["role"] not in ENTITY_ROLES:
        raise R3ContractError(f"unknown entity role: {entity['role']!r}")
    if _finite_vector(entity["body_origin"], 3) is None or _finite_vector(entity["quat_xyzw"], 4) is None:
        raise R3ContractError(f"nonfinite entity pose: {entity['logical_name']}")


def _required_contact_endpoint(entity: Mapping[str, Any]) -> None:
    if not isinstance(entity, Mapping):
        raise R3ContractError("contact endpoint is not an object")
    for key in ("logical_name", "role", "entity_id"):
        if key not in entity or not str(entity[key]):
            raise R3ContractError(f"contact endpoint missing {key}")
    if entity["role"] not in ENTITY_ROLES:
        raise R3ContractError(f"unknown contact endpoint role: {entity['role']!r}")


def _entity(row: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    matches = [item for item in row["entities"] if item.get("role") == role]
    if len(matches) != 1:
        return None
    return matches[0]


def _object_gripper_contact(row: Mapping[str, Any]) -> tuple[bool, bool, float]:
    found = False
    force_known = True
    max_force = 0.0
    for pair in row["contact_pairs"]:
        endpoints = (pair["entity_a"], pair["entity_b"])
        roles = {str(endpoint["role"]) for endpoint in endpoints}
        if "MANIPULATED_OBJECT" in roles and "GRIPPER" in roles:
            found = True
            force = float(pair["normal_constraint_force_scalar"])
            max_force = max(max_force, abs(force))
        if "normal_constraint_force_scalar" not in pair:
            force_known = False
    return found, force_known, max_force


def validate_contact_row(row: Mapping[str, Any], *, expected_step: int | None = None) -> None:
    """Validate one canonical FIT670 V2 row; no payload is modified."""
    forbidden = _walk_forbidden(row)
    if forbidden:
        raise R3ContractError(f"forbidden fields: {forbidden}")
    for key in ("episode_id", "step", "valid", "entities", "contact_pairs", "contact_ncon_total", "contact_truncated", "forward_before_capture", "eef_pos", "eef_quat_xyzw", "gripper_qpos", "protocol_steps_remaining"):
        if key not in row:
            raise R3ContractError(f"row missing {key}")
    if expected_step is not None and row["step"] != expected_step:
        raise R3ContractError(f"step closure expected {expected_step}, got {row['step']}")
    if not isinstance(row["step"], int) or row["step"] < 0 or not isinstance(row["valid"], bool):
        raise R3ContractError("invalid step or valid type")
    if _finite_vector(row["eef_pos"], 3) is None or _finite_vector(row["eef_quat_xyzw"], 4) is None or _finite_vector(row["gripper_qpos"], 2) is None:
        raise R3ContractError(f"nonfinite robot telemetry at step {row['step']}")
    if not isinstance(row["entities"], list) or not row["entities"]:
        raise R3ContractError(f"empty entities at step {row['step']}")
    for entity in row["entities"]:
        _required_entity(entity)
    if not any(item.get("role") == "MANIPULATED_OBJECT" for item in row["entities"]):
        raise R3ContractError(f"missing manipulated object at step {row['step']}")
    if not any(item.get("role") in {"OBJECT_TARGET", "REGION_TARGET"} for item in row["entities"]):
        raise R3ContractError(f"missing target entity at step {row['step']}")
    if not isinstance(row["contact_pairs"], list) or not isinstance(row["contact_ncon_total"], int):
        raise R3ContractError(f"invalid contact container at step {row['step']}")
    if row["contact_ncon_total"] != len(row["contact_pairs"]):
        raise R3ContractError(f"contact count mismatch at step {row['step']}")
    if row["contact_truncated"] is not False or row["forward_before_capture"] is not True:
        raise R3ContractError(f"contact capture contract failed at step {row['step']}")
    for pair in row["contact_pairs"]:
        if not isinstance(pair, Mapping) or "entity_a" not in pair or "entity_b" not in pair:
            raise R3ContractError(f"malformed contact pair at step {row['step']}")
        _required_contact_endpoint(pair["entity_a"])
        _required_contact_endpoint(pair["entity_b"])
        if _finite_vector(pair.get("position"), 3) is None or _finite_vector(pair.get("normal"), 3) is None:
            raise R3ContractError(f"nonfinite contact geometry at step {row['step']}")
        try:
            force = float(pair["normal_constraint_force_scalar"])
        except (KeyError, TypeError, ValueError):
            raise R3ContractError(f"missing normal constraint force at step {row['step']}") from None
        if not math.isfinite(force):
            raise R3ContractError(f"nonfinite normal constraint force at step {row['step']}")


def _position(entity: Mapping[str, Any] | None) -> list[float] | None:
    return _finite_vector(entity.get("body_origin") if entity else None, 3)


def _distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _delta(a: Sequence[float] | None, b: Sequence[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [x - y for x, y in zip(a, b)]


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float | None:
    if a is None or b is None:
        return None
    na = math.sqrt(sum(value * value for value in a))
    nb = math.sqrt(sum(value * value for value in b))
    if na <= 0.0 or nb <= 0.0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _contact_history(rows: Sequence[Mapping[str, Any]], index: int) -> tuple[bool, bool, float]:
    return _object_gripper_contact(rows[index])


def derive_episode_labels(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive five causal labels from a validated episode timeline."""
    if not rows:
        raise R3ContractError("empty episode")
    for index, row in enumerate(rows):
        validate_contact_row(row, expected_step=index)
    thresholds = protocol["teacher"]["frozen_thresholds"]
    initial_object = _position(_entity(rows[0], "MANIPULATED_OBJECT"))
    initial_object_z = initial_object[2] if initial_object is not None else None
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        obj = _entity(row, "MANIPULATED_OBJECT")
        target = _entity(row, "OBJECT_TARGET") or _entity(row, "REGION_TARGET")
        object_pos = _position(obj)
        target_pos = _position(target)
        eef_pos = _finite_vector(row["eef_pos"], 3)
        contact, force_known, force = _contact_history(rows, index)
        qpos = _finite_vector(row["gripper_qpos"], 2)
        close = qpos is not None and max(abs(value) for value in qpos) <= float(thresholds["qpos_close_threshold"])
        close_label = "TRUE" if close else "FALSE"
        close_reason = "PHYSICAL_QPOS"
        previous = rows[index - 1] if index else None
        previous_obj = _position(_entity(previous, "MANIPULATED_OBJECT")) if previous else None
        previous_eef = _finite_vector(previous["eef_pos"], 3) if previous else None
        object_delta_vector = _delta(object_pos, previous_obj)
        eef_delta_vector = _delta(eef_pos, previous_eef)
        object_delta = _distance(object_pos, previous_obj)
        eef_delta = _distance(eef_pos, previous_eef)
        lift = initial_object_z is not None and object_pos is not None and object_pos[2] - initial_object_z >= float(thresholds["lift_threshold_m"])
        if object_pos is None or eef_pos is None or object_delta is None or eef_delta is None:
            comotion = "UNKNOWN"
        elif object_delta < float(thresholds["comotion_min_displacement_m"]) or eef_delta < float(thresholds["comotion_min_displacement_m"]):
            comotion = "UNKNOWN"
        else:
            cosine = _cosine(object_delta_vector, eef_delta_vector)
            comotion = "UNKNOWN" if cosine is None else "TRUE" if cosine >= float(thresholds["comotion_cosine_threshold"]) else "FALSE"
        if object_pos is None or eef_pos is None:
            stable = "UNKNOWN"
        else:
            dwell = 0
            for previous_index in range(index, -1, -1):
                if not _contact_history(rows, previous_index)[0]:
                    break
                dwell += 1
            previous_relative = _delta(previous_obj, previous_eef)
            current_relative = _delta(object_pos, eef_pos)
            relative_motion = _distance(current_relative, previous_relative)
            stable = "TRUE" if contact and dwell >= 2 and relative_motion is not None and relative_motion <= float(thresholds["relative_motion_tolerance_m"]) else "FALSE" if force_known else "UNKNOWN"
        if not force_known or object_pos is None or eef_pos is None:
            physical = "UNKNOWN"
        elif not contact or stable == "FALSE":
            physical = "FALSE"
        elif stable == "UNKNOWN" or (not lift and comotion == "UNKNOWN"):
            physical = "UNKNOWN"
        else:
            physical = "TRUE" if lift or comotion == "TRUE" else "FALSE"
        if object_pos is None or target_pos is None:
            placement = "UNKNOWN"
        else:
            placement = "TRUE" if _distance(object_pos, target_pos) <= float(thresholds["placement_distance_threshold_m"]) else "FALSE"
        target_history: list[float] = []
        for previous_index in range(max(0, index - int(thresholds["placement_stability_consecutive_steps"])), index + 1):
            history_object = _position(_entity(rows[previous_index], "MANIPULATED_OBJECT"))
            history_target = _position(_entity(rows[previous_index], "OBJECT_TARGET") or _entity(rows[previous_index], "REGION_TARGET"))
            distance = _distance(history_object, history_target)
            if distance is None:
                target_history = []
                break
            target_history.append(distance)
        if len(target_history) < int(thresholds["placement_stability_consecutive_steps"]) + 1:
            placement_stability = "UNKNOWN"
        else:
            placement_stability = "TRUE" if max(target_history) - min(target_history) <= float(thresholds["placement_stability_translation_threshold_m"]) else "FALSE"
        released = "UNKNOWN" if qpos is None or not force_known else "TRUE" if max(abs(value) for value in qpos) >= float(thresholds["qpos_open_threshold"]) and not contact else "FALSE"
        safe = tri_and([placement, released, placement_stability])
        remaining = row.get("protocol_steps_remaining")
        k10 = "UNKNOWN" if not isinstance(remaining, int) else "TRUE" if remaining >= int(thresholds["k10"]) and safe != "TRUE" else "FALSE"
        previous_contact = bool(_contact_history(rows, index - 1)[0]) if index else False
        contact_loss = previous_contact and not contact
        slip = bool(contact and object_delta is not None and object_delta > float(thresholds["slip_relative_motion_threshold_m"]))
        instability = tri_or(["TRUE" if contact_loss or slip else "FALSE"] if force_known else ["UNKNOWN"])
        results.append({
            "step": index,
            "candidate_close": bool(row.get("candidate_close", False)),
            "labels": {
                "physical_criticality": _label(physical, "CONTACT_GEOMETRY_CAUSAL" if physical != "UNKNOWN" else "PHYSICAL_EVIDENCE_UNKNOWN"),
                "k10_feasibility": _label(k10, "PROTOCOL_HORIZON_AND_SAFE_RELEASE" if k10 != "UNKNOWN" else "HORIZON_UNKNOWN"),
                "safe_release": _label(safe, "PLACEMENT_RELEASE_STABILITY" if safe != "UNKNOWN" else "SAFE_RELEASE_COMPONENT_UNKNOWN"),
                "instability": _label(instability, "CONTACT_SLIP_TRANSITION" if instability != "UNKNOWN" else "CONTACT_EVIDENCE_UNKNOWN"),
                "gripper_closing_state": _label(close_label, close_reason),
            },
        })
    return results
