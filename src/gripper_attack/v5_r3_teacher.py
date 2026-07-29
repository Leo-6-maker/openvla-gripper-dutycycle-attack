"""Contact-complete, causal V23 Teacher for the R3 development canary.

This module intentionally has no fallback to the historical Fresh40 proxy.
It consumes only the canonical FIT670 V2 contact/geometry row contract and
returns explicit TRUE/FALSE/UNKNOWN labels.
"""
from __future__ import annotations

import math
import re
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
    "CONTACT_OTHER",
}
GRIPPER_BODY_PATTERN = re.compile(r"^gripper0_(?:.*finger.*|.*gripper.*)$")
FORBIDDEN_FIELDS = {
    "task_success", "terminal", "terminal_state", "reward", "outcome",
    "attack_result", "future", "future_frame", "future_label",
}


class R3ContractError(ValueError):
    """Raised when a canary row cannot be consumed without guessing."""


def _wxyz_to_xyzw(value: Any) -> list[float] | None:
    """Convert the collector's frozen MuJoCo wxyz convention to xyzw."""
    quat = _finite_vector(value, 4)
    return None if quat is None else [quat[1], quat[2], quat[3], quat[0]]


def _canonical_entity(entity: Mapping[str, Any]) -> dict[str, Any]:
    pose = entity.get("world_pose") if isinstance(entity.get("world_pose"), Mapping) else {}
    position = entity.get("position", pose.get("position"))
    quaternion = entity.get("rotation_wxyz", pose.get("quaternion"))
    converted = _wxyz_to_xyzw(quaternion)
    if _finite_vector(position, 3) is None or converted is None:
        raise R3ContractError(f"nonfinite collector entity pose: {entity.get('logical_name')!r}")
    role = str(entity.get("role", ""))
    if role not in ENTITY_ROLES - {"CONTACT_OTHER"}:
        raise R3ContractError(f"unknown collector entity role: {role!r}")
    for key in ("logical_name", "alias_to", "entity_id"):
        if key not in entity:
            raise R3ContractError(f"collector entity missing {key}")
    return {
        "logical_name": str(entity["logical_name"]),
        "alias_to": str(entity.get("alias_to") or ""),
        "role": role,
        "entity_id": int(entity["entity_id"]),
        "body_origin": [float(x) for x in position],
        "quat_xyzw": converted,
        "collector_entity_name": str(entity.get("entity_name") or ""),
    }


def _contact_endpoint(raw_name: Any, raw_id: Any, entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind a recorded contact endpoint without using the collector flag.

    The source collector records MuJoCo body names/ids but not endpoint role
    objects. Exact entity-name/id matches win; the frozen robot body prefix is
    the only allowed gripper binding. Everything else remains explicit
    CONTACT_OTHER rather than being relabeled as support.
    """
    name = str(raw_name or "")
    try:
        entity_id = int(raw_id)
    except (TypeError, ValueError):
        raise R3ContractError(f"contact endpoint has invalid entity id: {raw_id!r}") from None
    matches = []
    for entity in entities:
        names = {
            str(entity.get("entity_name") or ""),
            str(entity.get("logical_name") or ""),
            str(entity.get("alias_to") or ""),
        }
        if name in names and int(entity.get("entity_id", -1)) == entity_id:
            matches.append(entity)
    if len(matches) > 1:
        raise R3ContractError(f"ambiguous contact endpoint binding: {name!r}/{entity_id}")
    if matches:
        entity = matches[0]
        return {
            "logical_name": name,
            "role": str(entity["role"]),
            "entity_id": entity_id,
        }
    if GRIPPER_BODY_PATTERN.fullmatch(name):
        return {"logical_name": name, "role": "GRIPPER", "entity_id": entity_id, "binding_source": "FROZEN_GRIPPER_BODY_PATTERN"}
    raise R3ContractError(f"unbound contact endpoint: {name!r}/{entity_id}")


def canonicalize_fit670_episode(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Adapt an atomically sealed FIT670_EPISODE_V2 to the R3 row contract.

    Only current telemetry and the same-step raw action are consumed. The
    adapter deliberately drops sim_state, images, outcomes, and future rows.
    """
    if episode.get("schema") != "FIT670_EPISODE_V2":
        raise R3ContractError(f"unexpected FIT670 episode schema: {episode.get('schema')!r}")
    telemetry = episode.get("telemetry")
    steps = episode.get("steps")
    if not isinstance(telemetry, list) or not isinstance(steps, list) or len(telemetry) != len(steps) or not telemetry:
        raise R3ContractError("FIT670 telemetry/step closure is incomplete")
    episode_id = str(episode.get("episode_id") or "")
    if not episode_id:
        raise R3ContractError("FIT670 episode_id is empty")
    from .action_contract import raw_gripper_is_close

    rows: list[dict[str, Any]] = []
    gripper_body_ids: dict[str, int] = {}
    for expected_step, (raw, action) in enumerate(zip(telemetry, steps)):
        if not isinstance(raw, Mapping) or not isinstance(action, Mapping):
            raise R3ContractError(f"malformed FIT670 row at step {expected_step}")
        if raw.get("step") != expected_step or action.get("step") != expected_step:
            raise R3ContractError(f"FIT670 step closure failed at {expected_step}")
        entities_raw = raw.get("entities")
        if not isinstance(entities_raw, list) or not entities_raw:
            raise R3ContractError(f"FIT670 entities missing at step {expected_step}")
        entities = [_canonical_entity(item) for item in entities_raw]
        if not any(item["role"] == "MANIPULATED_OBJECT" for item in entities):
            raise R3ContractError(f"FIT670 manipulated object missing at {expected_step}")
        if not any(item["role"] in {"OBJECT_TARGET", "REGION_TARGET"} for item in entities):
            raise R3ContractError(f"FIT670 target missing at {expected_step}")
        contact_pairs = []
        raw_pairs = raw.get("contact_pairs")
        if not isinstance(raw_pairs, list) or raw.get("contact_ncon_total") != len(raw_pairs):
            raise R3ContractError(f"FIT670 contact closure failed at {expected_step}")
        for pair in raw_pairs:
            if not isinstance(pair, Mapping):
                raise R3ContractError(f"malformed FIT670 contact at {expected_step}")
            for key in ("body1", "body1_id", "body2", "body2_id", "position", "normal", "normal_constraint_force_scalar"):
                if key not in pair:
                    raise R3ContractError(f"FIT670 contact missing {key} at {expected_step}")
            for body_name, body_id in ((pair["body1"], pair["body1_id"]), (pair["body2"], pair["body2_id"])):
                name = str(body_name or "")
                if GRIPPER_BODY_PATTERN.fullmatch(name):
                    try:
                        numeric_id = int(body_id)
                    except (TypeError, ValueError):
                        raise R3ContractError(f"invalid gripper body id at {expected_step}") from None
                    previous_id = gripper_body_ids.setdefault(name, numeric_id)
                    if previous_id != numeric_id:
                        raise R3ContractError(f"gripper body identity changed: {name}")
            contact_pairs.append({
                "entity_a": _contact_endpoint(pair["body1"], pair["body1_id"], entities_raw),
                "entity_b": _contact_endpoint(pair["body2"], pair["body2_id"], entities_raw),
                "position": pair["position"],
                "normal": pair["normal"],
                "normal_constraint_force_scalar": pair["normal_constraint_force_scalar"],
                "collector_object_gripper_flag": bool(pair.get("is_object_gripper_contact", False)),
            })
        horizon = raw.get("horizon")
        if not isinstance(horizon, int) or horizon <= expected_step:
            raise R3ContractError(f"FIT670 horizon is invalid at {expected_step}")
        raw_action = action.get("raw_action_7d")
        if not isinstance(raw_action, list) or len(raw_action) != 7:
            raise R3ContractError(f"FIT670 raw action is invalid at {expected_step}")
        try:
            raw_gripper = float(raw_action[6])
        except (TypeError, ValueError):
            raise R3ContractError(f"FIT670 raw gripper is invalid at {expected_step}") from None
        if not math.isfinite(raw_gripper) or not 0.0 <= raw_gripper <= 1.0:
            raise R3ContractError(f"FIT670 raw gripper is invalid at {expected_step}")
        candidate_close = raw_gripper_is_close(raw_gripper)
        rows.append({
            "episode_id": episode_id,
            "suite": episode.get("suite"),
            "task_id": episode.get("task_id"),
            "state_id": episode.get("state_id"),
            "seed": episode.get("collection_seed"),
            "step": expected_step,
            "valid": True,
            "entities": entities,
            "contact_pairs": contact_pairs,
            "contact_ncon_total": int(raw["contact_ncon_total"]),
            "contact_truncated": raw.get("contact_truncated"),
            "forward_before_capture": raw.get("forward_before_capture"),
            "eef_pos": raw.get("robot0_eef_pos"),
            "eef_quat_xyzw": _wxyz_to_xyzw(raw.get("robot0_eef_quat")),
            "gripper_qpos": raw.get("robot0_gripper_qpos"),
            "protocol_steps_remaining": int(horizon - expected_step - 1),
            "candidate_close": bool(candidate_close),
            "candidate_close_source": "FIT670_STEP.raw_action_7d[6]",
        })
    return rows


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
    for key in ("episode_id", "step", "valid", "candidate_close", "entities", "contact_pairs", "contact_ncon_total", "contact_truncated", "forward_before_capture", "eef_pos", "eef_quat_xyzw", "gripper_qpos", "protocol_steps_remaining"):
        if key not in row:
            raise R3ContractError(f"row missing {key}")
    if expected_step is not None and row["step"] != expected_step:
        raise R3ContractError(f"step closure expected {expected_step}, got {row['step']}")
    if not isinstance(row["step"], int) or row["step"] < 0 or not isinstance(row["valid"], bool) or not isinstance(row["candidate_close"], bool):
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
        k10 = (
            "UNKNOWN"
            if not isinstance(remaining, int) or safe == "UNKNOWN"
            else "TRUE" if remaining >= int(thresholds["k10"]) else "FALSE"
        )
        previous_contact = bool(_contact_history(rows, index - 1)[0]) if index else False
        contact_loss = previous_contact and not contact
        slip = bool(contact and object_delta is not None and object_delta > float(thresholds["slip_relative_motion_threshold_m"]))
        instability = tri_or(["TRUE" if contact_loss or slip else "FALSE"] if force_known else ["UNKNOWN"])
        relation_identity = [
            {
                "logical_name": item.get("logical_name"),
                "alias_to": item.get("alias_to"),
                "role": item.get("role"),
                "entity_id": item.get("entity_id"),
            }
            for item in row["entities"]
            if item.get("role") in {"MANIPULATED_OBJECT", "OBJECT_TARGET", "REGION_TARGET"}
        ]
        right_censored = index == len(rows) - 1 and int(row["protocol_steps_remaining"]) > 0
        evidence = {
            "physical_criticality": ["object_gripper_contact", "contact_force", "object_eef_relative_pose", "object_eef_comotion", "lift", "support_state"],
            "k10_feasibility": ["protocol_steps_remaining", "safe_release_computed"],
            "safe_release": ["placement", "released_state", "placement_stability"],
            "instability": ["contact_transition", "relative_slip", "regrasp", "contact_loss"],
            "gripper_closing_state": ["physical_gripper_qpos"],
        }
        results.append({
            "episode_id": row["episode_id"],
            "suite": row.get("suite"),
            "task_id": row.get("task_id"),
            "state_id": row.get("state_id"),
            "seed": row.get("seed"),
            "step": index,
            "candidate_close": bool(row["candidate_close"]),
            "relation_identity": relation_identity,
            "right_censored": right_censored,
            "evidence_fields": evidence,
            "labels": {
                head: {
                    **label,
                    "valid_mask": bool(label["mask"]),
                    "evidence_fields": evidence[head],
                    "right_censored": right_censored,
                }
                for head, label in {
                    "physical_criticality": _label(physical, "CONTACT_GEOMETRY_CAUSAL" if physical != "UNKNOWN" else "PHYSICAL_EVIDENCE_UNKNOWN"),
                    "k10_feasibility": _label(k10, "PROTOCOL_HORIZON_AND_SAFE_RELEASE" if k10 != "UNKNOWN" else "HORIZON_OR_SAFE_RELEASE_UNKNOWN"),
                    "safe_release": _label(safe, "PLACEMENT_RELEASE_STABILITY" if safe != "UNKNOWN" else "SAFE_RELEASE_COMPONENT_UNKNOWN"),
                    "instability": _label(instability, "CONTACT_SLIP_TRANSITION" if instability != "UNKNOWN" else "CONTACT_EVIDENCE_UNKNOWN"),
                    "gripper_closing_state": _label(close_label, close_reason),
                }.items()
            },
        })
    return results
