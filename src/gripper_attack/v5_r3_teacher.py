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


def _canonical_relation_bindings(episode: Mapping[str, Any], entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bind each declared relation to one object and one target entity.

    FIT670 can contain several manipulated objects in one episode.  Role-only
    lookup is therefore unsafe; relation resolution name, role, and entity id
    must agree before a relation can be consumed.
    """
    raw_relations = episode.get("relations")
    if not isinstance(raw_relations, list):
        raise R3ContractError("collector relations must be a list")
    if not raw_relations:
        return []
    bindings: list[dict[str, Any]] = []
    for relation_index, relation in enumerate(raw_relations):
        if not isinstance(relation, Mapping):
            raise R3ContractError(f"malformed relation binding: {relation_index}")
        object_resolution = relation.get("object_resolution")
        target_resolution = relation.get("target_resolution")
        if not isinstance(object_resolution, Mapping) or not isinstance(target_resolution, Mapping):
            raise R3ContractError(f"relation resolution missing: {relation_index}")
        object_role = str(object_resolution.get("semantic_role") or relation.get("object_semantic_role") or "")
        target_role = str(target_resolution.get("semantic_role") or relation.get("target_semantic_role") or "")
        object_name = str(object_resolution.get("name") or relation.get("object_bddl") or "")
        target_name = str(target_resolution.get("name") or relation.get("target_bddl") or "")
        try:
            object_id = int(object_resolution["entity_id"])
            target_id = int(target_resolution["entity_id"])
        except (KeyError, TypeError, ValueError):
            raise R3ContractError(f"relation entity id missing: {relation_index}") from None
        object_matches = [
            entity for entity in entities
            if entity.get("logical_name") == object_name
            and entity.get("role") == object_role == "MANIPULATED_OBJECT"
            and int(entity.get("entity_id", -1)) == object_id
        ]
        if target_role not in {"OBJECT_TARGET", "REGION_TARGET"}:
            raise R3ContractError(f"relation target role is not a target: {relation_index}")
        target_matches = [
            entity for entity in entities
            if entity.get("logical_name") == target_name
            and entity.get("role") == target_role
            and int(entity.get("entity_id", -1)) == target_id
        ]
        if len(object_matches) != 1 or len(target_matches) != 1:
            raise R3ContractError(f"relation entity binding mismatch: {relation_index}")
        bindings.append({
            "relation_index": relation_index,
            "predicate": str(relation.get("predicate") or ""),
            "object": {
                "logical_name": object_name,
                "alias_to": object_matches[0].get("alias_to"),
                "role": object_role,
                "entity_id": object_id,
            },
            "target": {
                "logical_name": target_name,
                "alias_to": target_matches[0].get("alias_to"),
                "role": target_role,
                "entity_id": target_id,
            },
        })
    return bindings


def _contact_endpoint(raw_name: Any, raw_id: Any, entities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind a recorded contact endpoint without using the collector flag.

    The source collector records MuJoCo body names/ids but not endpoint role
    objects. Exact entity-name/id matches win; the frozen robot body prefix is
    the only allowed gripper binding. Everything else remains explicit
    CONTACT_OTHER rather than being relabeled as support.
    """
    name = str(raw_name or "")
    if not name:
        raise R3ContractError(f"unbound contact endpoint: {name!r}/{raw_id!r}")
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
    if name.startswith("gripper0_"):
        raise R3ContractError(f"unbound contact endpoint: {name!r}/{entity_id}")
    return {
        "logical_name": name,
        "role": "CONTACT_OTHER",
        "entity_id": entity_id,
        "binding_source": "UNRESOLVED_NON_GRIPPER_CONTACT_OTHER",
    }


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
    relations = episode.get("relations")
    geometry_not_applicable = episode.get("geometry_status") == "NOT_APPLICABLE" and isinstance(relations, list) and len(relations) == 0
    gripper_body_ids: dict[str, int] = {}
    for expected_step, (raw, action) in enumerate(zip(telemetry, steps)):
        if not isinstance(raw, Mapping) or not isinstance(action, Mapping):
            raise R3ContractError(f"malformed FIT670 row at step {expected_step}")
        if raw.get("step") != expected_step or action.get("step") != expected_step:
            raise R3ContractError(f"FIT670 step closure failed at {expected_step}")
        entities_raw = raw.get("entities")
        if not isinstance(entities_raw, list) or (not entities_raw and not geometry_not_applicable):
            raise R3ContractError(f"FIT670 entities missing at step {expected_step}")
        entities = [_canonical_entity(item) for item in entities_raw]
        if not geometry_not_applicable and not any(item["role"] == "MANIPULATED_OBJECT" for item in entities):
            raise R3ContractError(f"FIT670 manipulated object missing at {expected_step}")
        if not geometry_not_applicable and not any(item["role"] in {"OBJECT_TARGET", "REGION_TARGET"} for item in entities):
            raise R3ContractError(f"FIT670 target missing at {expected_step}")
        relation_bindings = _canonical_relation_bindings(episode, entities)
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
            "geometry_status": "NOT_APPLICABLE" if geometry_not_applicable else "APPLICABLE",
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
            "relation_bindings": relation_bindings,
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


def _selected_entity(row: Mapping[str, Any], role: str, relation_binding: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if relation_binding is None:
        return _entity(row, role)
    side = "object" if role == "MANIPULATED_OBJECT" else "target"
    expected = relation_binding.get(side)
    if not isinstance(expected, Mapping):
        return None
    matches = [
        item for item in row["entities"]
        if item.get("role") == expected.get("role")
        and item.get("logical_name") == expected.get("logical_name")
        and int(item.get("entity_id", -1)) == int(expected.get("entity_id", -2))
    ]
    return matches[0] if len(matches) == 1 else None


def _same_entity(endpoint: Mapping[str, Any], entity: Mapping[str, Any] | None) -> bool:
    if entity is None or endpoint.get("role") != entity.get("role"):
        return False
    try:
        return int(endpoint.get("entity_id", -1)) == int(entity.get("entity_id", -2))
    except (TypeError, ValueError):
        return False


def _object_gripper_contact(row: Mapping[str, Any], object_entity: Mapping[str, Any] | None = None) -> tuple[bool, bool, float]:
    found = False
    force_known = True
    max_force = 0.0
    for pair in row["contact_pairs"]:
        endpoints = (pair["entity_a"], pair["entity_b"])
        roles = {str(endpoint["role"]) for endpoint in endpoints}
        if "MANIPULATED_OBJECT" in roles and "GRIPPER" in roles:
            object_endpoint = next(endpoint for endpoint in endpoints if endpoint.get("role") == "MANIPULATED_OBJECT")
            if object_entity is not None and not _same_entity(object_endpoint, object_entity):
                continue
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
    not_applicable = row.get("geometry_status") == "NOT_APPLICABLE" and row.get("relation_bindings") == []
    if not isinstance(row["entities"], list) or (not row["entities"] and not not_applicable):
        raise R3ContractError(f"empty entities at step {row['step']}")
    if not not_applicable:
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


def _contact_history(rows: Sequence[Mapping[str, Any]], index: int, object_entity: Mapping[str, Any] | None = None) -> tuple[bool, bool, float]:
    return _object_gripper_contact(rows[index], object_entity)


def _derive_not_applicable_labels(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = protocol["teacher"]["frozen_thresholds"]
    evidence = {
        "physical_criticality": ["geometry_not_applicable"],
        "k10_feasibility": ["safe_release_computed"],
        "safe_release": ["placement", "released_state", "placement_stability"],
        "instability": ["contact_transition", "relative_slip", "regrasp", "contact_loss"],
        "gripper_closing_state": ["physical_gripper_qpos"],
    }
    output = []
    for index, row in enumerate(rows):
        qpos = _finite_vector(row["gripper_qpos"], 2)
        if qpos is None:
            closing = _label("UNKNOWN", "NONFINITE_GRIPPER_QPOS")
        else:
            closing = _label(
                "TRUE" if max(abs(value) for value in qpos) <= float(thresholds["qpos_close_threshold"]) else "FALSE",
                "PHYSICAL_QPOS",
            )
        labels = {
            "physical_criticality": _label("UNKNOWN", "GEOMETRY_NOT_APPLICABLE"),
            "k10_feasibility": _label("UNKNOWN", "GEOMETRY_NOT_APPLICABLE"),
            "safe_release": _label("UNKNOWN", "GEOMETRY_NOT_APPLICABLE"),
            "instability": _label("UNKNOWN", "GEOMETRY_NOT_APPLICABLE"),
            "gripper_closing_state": closing,
        }
        right_censored = index == len(rows) - 1 and int(row["protocol_steps_remaining"]) > 0
        output.append({
            "episode_id": row["episode_id"], "suite": row.get("suite"), "task_id": row.get("task_id"),
            "state_id": row.get("state_id"), "seed": row.get("seed"), "step": index,
            "candidate_close": bool(row["candidate_close"]), "relation_identity": [], "relation_index": None,
            "right_censored": right_censored, "evidence_fields": evidence,
            "labels": {
                head: {**label, "valid_mask": bool(label["mask"]), "evidence_fields": evidence[head], "right_censored": right_censored}
                for head, label in labels.items()
            },
        })
    return output


def _derive_single_relation_labels(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    relation_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Derive five causal labels for one relation from a validated timeline."""
    if not rows:
        raise R3ContractError("empty episode")
    for index, row in enumerate(rows):
        validate_contact_row(row, expected_step=index)
    if all(row.get("geometry_status") == "NOT_APPLICABLE" and row.get("relation_bindings") == [] for row in rows):
        return _derive_not_applicable_labels(rows, protocol)
    thresholds = protocol["teacher"]["frozen_thresholds"]
    initial_object = _position(_selected_entity(rows[0], "MANIPULATED_OBJECT", relation_binding))
    initial_object_z = initial_object[2] if initial_object is not None else None
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        obj = _selected_entity(row, "MANIPULATED_OBJECT", relation_binding)
        target = _selected_entity(row, "OBJECT_TARGET", relation_binding) or _selected_entity(row, "REGION_TARGET", relation_binding)
        object_pos = _position(obj)
        target_pos = _position(target)
        eef_pos = _finite_vector(row["eef_pos"], 3)
        contact, force_known, force = _contact_history(rows, index, obj)
        qpos = _finite_vector(row["gripper_qpos"], 2)
        close = qpos is not None and max(abs(value) for value in qpos) <= float(thresholds["qpos_close_threshold"])
        close_label = "TRUE" if close else "FALSE"
        close_reason = "PHYSICAL_QPOS"
        previous = rows[index - 1] if index else None
        previous_obj = _position(_selected_entity(previous, "MANIPULATED_OBJECT", relation_binding)) if previous else None
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
                if not _contact_history(rows, previous_index, _selected_entity(rows[previous_index], "MANIPULATED_OBJECT", relation_binding))[0]:
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
            history_object = _position(_selected_entity(rows[previous_index], "MANIPULATED_OBJECT", relation_binding))
            history_target = _position(_selected_entity(rows[previous_index], "OBJECT_TARGET", relation_binding) or _selected_entity(rows[previous_index], "REGION_TARGET", relation_binding))
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
        previous_contact = bool(_contact_history(rows, index - 1, _selected_entity(rows[index - 1], "MANIPULATED_OBJECT", relation_binding))[0]) if index else False
        contact_loss = previous_contact and not contact
        slip = bool(contact and object_delta is not None and object_delta > float(thresholds["slip_relative_motion_threshold_m"]))
        instability = tri_or(["TRUE" if contact_loss or slip else "FALSE"] if force_known else ["UNKNOWN"])
        if physical == "UNKNOWN":
            if obj is None:
                physical_reason = "MISSING_OBJECT_IDENTITY"
            elif object_pos is None or eef_pos is None or object_delta is None or eef_delta is None:
                physical_reason = "MISSING_GEOMETRY"
            elif not force_known:
                physical_reason = "MISSING_CONTACT_EVIDENCE"
            elif stable == "UNKNOWN":
                physical_reason = "STABILITY_UNKNOWN"
            elif comotion == "UNKNOWN":
                physical_reason = "INSUFFICIENT_CAUSAL_PREFIX"
            else:
                physical_reason = "PHYSICAL_EVIDENCE_UNKNOWN"
        else:
            physical_reason = "CONTACT_GEOMETRY_CAUSAL"
        safe_reason = "PLACEMENT_RELEASE_STABILITY" if safe != "UNKNOWN" else "SAFE_RELEASE_COMPONENT_UNKNOWN"
        k10_reason = "PROTOCOL_HORIZON_AND_SAFE_RELEASE" if k10 != "UNKNOWN" else "HORIZON_OR_SAFE_RELEASE_UNKNOWN"
        if relation_binding is None:
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
        else:
            relation_identity = [relation_binding["object"], relation_binding["target"]]
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
            "relation_index": relation_binding.get("relation_index") if relation_binding is not None else None,
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
                    "physical_criticality": _label(physical, physical_reason),
                    "k10_feasibility": _label(k10, k10_reason),
                    "safe_release": _label(safe, safe_reason),
                    "instability": _label(instability, "CONTACT_SLIP_TRANSITION" if instability != "UNKNOWN" else "CONTACT_EVIDENCE_UNKNOWN"),
                    "gripper_closing_state": _label(close_label, close_reason),
                }.items()
            },
        })
    return results


def _aggregate_relation_label(labels: Sequence[Mapping[str, Any]], head: str) -> dict[str, Any]:
    """Aggregate one head across relations without converting UNKNOWN to FALSE."""
    values = [str(label["value"]) for label in labels]
    value = tri_or(values)
    if value == "TRUE":
        reason = "MULTI_RELATION_OR_TRUE"
    elif value == "FALSE":
        reason = "MULTI_RELATION_OR_FALSE"
    else:
        reason = "MULTI_RELATION_OR_UNKNOWN"
    evidence_fields = sorted({
        str(field)
        for label in labels
        for field in label.get("evidence_fields", [])
    })
    right_censored = any(bool(label.get("right_censored")) for label in labels)
    return {
        "value": value,
        "mask": value != "UNKNOWN",
        "valid_mask": value != "UNKNOWN",
        "reason": reason,
        "evidence_fields": evidence_fields,
        "right_censored": right_censored,
    }


def derive_episode_labels(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive one step stream, aggregating declared multi-object relations.

    A source episode remains one contiguous step stream.  Each declared
    relation is evaluated independently, then heads are combined with the
    frozen tri-state OR semantics: TRUE wins, all-known FALSE is FALSE, and
    otherwise the result is UNKNOWN.
    """
    if not rows:
        raise R3ContractError("empty episode")
    bindings = rows[0].get("relation_bindings")
    if not bindings:
        return _derive_single_relation_labels(rows, protocol, None)
    if not isinstance(bindings, list):
        raise R3ContractError("relation_bindings is not a list")
    for row in rows:
        if row.get("relation_bindings") != bindings:
            raise R3ContractError("relation binding changed within episode")
    per_relation = [
        _derive_single_relation_labels(rows, protocol, binding)
        for binding in bindings
    ]
    if not per_relation or any(len(stream) != len(rows) for stream in per_relation):
        raise R3ContractError("relation label stream is incomplete")
    output: list[dict[str, Any]] = []
    for step_index, source_row in enumerate(rows):
        relation_rows = [stream[step_index] for stream in per_relation]
        labels = {
            head: _aggregate_relation_label(
                [relation_row["labels"][head] for relation_row in relation_rows], head
            )
            for head in HEADS
        }
        output.append({
            "episode_id": source_row["episode_id"],
            "suite": source_row.get("suite"),
            "task_id": source_row.get("task_id"),
            "state_id": source_row.get("state_id"),
            "seed": source_row.get("seed"),
            "step": step_index,
            "candidate_close": bool(source_row["candidate_close"]),
            "relation_identity": [binding["object"] | {"side": "object"} for binding in bindings]
            + [binding["target"] | {"side": "target"} for binding in bindings],
            "relation_indices": [int(binding["relation_index"]) for binding in bindings],
            "relation_count": len(bindings),
            "relation_bindings": bindings,
            "relation_labels": [
                {
                    "relation_index": relation_row.get("relation_index"),
                    "relation_identity": relation_row.get("relation_identity"),
                    "labels": relation_row["labels"],
                }
                for relation_row in relation_rows
            ],
            "right_censored": any(bool(relation_row.get("right_censored")) for relation_row in relation_rows),
            "evidence_fields": {
                head: labels[head]["evidence_fields"] for head in HEADS
            },
            "labels": labels,
        })
    return output
