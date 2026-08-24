import json
import math
from pathlib import Path

import pytest

from gripper_attack.v5_r3_teacher import (
    R3ContractError,
    _aggregate_relation_label,
    _canonical_relation_bindings,
    _object_gripper_contact,
    canonicalize_fit670_episode,
    derive_episode_labels,
    quaternion_geodesic,
    tri_and,
    tri_or,
    validate_contact_row,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs" / "R3_DEV_PROTOCOL.json").read_text(encoding="utf-8"))


def _entity(name, role, position):
    return {
        "logical_name": name,
        "alias_to": None,
        "role": role,
        "entity_id": f"body:{name}",
        "body_origin": list(position),
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _row(step, *, contact=True, target=True, future=None):
    obj = _entity("cube_1", "MANIPULATED_OBJECT", [0.0, 0.0, 0.03 if step >= 2 else 0.0])
    entities = [obj, _entity("gripper", "GRIPPER", [0.0, 0.0, 0.03 if step >= 2 else 0.0])]
    if target:
        entities.append(_entity("target_1", "OBJECT_TARGET", [0.0, 0.0, 0.03 if step >= 2 else 0.0]))
    pair = {
        "entity_a": {"logical_name": "cube_1", "role": "MANIPULATED_OBJECT", "entity_id": "body:cube_1"},
        "entity_b": {"logical_name": "gripper", "role": "GRIPPER", "entity_id": "body:gripper"},
        "position": [0.0, 0.0, 0.03],
        "normal": [0.0, 0.0, 1.0],
        "normal_constraint_force_scalar": 1.0,
    }
    row = {
        "episode_id": "suite/task_00/state_00",
        "step": step,
        "valid": True,
        "candidate_close": contact,
        "entities": entities,
        "contact_pairs": [pair] if contact else [],
        "contact_ncon_total": 1 if contact else 0,
        "contact_truncated": False,
        "forward_before_capture": True,
        "eef_pos": [0.0, 0.0, 0.03 if step >= 2 else 0.0],
        "eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.0, 0.0] if contact else [0.03, 0.03],
        "protocol_steps_remaining": 520 - step,
    }
    if future is not None:
        row[future] = True
    return row


def test_tri_state_and_or_contract():
    assert tri_and(["TRUE", "TRUE"]) == "TRUE"
    assert tri_and(["TRUE", "UNKNOWN"]) == "UNKNOWN"
    assert tri_and(["FALSE", "UNKNOWN"]) == "FALSE"
    assert tri_or(["FALSE", "FALSE"]) == "FALSE"
    assert tri_or(["TRUE", "UNKNOWN"]) == "TRUE"
    assert tri_or(["FALSE", "UNKNOWN"]) == "UNKNOWN"


def test_q_minus_q_is_same_rotation():
    q = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    assert quaternion_geodesic(q, [-x for x in q]) < 1e-12


def test_contact_count_and_schema_are_fail_closed():
    row = _row(0)
    validate_contact_row(row, expected_step=0)
    row["contact_ncon_total"] = 0
    with pytest.raises(R3ContractError):
        validate_contact_row(row, expected_step=0)


def test_forbidden_outcome_and_future_fields_rejected():
    for key in ("task_success", "terminal", "reward", "outcome", "future_frame"):
        row = _row(0, future=key)
        with pytest.raises(R3ContractError):
            validate_contact_row(row, expected_step=0)


def test_teacher_is_causal_and_emits_all_five_heads():
    rows = [_row(step, contact=step < 4) for step in range(6)]
    derived = derive_episode_labels(rows, PROTOCOL)
    assert len(derived) == len(rows)
    assert set(derived[0]["labels"]) == {
        "physical_criticality", "k10_feasibility", "safe_release", "instability", "gripper_closing_state"
    }
    assert all({"value", "mask", "reason", "valid_mask", "evidence_fields", "right_censored"}.issubset(item["labels"][head]) for item in derived for head in item["labels"])
    assert all(item["labels"][head]["value"] in {"TRUE", "FALSE", "UNKNOWN"} for item in derived for head in item["labels"])


def test_future_suffix_does_not_change_prefix_labels():
    prefix = [_row(step, contact=step < 4) for step in range(4)]
    suffix_a = [_row(4, contact=False), _row(5, contact=False)]
    suffix_b = [_row(4, contact=True), _row(5, contact=True)]
    first = derive_episode_labels(prefix + suffix_a, PROTOCOL)[:4]
    second = derive_episode_labels(prefix + suffix_b, PROTOCOL)[:4]
    assert first == second


def test_k10_unknown_safe_release_is_not_false():
    # No object-gripper contact gives a known released state, while the
    # two-step prefix is too short to establish placement stability.
    rows = [_row(0, contact=False), _row(1, contact=False)]
    labels = derive_episode_labels(rows, PROTOCOL)
    assert labels[0]["labels"]["safe_release"]["value"] == "UNKNOWN"
    assert labels[0]["labels"]["k10_feasibility"]["value"] == "UNKNOWN"


def _multi_object_row(step, *, selected_contact=True):
    def entity(name, role, entity_id, z=0.0):
        return {
            "logical_name": name,
            "alias_to": None,
            "role": role,
            "entity_id": entity_id,
            "body_origin": [0.0, 0.0, z],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }

    selected = entity("cube_1", "MANIPULATED_OBJECT", 21, 0.03 if step else 0.0)
    other = entity("tomato_sauce_1", "MANIPULATED_OBJECT", 23)
    target = entity("basket_1_contain_region", "REGION_TARGET", 21)
    gripper = entity("gripper", "GRIPPER", 7, 0.03 if step else 0.0)
    selected_pair = {
        "entity_a": {"logical_name": "cube_1", "role": "MANIPULATED_OBJECT", "entity_id": 21},
        "entity_b": {"logical_name": "gripper", "role": "GRIPPER", "entity_id": 7},
        "position": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "normal_constraint_force_scalar": 1.0,
    }
    other_pair = {
        "entity_a": {"logical_name": "tomato_sauce_1", "role": "MANIPULATED_OBJECT", "entity_id": 23},
        "entity_b": {"logical_name": "gripper", "role": "GRIPPER", "entity_id": 7},
        "position": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "normal_constraint_force_scalar": 1.0,
    }
    pairs = [selected_pair if selected_contact else other_pair]
    return {
        "episode_id": "multi/object/state",
        "step": step,
        "valid": True,
        "candidate_close": True,
        "entities": [selected, other, target, gripper],
        "contact_pairs": pairs,
        "contact_ncon_total": len(pairs),
        "contact_truncated": False,
        "forward_before_capture": True,
        "eef_pos": [0.0, 0.0, 0.03 if step else 0.0],
        "eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.0, 0.0],
        "protocol_steps_remaining": 100 - step,
        "relation_bindings": [
            {
                "relation_index": 0,
                "predicate": "in",
                "object": {"logical_name": "cube_1", "alias_to": None, "role": "MANIPULATED_OBJECT", "entity_id": 21},
                "target": {"logical_name": "basket_1_contain_region", "alias_to": None, "role": "REGION_TARGET", "entity_id": 21},
            },
            {
                "relation_index": 1,
                "predicate": "in",
                "object": {"logical_name": "tomato_sauce_1", "alias_to": None, "role": "MANIPULATED_OBJECT", "entity_id": 23},
                "target": {"logical_name": "basket_1_contain_region", "alias_to": None, "role": "REGION_TARGET", "entity_id": 21},
            },
        ],
    }


def test_multi_object_relation_binding_is_not_role_only():
    rows = [_multi_object_row(0), _multi_object_row(1)]
    binding_episode = {
        "relations": [
            {
                "predicate": "In",
                "object_resolution": {"name": "cube_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 21},
                "target_resolution": {"name": "basket_1_contain_region", "semantic_role": "REGION_TARGET", "entity_id": 21},
            },
            {
                "predicate": "In",
                "object_resolution": {"name": "tomato_sauce_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 23},
                "target_resolution": {"name": "basket_1_contain_region", "semantic_role": "REGION_TARGET", "entity_id": 21},
            },
        ]
    }
    bindings = _canonical_relation_bindings(binding_episode, rows[0]["entities"])
    for row in rows:
        row["relation_bindings"] = bindings
    selected_contact = _object_gripper_contact(rows[0], rows[0]["entities"][0])
    other_contact = _object_gripper_contact(rows[0], rows[0]["entities"][1])
    assert selected_contact[0] is True
    assert other_contact[0] is False
    labels = derive_episode_labels(rows, PROTOCOL)
    assert len(labels) == 2
    assert len(labels[0]["relation_labels"]) == 2
    assert labels[1]["labels"]["physical_criticality"]["value"] == "TRUE"
    assert labels[1]["relation_labels"][0]["labels"]["physical_criticality"]["value"] == "TRUE"
    assert labels[1]["relation_labels"][1]["labels"]["physical_criticality"]["value"] == "FALSE"
    assert labels[0]["relation_labels"][0]["relation_identity"][0]["logical_name"] == "cube_1"
    assert labels[0]["relation_labels"][1]["relation_identity"][0]["logical_name"] == "tomato_sauce_1"


def test_relation_binding_requires_name_role_and_entity_id():
    entities = [
        {"logical_name": "cube_1", "alias_to": "cube_1_main", "role": "MANIPULATED_OBJECT", "entity_id": 21},
        {"logical_name": "basket_1_contain_region", "alias_to": "", "role": "REGION_TARGET", "entity_id": 21},
        {"logical_name": "tomato_sauce_1", "alias_to": "tomato_sauce_1_main", "role": "MANIPULATED_OBJECT", "entity_id": 23},
    ]
    episode = {
        "relations": [{
            "predicate": "In",
            "object_resolution": {"name": "cube_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 21},
            "target_resolution": {"name": "basket_1_contain_region", "semantic_role": "REGION_TARGET", "entity_id": 21},
        }]
    }
    binding = _canonical_relation_bindings(episode, entities)
    assert binding[0]["object"]["logical_name"] == "cube_1"
    episode["relations"][0]["object_resolution"]["entity_id"] = 23
    with pytest.raises(R3ContractError):
        _canonical_relation_bindings(episode, entities)


def test_relation_binding_rejects_non_target_role():
    entities = [
        {"logical_name": "cube_1", "alias_to": "cube_1_main", "role": "MANIPULATED_OBJECT", "entity_id": 21},
        {"logical_name": "other_1", "alias_to": "other_1_main", "role": "MANIPULATED_OBJECT", "entity_id": 23},
    ]
    episode = {
        "relations": [{
            "predicate": "In",
            "object_resolution": {"name": "cube_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 21},
            "target_resolution": {"name": "other_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 23},
        }]
    }
    with pytest.raises(R3ContractError):
        _canonical_relation_bindings(episode, entities)


def test_multi_relation_tri_state_aggregation_contract():
    def label(value, right_censored=False):
        return {"value": value, "evidence_fields": ["x"], "right_censored": right_censored}

    assert _aggregate_relation_label([label("TRUE"), label("UNKNOWN")], "physical_criticality")["value"] == "TRUE"
    unknown = _aggregate_relation_label([label("FALSE"), label("UNKNOWN")], "physical_criticality")
    assert unknown["value"] == "UNKNOWN"
    assert unknown["mask"] is False and unknown["valid_mask"] is False
    assert _aggregate_relation_label([label("FALSE"), label("FALSE")], "physical_criticality")["value"] == "FALSE"
    censored = _aggregate_relation_label([label("FALSE"), label("UNKNOWN", True)], "physical_criticality")
    assert censored["right_censored"] is True


def test_canonicalize_inserts_relation_bindings_on_every_row():
    def raw_entity(name, role, entity_id):
        return {
            "logical_name": name,
            "entity_name": f"{name}_main",
            "alias_to": f"{name}_main",
            "role": role,
            "entity_id": entity_id,
            "position": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    entities = [
        raw_entity("cube_1", "MANIPULATED_OBJECT", 21),
        raw_entity("tomato_sauce_1", "MANIPULATED_OBJECT", 23),
        raw_entity("basket_1_contain_region", "REGION_TARGET", 21),
    ]
    contact = {
        "body1": "cube_1_main", "body1_id": 21,
        "body2": "gripper0_left_finger", "body2_id": 7,
        "position": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
        "normal_constraint_force_scalar": 1.0,
    }
    episode = {
        "schema": "FIT670_EPISODE_V2",
        "episode_id": "libero_10/task_00/state_00",
        "suite": "libero_10", "task_id": 0, "state_id": 0,
        "relations": [
            {"predicate": "In", "object_resolution": {"name": "cube_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 21}, "target_resolution": {"name": "basket_1_contain_region", "semantic_role": "REGION_TARGET", "entity_id": 21}},
            {"predicate": "In", "object_resolution": {"name": "tomato_sauce_1", "semantic_role": "MANIPULATED_OBJECT", "entity_id": 23}, "target_resolution": {"name": "basket_1_contain_region", "semantic_role": "REGION_TARGET", "entity_id": 21}},
        ],
        "telemetry": [
            {
                "step": step, "horizon": 10, "entities": entities,
                "contact_pairs": [contact], "contact_ncon_total": 1,
                "contact_truncated": False, "forward_before_capture": True,
                "robot0_eef_pos": [0.0, 0.0, 0.0], "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
                "robot0_gripper_qpos": [0.0, 0.0],
            }
            for step in range(2)
        ],
        "steps": [{"step": step, "raw_action_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]} for step in range(2)],
    }
    rows = canonicalize_fit670_episode(episode)
    assert len(rows) == 2
    assert all([binding["object"]["logical_name"] for binding in row["relation_bindings"]] == ["cube_1", "tomato_sauce_1"] for row in rows)
