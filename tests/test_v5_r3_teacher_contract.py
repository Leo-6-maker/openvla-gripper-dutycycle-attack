import json
import math
from pathlib import Path

import pytest

from gripper_attack.v5_r3_teacher import (
    R3ContractError,
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
    assert all(set(item["labels"][head]) == {"value", "mask", "reason"} for item in derived for head in item["labels"])
    assert all(item["labels"][head]["value"] in {"TRUE", "FALSE", "UNKNOWN"} for item in derived for head in item["labels"])


def test_future_suffix_does_not_change_prefix_labels():
    prefix = [_row(step, contact=step < 4) for step in range(4)]
    suffix_a = [_row(4, contact=False), _row(5, contact=False)]
    suffix_b = [_row(4, contact=True), _row(5, contact=True)]
    first = derive_episode_labels(prefix + suffix_a, PROTOCOL)[:4]
    second = derive_episode_labels(prefix + suffix_b, PROTOCOL)[:4]
    assert first == second
