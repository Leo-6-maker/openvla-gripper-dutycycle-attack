import math
import pytest

from gripper_attack.v5_r3_teacher import R3ContractError, canonicalize_fit670_episode


def _entity(name, role, entity_id):
    return {
        "logical_name": name,
        "alias_to": None,
        "role": role,
        "entity_id": entity_id,
        "entity_name": name,
        "position": [0.0, 0.0, 0.0],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _episode(raw_grippers=(0.0, 1.0)):
    rows = []
    steps = []
    for step, raw_gripper in enumerate(raw_grippers):
        rows.append({
            "step": step,
            "horizon": 10,
            "entities": [
                _entity("cube_1", "MANIPULATED_OBJECT", 10),
                _entity("target_1", "OBJECT_TARGET", 11),
            ],
            "contact_pairs": [{
                "body1": "cube_1", "body1_id": 10,
                "body2": "gripper0_leftfinger", "body2_id": 20,
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "normal_constraint_force_scalar": 1.0,
            }],
            "contact_ncon_total": 1,
            "contact_truncated": False,
            "forward_before_capture": True,
            "robot0_eef_pos": [0.0, 0.0, 0.0],
            "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
            "robot0_gripper_qpos": [0.0, 0.0],
        })
        steps.append({"step": step, "raw_action_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, raw_gripper]})
    return {
        "schema": "FIT670_EPISODE_V2",
        "episode_id": "libero_10/task_00/state_00",
        "suite": "libero_10",
        "task_id": 0,
        "state_id": 0,
        "collection_seed": 0,
        "telemetry": rows,
        "steps": steps,
    }


def test_fit670_adapter_uses_raw_action_gripper_only():
    rows = canonicalize_fit670_episode(_episode())
    assert [row["candidate_close"] for row in rows] == [True, False]
    assert rows[0]["candidate_close_source"] == "FIT670_STEP.raw_action_7d[6]"
    assert rows[0]["eef_quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert rows[0]["protocol_steps_remaining"] == 9
    assert rows[1]["protocol_steps_remaining"] == 8


def test_fit670_adapter_rejects_incomplete_horizon():
    episode = _episode()
    episode["telemetry"][1]["horizon"] = 1
    with pytest.raises(R3ContractError, match="horizon"):
        canonicalize_fit670_episode(episode)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_fit670_adapter_rejects_nonfinite_or_out_of_range_raw_gripper(value):
    episode = _episode()
    episode["steps"][0]["raw_action_7d"][6] = value
    with pytest.raises(R3ContractError, match="raw gripper"):
        canonicalize_fit670_episode(episode)


def test_fit670_adapter_rejects_unbound_gripper_body():
    episode = _episode()
    episode["telemetry"][0]["contact_pairs"][0]["body2"] = "gripper0_unknown_body"
    with pytest.raises(R3ContractError, match="unbound contact endpoint"):
        canonicalize_fit670_episode(episode)


def test_fit670_adapter_rejects_gripper_body_id_drift():
    episode = _episode()
    episode["telemetry"][1]["contact_pairs"][0]["body2_id"] = 21
    with pytest.raises(R3ContractError, match="identity changed"):
        canonicalize_fit670_episode(episode)
