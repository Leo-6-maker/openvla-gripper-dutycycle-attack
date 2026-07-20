from __future__ import annotations

from copy import deepcopy

from gripper_attack.v5_physics import derive_episode_rows, parse_bddl_task_role


PROTOCOL = {
    "candidate_close": {"close_threshold": 0.5},
    "history": {"score_window_steps": 5, "minimum_stable_grasp_dwell": 10},
    "fixed_constants": {
        "relative_position_scale_m": 0.03,
        "relative_quaternion_scale": 0.20,
        "lift_scale_m": 0.03,
        "target_progress_scale_m": 0.20,
        "tier3_min_utility": 0.75,
        "tier3_min_stable_grasp": 0.70,
        "tier3_min_lift": 0.50,
        "tier3_max_release_risk": 0.35,
        "tier3_max_regrasp_risk": 0.35,
        "tier2_min_utility": 0.50,
        "tier2_min_stable_grasp": 0.50,
        "tier2_max_release_risk": 0.60,
        "tier2_max_regrasp_risk": 0.60,
        "tier1_min_utility": 0.25,
    },
}


def _bddl(goal: str) -> str:
    return f"""(define (problem test)\n(:objects\n  obj_1 - cube\n  target_1 - plate\n)\n(:init\n  (On obj_1 floor_region)\n)\n(:goal\n  {goal}\n)\n)\n"""


def _episode(length: int = 12):
    steps = []
    sidecars = []
    for index in range(length):
        z = 0.03 * index / max(1, length - 1)
        # FIXED (Gate D2.0): raw=0.0 = CLOSE in OpenVLA space (was 1.0=OPEN, wrong)
        steps.append({"clean_action_raw_7d": [0, 0, 0, 0, 0, 0, 0.0], "valid": True})
        state = [0.1 * index / max(1, length - 1), 0.0, z, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        pairs = [["obj_1_g1", "gripper0_finger1_pad_collision"]]
        if index == 0:
            pairs.append(["obj_1_g1", "floor"])
        sidecars.append({"object_state": state, "robot0_eef_pos": [0.1 * index / max(1, length - 1), 0.0, z], "robot0_gripper_qpos": [0.0, 0.0], "mujoco_contact_pairs": pairs})
    return steps, sidecars


def test_task_role_uses_goal_object_and_explicitly_holds_non_grasp_goal():
    role = parse_bddl_task_role(_bddl("(And (In obj_1 target_1_contain_region))"), suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"])
    assert role.status == "PASS"
    assert role.manipulated_objects == ("obj_1",)
    assert role.target_names == ("target_1_contain_region",)

    non_grasp = parse_bddl_task_role(_bddl("(And (Open cabinet_1_middle_region))"), suite="libero_goal", task_idx=0, object_names=["obj_1", "target_1"])
    assert non_grasp.status == "NO_MANIPULATION_TARGET"


def test_physics_teacher_uses_contiguous_candidate_segments_and_tier3_dwell():
    role = parse_bddl_task_role(_bddl("(And (In obj_1 target_1_contain_region))"), suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"])
    steps, sidecars = _episode()
    rows, windows = derive_episode_rows(
        steps,
        sidecars,
        role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}, "target_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        PROTOCOL,
    )
    assert len(windows) == 1
    assert windows[0]["start_step"] == 0
    assert windows[0]["end_step"] == 11
    assert windows[0]["step_count"] == 12
    assert max(row["utility_tier"] for row in rows) == 3
    assert all(row["window_id"] == "candidate:0" for row in rows)


def test_unknown_or_non_applicable_role_never_produces_rankable_window():
    role = parse_bddl_task_role(_bddl("(And (Open cabinet_1_middle_region))"), suite="libero_goal", task_idx=0, object_names=["obj_1", "target_1"])
    steps, sidecars = _episode()
    rows, windows = derive_episode_rows(
        steps,
        sidecars,
        role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        PROTOCOL,
    )
    assert all(row["known_mask"] is False for row in rows)
    assert windows[0]["rankable"] is False


def test_physics_v21_does_not_treat_zero_motion_or_unknown_target_as_positive_evidence():
    protocol = deepcopy(PROTOCOL)
    protocol["schema"] = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21"
    protocol["window_policy"] = {"loader_preserve_candidate_segment": True}
    role = parse_bddl_task_role(_bddl("(And (In obj_1 unknown_target))"), suite="libero_object", task_idx=0, object_names=["obj_1", "unknown_target"])
    steps, sidecars = _episode(12)
    for sidecar in sidecars:
        sidecar["object_state"][:3] = [0.0, 0.0, 0.0]
        sidecar["robot0_eef_pos"] = [0.0, 0.0, 0.0]
    rows, _ = derive_episode_rows(
        steps,
        sidecars,
        role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        protocol,
    )
    assert all(row["object_eef_comotion_score"] < 1.0 for row in rows)
    assert all(row["target_progress_known"] is False for row in rows)
    assert all(row["target_progress"] == 0.0 for row in rows)
    assert all("target_progress" in row["component_valid_mask"] for row in rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Gate D2.1.1: Action contract tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_action_contract_raw_close():
    from gripper_attack.action_contract import classify_openvla_raw_gripper, GripperIntent
    assert classify_openvla_raw_gripper(0.0) == GripperIntent.CLOSE
    assert classify_openvla_raw_gripper(0.49) == GripperIntent.CLOSE
    assert classify_openvla_raw_gripper(0.9961) == GripperIntent.OPEN
    assert classify_openvla_raw_gripper(1.0) == GripperIntent.OPEN
    assert classify_openvla_raw_gripper(0.5) == GripperIntent.BOUNDARY


def test_action_contract_env_close():
    from gripper_attack.action_contract import classify_libero_env_gripper, GripperIntent
    assert classify_libero_env_gripper(1.0) == GripperIntent.CLOSE
    assert classify_libero_env_gripper(-1.0) == GripperIntent.OPEN
    assert classify_libero_env_gripper(0.0) == GripperIntent.BOUNDARY


def test_action_contract_postprocess_parity():
    from gripper_attack.action_contract import (
        postprocess_gripper_openvla_to_libero, raw_gripper_is_close, env_gripper_is_close)
    for raw in [0.0, 0.3, 0.49, 0.51, 0.7, 1.0]:
        env = postprocess_gripper_openvla_to_libero(raw)
        assert raw_gripper_is_close(raw) == env_gripper_is_close(env)


def test_action_contract_validate_missing_field():
    import pytest
    from gripper_attack.action_contract import validate_raw_action
    with pytest.raises(KeyError):
        validate_raw_action({"clean_action_raw_7d": [0, 0, 0, 0, 0, 0, 0.0]}, field="nonexistent")
    with pytest.raises(KeyError):
        validate_raw_action({}, field="clean_action_raw_7d")


def test_action_contract_validate_non_finite():
    import pytest
    from gripper_attack.action_contract import validate_raw_action
    with pytest.raises(ValueError):
        validate_raw_action({"clean_action_raw_7d": [0, 0, 0, 0, 0, 0, float('nan')]})


def test_action_contract_validate_too_short():
    import pytest
    from gripper_attack.action_contract import validate_raw_action
    with pytest.raises(KeyError):
        validate_raw_action({"clean_action_raw_7d": [0, 0, 0, 0, 0]})


def test_candidate_close_boundary_returns_false():
    """raw=0.5 must return False (not close), not default to True."""
    protocol = deepcopy(PROTOCOL)
    protocol["schema"] = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21"
    protocol["window_policy"] = {"loader_preserve_candidate_segment": True}
    role = parse_bddl_task_role(_bddl("(And (In obj_1 target_1_contain_region))"), suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"])
    steps, sidecars = _episode(12)
    # Set raw to boundary value
    for s in steps:
        s["clean_action_raw_7d"][6] = 0.5
    rows, _ = derive_episode_rows(
        steps, sidecars, role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        protocol,
    )
    # Boundary raw → candidate_close must be False, not True
    assert all(not row["candidate_close"] for row in rows), "raw=0.5 must not be candidate close"
    # D2.1.2: BOUNDARY must set known_mask=False
    assert all(not row["known_mask"] for row in rows), "BOUNDARY must set known_mask=False"
    assert all(row["phase_name"] == "UNKNOWN" for row in rows), "BOUNDARY must set phase=UNKNOWN"


def test_candidate_close_open_returns_false():
    """raw=1.0 (OPEN) must return False."""
    protocol = deepcopy(PROTOCOL)
    protocol["schema"] = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21"
    protocol["window_policy"] = {"loader_preserve_candidate_segment": True}
    role = parse_bddl_task_role(_bddl("(And (In obj_1 target_1_contain_region))"), suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"])
    steps, sidecars = _episode(12)
    for s in steps:
        s["clean_action_raw_7d"][6] = 1.0  # OPEN
    rows, _ = derive_episode_rows(
        steps, sidecars, role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        protocol,
    )
    assert all(not row["candidate_close"] for row in rows), "raw=1.0 (OPEN) must not be candidate close"


def test_candidate_close_open_close_produces_two_segments():
    """close-open-close must produce two candidate segments."""
    protocol = deepcopy(PROTOCOL)
    protocol["schema"] = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21"
    protocol["window_policy"] = {"loader_preserve_candidate_segment": True}
    role = parse_bddl_task_role(_bddl("(And (In obj_1 target_1_contain_region))"), suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"])
    steps, sidecars = _episode(12)
    # close (0-3), open (4-7), close (8-11)
    for i, s in enumerate(steps):
        if i < 4:
            s["clean_action_raw_7d"][6] = 0.0  # CLOSE
        elif i < 8:
            s["clean_action_raw_7d"][6] = 1.0  # OPEN
        else:
            s["clean_action_raw_7d"][6] = 0.0  # CLOSE
    rows, _ = derive_episode_rows(
        steps, sidecars, role,
        {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}},
        protocol,
    )
    # window_id is a string like "candidate:0"
    segment_ids = set(row["window_id"] for row in rows if row.get("window_id", "").startswith("candidate"))
    assert len(segment_ids) == 2, "Expected 2 candidate segments (close-open-close), got {}: {}".format(
        len(segment_ids), sorted(segment_ids))
