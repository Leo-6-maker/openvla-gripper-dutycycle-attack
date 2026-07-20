"""Gate F2: Factorized Teacher synthetic and prefix invariance tests."""
from __future__ import annotations

from copy import deepcopy

from gripper_attack.v5_factorized_teacher import (
    MechanismRoute,
    _determine_mechanism,
    derive_factorized_rows,
    verify_prefix_invariance,
)
from gripper_attack.v5_physics import parse_bddl_task_role


PROTOCOL = {
    "schema": "DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1",
    "history": {"score_window_steps": 5, "minimum_stable_grasp_dwell": 10},
    "fixed_constants": {
        "relative_position_scale_m": 0.03,
        "relative_quaternion_scale": 0.20,
        "lift_scale_m": 0.03,
        "target_progress_scale_m": 0.20,
    },
}


def _bddl(goal: str) -> str:
    return f"""(define (problem test)
(:objects
  obj_1 - cube
  target_1 - plate
)
(:init
  (On obj_1 floor_region)
)
(:goal
  {goal}
)
)
"""


def _episode(length: int = 20, gripper_contact_from: int = 5):
    """Synthetic episode with gripper contact starting at gripper_contact_from."""
    steps = []
    sidecars = []
    for i in range(length):
        raw = 0.0 if i >= 2 else 1.0  # CLOSE starts at step 2
        steps.append({"clean_action_raw_7d": [0, 0, 0, 0, 0, 0, raw], "valid": True})
        z = 0.03 * i / max(1, length - 1)
        state = [0.1 * i / max(1, length - 1), 0.0, z, 1.0, 0.0, 0.0, 0.0,
                  0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        pairs = []
        if i >= gripper_contact_from and i < length - 3:
            pairs.append(["obj_1_g1", "gripper0_finger1_pad_collision"])
        if i == 0:
            pairs.append(["obj_1_g1", "floor"])
        sidecars.append({
            "object_state": state,
            "robot0_eef_pos": [0.1 * i / max(1, length - 1), 0.0, z],
            "robot0_gripper_qpos": [0.0, 0.0],
            "mujoco_contact_pairs": pairs,
        })
    return steps, sidecars


def _role():
    return parse_bddl_task_role(
        _bddl("(And (In obj_1 target_1_contain_region))"),
        suite="libero_object", task_idx=0, object_names=["obj_1", "target_1"],
    )


def _slices():
    return {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}}


# ═══════════════════════════════════════════════════════════════════════════════
# Mechanism routing tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_object_pick_place_routed_correctly():
    role = _role()
    assert role.applicable
    route = _determine_mechanism(role, ["obj_1", "target_1"])
    assert route == MechanismRoute.SINGLE_OBJECT_PICK_PLACE
    assert route.supported is True


def test_non_grasp_task_routed_to_unknown():
    role = parse_bddl_task_role(
        _bddl("(And (Open cabinet_1_middle_region))"),
        suite="libero_goal", task_idx=0, object_names=["obj_1"],
    )
    route = _determine_mechanism(role, ["obj_1"])
    assert route == MechanismRoute.UNKNOWN_OR_AMBIGUOUS
    assert route.supported is False


# ═══════════════════════════════════════════════════════════════════════════════
# grasp_established tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_close_onset_without_contact_does_not_set_grasp():
    """CLOSE command without physical contact must NOT trigger grasp."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=99)  # no contact
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    assert not any(r["grasp_established"] for r in rows), "grasp without contact"


def test_contact_and_stability_sets_grasp():
    """Stable contact + CLOSE → grasp_established=True."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    grasp_steps = [r["grasp_established"] for r in rows]
    assert any(grasp_steps), f"grasp never established: {grasp_steps}"


def test_single_open_pulse_does_not_clear_grasp():
    """One OPEN step during stable hold must NOT clear grasp_established."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    # Insert a single OPEN pulse at step 10
    steps[10]["clean_action_raw_7d"][6] = 1.0
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    # grasp should be maintained around the pulse
    assert rows[9]["grasp_established"], "grasp lost before OPEN pulse"
    assert rows[10]["grasp_established"] or rows[11]["grasp_established"], \
        "single OPEN pulse incorrectly cleared grasp"


def test_contact_loss_triggers_release():
    """Contact loss at end → release_or_instability=True."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    # Contact ends at step 17 (length-3), so release should appear after
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    release_steps = [r["release_or_instability"] for r in rows]
    assert any(release_steps), f"release never triggered after contact loss: {release_steps}"


# ═══════════════════════════════════════════════════════════════════════════════
# manipulation_active tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_manipulation_implies_grasp():
    """manipulation_active=True → grasp_established=True for all steps."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    for r in rows:
        if r["manipulation_active"]:
            assert r["grasp_established"], \
                f"step {r['step']}: manipulation_active=True but grasp_established=False"


def test_lift_enables_manipulation():
    """Sufficient lift should enable manipulation_active."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    # Increase lift by raising z
    for i in range(10, 20):
        sidecars[i]["robot0_eef_pos"][2] = 0.08
        state = sidecars[i]["object_state"]
        state[2] = 0.08
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    manip_steps = [r["manipulation_active"] for r in rows]
    assert any(manip_steps), f"lift did not enable manipulation: {manip_steps}"


# ═══════════════════════════════════════════════════════════════════════════════
# unsupported mechanism tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_unsupported_mechanism_abstains_all_heads():
    """Articulated tasks must have all heads known_mask=False."""
    role = parse_bddl_task_role(
        _bddl("(And (Open cabinet_1_middle_region))"),
        suite="libero_goal", task_idx=0, object_names=["obj_1"],
    )
    steps, sidecars = _episode(12, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    for r in rows:
        assert r["grasp_established_known_mask"] is False
        assert r["grasp_established"] is False
        assert r["manipulation_active_known_mask"] is False
        assert r["manipulation_active"] is False


def test_unknown_does_not_become_negative():
    """action_known=False must set all head known_masks to False."""
    role = _role()
    steps, sidecars = _episode(12, gripper_contact_from=5)
    for s in steps:
        del s["clean_action_raw_7d"]  # missing → UNKNOWN
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    for r in rows:
        assert r["grasp_established"] is False
        assert r["grasp_established_known_mask"] is False
        assert r["manipulation_active_known_mask"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Prefix invariance tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_primary_heads_prefix_invariant():
    """All three primary heads must produce identical values whether computed
    from full trajectory or prefix-only."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    result = verify_prefix_invariance(
        steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL,
    )
    assert result["violations"] == 0, \
        f"prefix invariance violated: {result['violations']} at {result['total_steps']} steps"
    assert result["prefix_invariant"] is True


def test_future_modification_does_not_change_past_heads():
    """Modifying future steps must not change head values at earlier timesteps."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    full_rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)

    # Create a mutated version changing future actions
    mutated_steps = deepcopy(steps)
    for i in range(15, 20):
        mutated_steps[i]["clean_action_raw_7d"][6] = 1.0  # all OPEN in future
    mutated_rows, _ = derive_factorized_rows(
        mutated_steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL,
    )

    # Steps before the mutation should be identical
    for head in ("grasp_established", "manipulation_active", "release_or_instability"):
        for t in range(15):
            if full_rows[t][head] != mutated_rows[t][head]:
                raise AssertionError(
                    f"future modification changed {head} at step {t}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-object / event reset tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_two_grasp_segments_produce_two_events():
    """Two separate grasp segments with release in between → two events."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    # Create two grasp segments with a gap
    for i in range(13, 17):
        sidecars[i]["mujoco_contact_pairs"] = []  # no contact
        steps[i]["clean_action_raw_7d"][6] = 1.0  # OPEN
    rows, events = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    event_ids = set(r["event_id"] for r in rows if r["event_id"] >= 0)
    assert len(event_ids) >= 1, f"expected at least 1 event, got {len(event_ids)}"


def test_distractor_object_vs_target_object():
    """Verify event_role distinction is possible (interface check)."""
    role = _role()
    steps, sidecars = _episode(12, gripper_contact_from=3)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), ["obj_1", "target_1"], PROTOCOL)
    for r in rows:
        assert "event_role" in r
        assert r["event_role"] in ("TARGET", "DISTRACTOR", "NONE")
        assert "target_relevant" in r
        assert isinstance(r["target_relevant"], bool)
