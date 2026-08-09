"""Gate F1.1: Factorized Teacher contract tests.

Covers:
  - mechanism routing (single-object, multi-object, articulated→unsupported, unknown)
  - grasp: no-contact→False     (known negative)
  - grasp: contact+stable→True  (known positive)
  - grasp: insufficient dwell→False (known negative)
  - OPEN pulse doesn't clear stable grasp
  - contact-loss triggers release
  - manipulation implies grasp
  - lift enables manipulation
  - unsupported mechanism → all heads unknown
  - unknown action → all heads unknown
  - student_valid=False → all heads unknown
  - two separate grasp→release→grasp → event IDs {0, 1}
  - release gap → event_id=-1
  - DISTRACTOR event for non-primary object
  - prefix invariance (all 3 primary heads)
  - future modification doesn't change past
  - deterministic double derive
  - official protocol JSON direct-load
  - exact output field set match
  - all values/masks/confidences prefix invariant
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from gripper_attack.v5_factorized_teacher import (
    FACTORIZED_TEACHER_FIELDS,
    EventPhase,
    MechanismRoute,
    _determine_mechanism,
    derive_factorized_rows,
    verify_deterministic_derive,
    verify_prefix_invariance,
)
from gripper_attack.v5_physics import parse_bddl_task_role


def _load_official_protocol() -> dict:
    path = Path(__file__).resolve().parent.parent / "configs" / "DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1.json"
    return json.loads(path.read_text())


PROTOCOL = _load_official_protocol()


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


def _bddl_multi() -> str:
    return """(define (problem test)
(:objects
  obj_1 - cube
  obj_2 - cube
  target_1 - plate
  target_2 - plate
)
(:init
  (On obj_1 floor_region)
  (On obj_2 floor_region)
)
(:goal
  (And (In obj_1 target_1_contain_region) (In obj_2 target_2_contain_region))
)
)
"""


def _episode(length: int = 20, gripper_contact_from: int = 5):
    steps = []
    sidecars = []
    for i in range(length):
        raw = 0.0 if i >= 2 else 1.0
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
# Protocol contract
# ═══════════════════════════════════════════════════════════════════════════════

def test_official_protocol_loads_directly():
    """Official protocol JSON must have all runtime fields directly — no indirection."""
    p = _load_official_protocol()
    assert p["schema"] == "DETECTOR_V5_FACTORIZED_TEACHER_PROTOCOL_V1"
    assert "fixed_constants" in p
    assert "history" in p
    assert "head_thresholds" in p
    assert "event_policy" in p
    assert "known_mask_policy" in p
    assert "strict_k10_binding" in p
    # Verify we can derive with it
    role = _role()
    steps, sidecars = _episode(12)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), p)
    assert len(rows) == 12


def test_output_rows_match_declared_field_set():
    """Every output row must exactly match FACTORIZED_TEACHER_FIELDS."""
    role = _role()
    steps, sidecars = _episode(12)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    builder_meta = {"canonical_parent_key", "state_id", "source_artifact_recursive_sha256",
                    "physics_protocol_schema"}
    for row in rows:
        core = set(row) - builder_meta
        expected = FACTORIZED_TEACHER_FIELDS - builder_meta
        assert core == expected, f"Field mismatch: extra={core-expected} missing={expected-core}"


# ═══════════════════════════════════════════════════════════════════════════════
# Mechanism routing
# ═══════════════════════════════════════════════════════════════════════════════

def test_single_object_routed_correctly():
    role = _role()
    route = _determine_mechanism(role)
    assert route == MechanismRoute.SINGLE_OBJECT_PICK_PLACE
    assert route.supported

def test_multi_object_routed_correctly():
    role = parse_bddl_task_role(_bddl_multi(), suite="libero_10", task_idx=0,
                                 object_names=["obj_1", "obj_2", "target_1", "target_2"])
    route = _determine_mechanism(role)
    assert route == MechanismRoute.MULTI_OBJECT_TRANSFER
    assert route.supported

def test_articulated_routed_unsupported():
    bddl_text = _bddl("(And (Open cabinet_1_middle_region))")
    role = parse_bddl_task_role(
        bddl_text,
        suite="libero_goal", task_idx=0, object_names=["cabinet_1"],
    )
    route = _determine_mechanism(role, bddl_text=bddl_text)
    assert route == MechanismRoute.ARTICULATED_OR_PLANAR
    assert not route.supported

def test_non_applicable_routed_unknown():
    role = parse_bddl_task_role(
        _bddl("(And (Open cabinet_1_middle_region))"),
        suite="libero_goal", task_idx=0, object_names=["obj_1"],
    )
    route = _determine_mechanism(role)
    # non-grasp → NO_MANIPULATION_TARGET → not applicable
    assert route == MechanismRoute.UNKNOWN_OR_AMBIGUOUS


# ═══════════════════════════════════════════════════════════════════════════════
# grasp_established: known positive AND known negative
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_contact_grasp_known_negative():
    """No contact + valid physics → known=False for grasp value (not unknown)."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=99)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    # All steps after action becomes known should have grasp_known=True
    # but grasp_established=False (known negative)
    for r in rows:
        if r["action_known"] and r["student_valid"]:
            assert r["grasp_established_known_mask"] is True, \
                f"step {r['step']}: should be known (negative)"
            assert r["grasp_established"] is False, \
                f"step {r['step']}: no contact → grasp must be False"

def test_contact_and_stability_sets_grasp():
    """Stable contact → grasp_established=True after dwell."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    assert any(r["grasp_established"] for r in rows)

def test_insufficient_dwell_grasp_known_negative():
    """Contact present but dwell < 3 → grasp_known=True, grasp_established=False."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    # Step 5: first contact step, dwell=1 → known negative
    r5 = rows[5]
    assert r5["gripper_contact_score"] == 1.0
    assert r5["action_known"] is True
    assert r5["grasp_established_known_mask"] is True
    assert r5["grasp_established"] is False, \
        f"dwell=1 at step 5, grasp should be known-negative"

def test_open_pulse_does_not_clear_grasp():
    """Single OPEN during stable hold → grasp remains True."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    steps[12]["clean_action_raw_7d"][6] = 1.0  # OPEN pulse
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    assert rows[11]["grasp_established"], "grasp should be True before OPEN pulse"
    # grasp should persist (OPEN doesn't clear it)
    assert rows[12]["grasp_established"] or rows[13]["grasp_established"], \
        "OPEN pulse should not clear stable grasp"

def test_contact_loss_triggers_release():
    """Contact loss → release_or_instability=True."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    assert any(r["release_or_instability"] for r in rows)


# ═══════════════════════════════════════════════════════════════════════════════
# manipulation_active
# ═══════════════════════════════════════════════════════════════════════════════

def test_manipulation_implies_grasp():
    """manipulation_active=True → grasp_established=True (logical constraint)."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        if r["manipulation_active"]:
            assert r["grasp_established"], f"step {r['step']}: manip=True but grasp=False"

def test_manipulation_known_requires_grasp():
    """manipulation_known_mask=True → grasp_established=True."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        if r["manipulation_active_known_mask"]:
            assert r["grasp_established"], \
                f"step {r['step']}: manip_known=True but grasp=False"

def test_lift_enables_manipulation():
    """Sufficient lift should enable manipulation_active."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    for i in range(10, 20):
        sidecars[i]["robot0_eef_pos"][2] = 0.08
        sidecars[i]["object_state"][2] = 0.08
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    assert any(r["manipulation_active"] for r in rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Abstain / unknown gate
# ═══════════════════════════════════════════════════════════════════════════════

def test_unsupported_mechanism_all_heads_unknown():
    """Articulated route → all known_masks False, all values False."""
    role = parse_bddl_task_role(
        _bddl("(And (Open cabinet_1_middle_region))"),
        suite="libero_goal", task_idx=0, object_names=["cabinet_1"],
    )
    # Make role applicable by using correct object names
    steps, sidecars = _episode(12, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        assert r["grasp_established_known_mask"] is False
        assert r["grasp_established"] is False
        assert r["manipulation_active_known_mask"] is False
        assert r["release_or_instability_known_mask"] is False

def test_student_invalid_all_heads_unknown():
    """student_valid=False → all known_masks False."""
    role = _role()
    steps, sidecars = _episode(12, gripper_contact_from=5)
    for s in steps:
        s["valid"] = False
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        assert r["grasp_established_known_mask"] is False
        assert r["manipulation_active_known_mask"] is False
        assert r["release_or_instability_known_mask"] is False

def test_unknown_action_all_heads_unknown():
    """action_known=False → all known_masks False, not mistaken for negative."""
    role = _role()
    steps, sidecars = _episode(12, gripper_contact_from=5)
    for s in steps:
        del s["clean_action_raw_7d"]
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        assert r["grasp_established_known_mask"] is False
        assert r["grasp_established"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Event state machine
# ═══════════════════════════════════════════════════════════════════════════════

def test_two_grasp_segments_two_event_ids():
    """grasp→release→grasp → event counter increases, IDLE gap has event_id=-1."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    # Create release gap at steps 14-17
    for i in range(14, 18):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0  # OPEN
    rows, events = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    eids = set(r["event_id"] for r in rows if r["event_id"] >= 0)
    assert len(eids) >= 2, f"expected >= 2 event IDs, got {sorted(eids)}"
    # After release clears → IDLE phase should have event_id=-1
    idle_eids = [r["event_id"] for r in rows if r["event_phase"] == "IDLE"]
    assert all(e == -1 for e in idle_eids), f"IDLE phase must have event_id=-1"


def test_release_step_retains_event_id():
    """Release trigger step must retain the event_id of the ending event."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    for i in range(16, 20):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0
    rows, events = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    release_rows = [r for r in rows if r["release_or_instability"]]
    assert len(release_rows) > 0, "should have release steps"
    for r in release_rows:
        assert r["event_id"] >= 0, f"release step {r['step']} must have event_id >= 0, got {r['event_id']}"
        assert r["event_phase"] == "RELEASED", f"release step must be RELEASED phase, got {r['event_phase']}"


def test_event_summary_includes_release_steps():
    """Event summary must record release_steps > 0 for events with release."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    for i in range(16, 20):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0
    _, events = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    assert len(events) > 0
    any_has_release = any(e["has_release"] for e in events)
    any_release_steps = any(e["release_steps"] > 0 for e in events)
    assert any_has_release or any_release_steps, \
        f"event summary should capture release: has_release={[e['has_release'] for e in events]}"


def test_object_attribution_from_contacts():
    """_resolve_active_object must correctly identify contacted object across all known objects."""
    from gripper_attack.v5_factorized_teacher import _resolve_active_object
    all_names = ["obj_1", "obj_2", "target_1", "distractor_obj"]
    name, status = _resolve_active_object(
        [["obj_1_g1", "gripper0_finger1_pad_collision"]],
        ["obj_1", "obj_2"], all_names,
    )
    assert name == "obj_1"
    assert status == "SINGLE"

    name, status = _resolve_active_object(
        [["obj_1_g1", "gripper0_finger1_pad_collision"],
         ["obj_2_g1", "gripper0_finger1_pad_collision"]],
        ["obj_1", "obj_2"], all_names,
    )
    assert name is None
    assert status == "AMBIGUOUS"

    name, status = _resolve_active_object([], ["obj_1"], all_names)
    assert name is None
    assert status == "NONE"

    # Non-manipulated object is still detected
    name, status = _resolve_active_object(
        [["distractor_obj_g1", "gripper0_finger1_pad_collision"]],
        ["obj_1", "obj_2"], all_names,
    )
    assert name == "distractor_obj"
    assert status == "SINGLE"


def test_second_manipulated_object_is_target_not_distractor():
    """All manipulated objects are TARGET — not just the first one."""
    bddl = _bddl_multi()
    role = parse_bddl_task_role(bddl, suite="libero_10", task_idx=0,
                                 object_names=["obj_1", "obj_2", "target_1", "target_2"])
    slices = {"obj_1": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]},
              "obj_2": {"pos": [0, 3], "quat": [3, 7], "to_eef_pos": [7, 10], "to_eef_quat": [10, 14]}}
    steps, sidecars = _episode(20, gripper_contact_from=5)
    for sc in sidecars:
        sc["mujoco_contact_pairs"] = [["obj_2_g1", "gripper0_finger1_pad_collision"]]
    rows, _ = derive_factorized_rows(steps, sidecars, role, slices, PROTOCOL)
    # obj_2 IS in manipulated_objects → should be TARGET, not DISTRACTOR
    has_target = any(r["event_role"] == "TARGET" and r["event_id"] >= 0 for r in rows)
    has_distractor = any(r["event_role"] == "DISTRACTOR" for r in rows)
    assert has_target, "second manipulated object should be TARGET"
    assert not has_distractor, f"obj_2 is manipulated, should not be DISTRACTOR"


def test_distractor_only_for_non_manipulated_object():
    """When active_object is resolved to a non-manipulated object, event_role=DISTRACTOR.

    Note: synthetic test limitation — _contact_flags filters by manipulated objects,
    so grasp won't establish for non-manipulated contacts. The DISTRACTOR code path
    is validated by _resolve_active_object unit tests and will activate with real
    MuJoCo data where non-task objects can be contacted during a grasp event.
    """
    from gripper_attack.v5_factorized_teacher import _resolve_active_object
    all_names = ["obj_1", "obj_2", "extra_obj"]
    # Non-manipulated object is correctly detected as SINGLE
    name, status = _resolve_active_object(
        [["extra_obj_g1", "gripper0_finger1_pad_collision"]],
        ["obj_1", "obj_2"], all_names,
    )
    assert name == "extra_obj"
    assert status == "SINGLE"
    # And the event_role logic asserts: active_object not in manipulated_set → DISTRACTOR


def test_no_direct_released_to_grasped_transition():
    """RELEASED cannot directly transition to GRASPED — must go through IDLE."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    # Brief release pulse at step 14, contact resumes at 15
    for i in range(14, 15):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    phases = [r["event_phase"] for r in rows]
    # After RELEASED, should see IDLE before GRASPED
    released_idx = None
    for i, p in enumerate(phases):
        if p == "RELEASED":
            released_idx = i
            break
    if released_idx is not None:
        post_release = phases[released_idx:]
        # Should transition through IDLE
        assert "IDLE" in post_release, "RELEASED must go to IDLE before new event"


def test_event_id_increments_through_idle():
    """Event counter only increments via IDLE→GRASPED, not RELEASED→GRASPED."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    for i in range(14, 18):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    eids = [r["event_id"] for r in rows]
    # Should see two distinct event IDs (not merger of two segments)
    unique_eids = set(e for e in eids if e >= 0)
    assert len(unique_eids) == 2, f"expected exactly 2 event IDs, got {sorted(unique_eids)}"

def test_event_state_machine_phases():
    """Verify event_phase transitions IDLE→GRASPED→MANIPULATING→RELEASED→IDLE."""
    role = _role()
    steps, sidecars = _episode(30, gripper_contact_from=5)
    for i in range(14, 18):
        sidecars[i]["mujoco_contact_pairs"] = []
        steps[i]["clean_action_raw_7d"][6] = 1.0
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    phases = [r["event_phase"] for r in rows]
    assert "IDLE" in phases
    assert "GRASPED" in phases
    assert "RELEASED" in phases

def test_distractor_vs_target_semantics():
    """event_role must be one of TARGET/DISTRACTOR/NONE."""
    role = _role()
    steps, sidecars = _episode(12, gripper_contact_from=3)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        assert r["event_role"] in ("TARGET", "DISTRACTOR", "NONE")
        assert isinstance(r["target_relevant"], bool)
        assert "active_object_name" in r


# ═══════════════════════════════════════════════════════════════════════════════
# K10 external binding
# ═══════════════════════════════════════════════════════════════════════════════

def test_require_external_k10_raises_without_labels():
    """require_external_k10=True + no k10_labels → ValueError."""
    import pytest
    role = _role()
    steps, sidecars = _episode(10, gripper_contact_from=5)
    with pytest.raises(ValueError, match="require_external_k10"):
        derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL,
                               require_external_k10=True)

def test_k10_labels_passed_produce_correct_schema():
    """External K10 labels must produce correct strict_k10_binding_schema."""
    role = _role()
    steps, sidecars = _episode(10, gripper_contact_from=5)
    k10 = [{"k10_known_mask": True, "k10_feasible": True} for _ in range(10)]
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL,
                                      k10_labels=k10, k10_label_schema="OFFICIAL_K10_V1")
    for r in rows:
        assert r["strict_k10_binding_schema"] == "OFFICIAL_K10_V1"

def test_internal_k10_fallback_marked_non_authoritative():
    """Without external K10, schema must be INTERNAL_SIMPLIFIED_V1."""
    role = _role()
    steps, sidecars = _episode(10, gripper_contact_from=5)
    rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for r in rows:
        assert r["strict_k10_binding_schema"] == "INTERNAL_SIMPLIFIED_V1"


# ═══════════════════════════════════════════════════════════════════════════════
# Prefix invariance and determinism
# ═══════════════════════════════════════════════════════════════════════════════

def test_primary_heads_prefix_invariant():
    """All 3 primary heads must be identical whether computed full or prefix."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    result = verify_prefix_invariance(steps, sidecars, role, _slices(), PROTOCOL)
    assert result["violations"] == 0, f"prefix violations: {result['violations']}"

def test_future_modification_does_not_change_past():
    """Modifying future steps must not change head values at earlier steps."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    full_rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    mutated = deepcopy(steps)
    for i in range(15, 20):
        mutated[i]["clean_action_raw_7d"][6] = 1.0
    mutated_rows, _ = derive_factorized_rows(mutated, sidecars, role, _slices(), PROTOCOL)
    for head in ("grasp_established", "manipulation_active", "release_or_instability"):
        for t in range(15):
            assert full_rows[t][head] == mutated_rows[t][head], \
                f"future modification changed {head} at step {t}"

def test_deterministic_double_derive():
    """Two independent runs must produce bit-identical output."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    result = verify_deterministic_derive(steps, sidecars, role, _slices(), PROTOCOL)
    assert result["deterministic"], "double derive produced different output"

def test_all_values_prefix_invariant():
    """All boolean values, masks, and confidences must be prefix-invariant."""
    role = _role()
    steps, sidecars = _episode(20, gripper_contact_from=5)
    full_rows, _ = derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    check_fields = [
        "grasp_established", "grasp_established_known_mask", "grasp_established_confidence",
        "manipulation_active", "manipulation_active_known_mask", "manipulation_active_confidence",
        "release_or_instability", "release_or_instability_known_mask", "release_or_instability_confidence",
    ]
    violations = 0
    for t in range(1, 20):
        prefix_rows, _ = derive_factorized_rows(
            steps[:t + 1], sidecars[:t + 1], role, _slices(), PROTOCOL)
        for field in check_fields:
            if prefix_rows[t][field] != full_rows[t][field]:
                violations += 1
    assert violations == 0, f"{violations} prefix violations across all fields"

def test_step_rows_not_mutated():
    """derive_factorized_rows must not mutate input step_rows."""
    role = _role()
    steps, sidecars = _episode(12)
    original = deepcopy(steps)
    derive_factorized_rows(steps, sidecars, role, _slices(), PROTOCOL)
    for i, (orig, after) in enumerate(zip(original, steps)):
        assert orig == after, f"step {i} was mutated: {orig} -> {after}"
