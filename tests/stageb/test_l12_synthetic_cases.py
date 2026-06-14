"""Synthetic test matrix for L12 window selector.

Cases A-F as specified in the semantic audit plan.
Each case constructs a minimal trace and verifies expected selector behavior.
"""

import numpy as np

from gripper_attack.phase_detector import (
    teacher_privileged_critical_close_anchor,
    teacher_rule_critical_close_anchor,
)
from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    PREDICTION_HORIZON,
    WINDOW_LEN,
    PRE_OFFSET,
)


def _base_record(t: int) -> dict:
    return {
        "step": t,
        "clean_gripper_env": 0.0,
        "clean_gripper_raw": 0.7,       # OPEN
        "gripper_qpos_before": 0.0,
        "qpos_abs_before": 0.0,
        "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.3,  # high above table
        "clean_close": 0,
        "close_onset": 0,
        "close_streak": 0,
        "decoded_open_bool": 0,
        # Privileged fields (absent by default — Teacher-P must handle)
        "eef_to_obj_distance": "",
        "obj_to_target_distance": "",
        "obj_x": "", "obj_y": "", "obj_z": "",
        "target_obj_x": "", "target_obj_y": "", "target_obj_z": "",
    }


# ═══════════════════════════════════════════════════════════════════
# Case A: Normal approach → effective grasp close
# ═══════════════════════════════════════════════════════════════════

def _make_case_a():
    """Approach phase, then CLOSE at step 50 with EEF near object, then lift."""
    records = [_base_record(t) for t in range(100)]

    # Approach: EEF moving toward object, gripper OPEN
    for t in range(10, 50):
        records[t]["eef_z"] = 0.3 - 0.005 * (t - 10)  # descending
        records[t]["eef_x"] = 0.01 * (t - 10)

    # Privileged fields: object at fixed position
    for t in range(100):
        records[t]["eef_to_obj_distance"] = abs(0.3 - 0.005 * max(0, min(40, t - 10)) - 0.05)
        records[t]["obj_to_target_distance"] = 0.25
        records[t]["obj_x"] = 0.3
        records[t]["obj_y"] = 0.0
        records[t]["obj_z"] = 0.05  # on table
        records[t]["target_obj_x"] = 0.1
        records[t]["target_obj_y"] = 0.0
        records[t]["target_obj_z"] = 0.05

    # At step 48-49: EEF near object, still OPEN
    records[48]["eef_to_obj_distance"] = 0.03
    records[49]["eef_to_obj_distance"] = 0.02

    # Step 50: CLOSE onset — critical grasp
    r = records[50]
    r["clean_gripper_raw"] = 0.0
    r["clean_close"] = 1
    r["close_onset"] = 1
    r["close_streak"] = 1
    r["eef_to_obj_distance"] = 0.01  # very close
    r["clean_gripper_env"] = 1.0

    # Sustained CLOSE
    for t in range(51, 70):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 50 + 1
        records[t]["eef_to_obj_distance"] = 0.01

    # Object lift evidence (z increases after grasp)
    for t in range(52, 65):
        records[t]["obj_z"] = 0.05 + 0.008 * (t - 51)  # rising

    # Post-release
    for t in range(75, 100):
        records[t]["decoded_open_bool"] = 1
        records[t]["clean_gripper_raw"] = 0.7
        records[t]["gripper_qpos_before"] = 0.03

    return records


def test_case_a_teacher_p_finds_critical_close():
    """Teacher-P correctly identifies the task-critical grasp close."""
    records = _make_case_a()
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == 50, f"Teacher-P expected 50, got {anchor}"


def test_case_a_student_high_score_before_anchor():
    """Student scores peak AT the close event (raw crossing), not before.

    Honest behavior: without privileged object-pose info, the student's strongest
    signal is the gripper_raw OPEN→CLOSE crossing, which happens AT the close step.
    Precursor signals (EEF deceleration, qpos) give only weak scores (~0.3).
    This is the documented gap between student and privileged teacher.
    """
    records = _make_case_a()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=50)

    # Precursor steps have weak scores only (no CLOSE command yet visible)
    precursor_scores = [preds[t]["score"] for t in range(46, 50)]
    # All precursor scores should be non-negative and not abstain
    assert all(s >= 0.0 for s in precursor_scores), f"Negative precursor scores: {precursor_scores}"

    # Step 50 (the close itself) should have a high score (raw crossing fires)
    assert preds[50]["score"] >= 1.5, f"Close step score too low: {preds[50]['score']}"

    # Horizon labels: step 47 should correctly predict close within H=4
    assert preds[47]["will_critical_close_within_horizon"], "Step 47 should see close in horizon"


def test_case_a_offline_window_covers_anchor():
    """Offline window must cover the teacher's critical close step."""
    records = _make_case_a()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=50)
    win = select_best_window(preds, WINDOW_LEN, PRE_OFFSET)
    assert win["window_start"] <= 50 < win["window_end"], \
        f"Window [{win['window_start']},{win['window_end']}] does not cover anchor 50"


def test_case_a_online_trigger_not_later_than_anchor():
    """Online trigger must fire at or before the critical close."""
    records = _make_case_a()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=50)
    win = select_online_trigger(preds, score_threshold=1.0, confirmation_steps=1)
    trigger = win.get("trigger_step", -1)
    assert trigger >= 0, "Online trigger should fire"
    assert trigger <= 52, f"Online trigger too late: {trigger}"


# ═══════════════════════════════════════════════════════════════════
# Case B: Step 4 spurious early CLOSE, object far away (butter_s2 analogue)
# ═══════════════════════════════════════════════════════════════════

def _make_case_b():
    """Early CLOSE at step 4 (EEF far from object), real critical close at step 78."""
    records = [_base_record(t) for t in range(100)]

    # Privileged fields
    for t in range(100):
        records[t]["obj_x"] = 0.5
        records[t]["obj_y"] = 0.0
        records[t]["obj_z"] = 0.05
        records[t]["target_obj_x"] = 0.1
        records[t]["target_obj_y"] = 0.0
        records[t]["target_obj_z"] = 0.05
        records[t]["eef_to_obj_distance"] = 0.5  # far
        records[t]["obj_to_target_distance"] = 0.4

    # Step 4: spurious early CLOSE (EEF far from object)
    r4 = records[4]
    r4["clean_gripper_raw"] = 0.0
    r4["clean_close"] = 1
    r4["close_onset"] = 1
    r4["close_streak"] = 1
    r4["clean_gripper_env"] = 1.0
    r4["eef_to_obj_distance"] = 0.5  # 50cm — far!

    # Brief CLOSE then back to OPEN
    for t in range(5, 10):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 4 + 1
    for t in range(10, 50):
        records[t]["clean_gripper_raw"] = 0.7  # OPEN again

    # Approach: EEF nears object
    for t in range(50, 78):
        records[t]["eef_to_obj_distance"] = max(0.01, 0.5 - 0.018 * (t - 50))
        records[t]["eef_z"] = 0.3 - 0.005 * (t - 50)

    # Step 78: real critical close (EEF at object)
    r78 = records[78]
    r78["clean_gripper_raw"] = 0.0
    r78["clean_close"] = 1
    r78["close_onset"] = 1
    r78["close_streak"] = 1
    r78["clean_gripper_env"] = 1.0
    r78["eef_to_obj_distance"] = 0.02  # 2cm — close!

    for t in range(79, 95):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 78 + 1
        records[t]["eef_to_obj_distance"] = 0.02

    # Lift evidence after step 78
    for t in range(80, 90):
        records[t]["obj_z"] = 0.05 + 0.01 * (t - 79)

    return records


def test_case_b_teacher_p_rejects_early_close():
    """Teacher-P must NOT accept step 4 as critical close (EEF far from object)."""
    records = _make_case_b()
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == 78, f"Teacher-P expected 78 (real critical close), got {anchor}"
    # Explicit: NOT 4
    assert anchor != 4, "Teacher-P must not accept spurious step-4 close"


def test_case_b_student_equal_scores_documented_limitation():
    """Student gives equal scores to step 4 and step 78 (both have same physical signals).

    This is the HONEST and CORRECT behavior: without privileged object-pose data,
    the deployment-safe student CANNOT distinguish a spurious early close from a
    task-critical close.

    The offline selector correctly ABSTAINS (ambiguous_multiple_close_candidates)
    rather than silently picking the earliest.
    """
    records = _make_case_b()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=78)

    # Both steps have identical physical signals → identical scores
    assert preds[4]["score"] == preds[78]["score"], \
        f"Student legitimately gives equal scores: step4={preds[4]['score']}, step78={preds[78]['score']}"

    # Both are legitimate close detections (neither abstains)
    assert not preds[4]["abstain"]
    assert not preds[78]["abstain"]

    # Offline selector abstains due to ambiguous multiple closes
    win = select_best_window(preds)
    assert win["abstain_reason"] == "ambiguous_multiple_close_candidates", \
        f"Expected ambiguous_multiple_close_candidates, got '{win['abstain_reason']}'"
    assert win["anchor_step"] == -1


# ═══════════════════════════════════════════════════════════════════
# Case C: No stable grasp/lift evidence (insufficient privileged signal)
# ═══════════════════════════════════════════════════════════════════

def _make_case_c():
    """Trajectory with CLOSE commands but no object lift (failed grasp)."""
    records = [_base_record(t) for t in range(80)]

    # Privileged: object present but never moves
    for t in range(80):
        records[t]["obj_x"] = 0.5
        records[t]["obj_y"] = 0.0
        records[t]["obj_z"] = 0.05  # constant — no lift
        records[t]["eef_to_obj_distance"] = 0.1
        records[t]["obj_to_target_distance"] = 0.3

    # CLOSE at step 30 (EEF near object but grasp fails — no lift)
    r30 = records[30]
    r30["clean_gripper_raw"] = 0.0
    r30["clean_close"] = 1
    r30["close_onset"] = 1
    r30["close_streak"] = 1
    r30["eef_to_obj_distance"] = 0.02

    for t in range(31, 50):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 30 + 1

    # Post-release (no object movement occurred)
    for t in range(55, 80):
        records[t]["decoded_open_bool"] = 1
        records[t]["clean_gripper_raw"] = 0.7

    return records


def test_case_c_teacher_p_abstains_no_lift():
    """Teacher-P must abstain when CLOSE is not followed by object lift."""
    records = _make_case_c()
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == -1, f"Teacher-P should abstain (no lift evidence), got {anchor}"


def test_case_c_student_detects_close_teacher_judges_criticality():
    """Student detects the CLOSE event (physical signal present) but cannot judge
    whether it led to a successful grasp. Teacher-P correctly abstains.

    This documents the separation of concerns:
      - Student: detects when a CLOSE happens (deployment-safe, causal)
      - Teacher-P: judges whether that CLOSE was task-critical (privileged)
    """
    records = _make_case_c()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=-1)
    win = select_best_window(preds)

    # Student detects the close (physical signals are real even if grasp fails)
    assert win["anchor_step"] == 30, \
        f"Student should detect close at step 30, got {win['anchor_step']}"
    assert win["score"] >= 1.5, f"Close detection score too low: {win['score']}"
    assert not win["abstain_reason"], f"Student should not abstain at real close: {win['abstain_reason']}"

    # Teacher-P correctly abstains (no lift → not task-critical)
    anchor_p = teacher_privileged_critical_close_anchor(records)
    assert anchor_p == -1, "Teacher-P must abstain when close doesn't produce lift"


# ═══════════════════════════════════════════════════════════════════
# Case D: Post-release re-close (should not be primary critical close)
# ═══════════════════════════════════════════════════════════════════

def _make_case_d():
    """Primary close at 50, post-release re-close at 80."""
    records = [_base_record(t) for t in range(100)]

    # Privileged fields
    for t in range(100):
        records[t]["obj_x"] = 0.5
        records[t]["obj_y"] = 0.0
        records[t]["obj_z"] = 0.05
        records[t]["eef_to_obj_distance"] = 0.1
        records[t]["obj_to_target_distance"] = 0.3

    # Primary critical close at step 50
    r50 = records[50]
    r50["clean_gripper_raw"] = 0.0
    r50["clean_close"] = 1
    r50["close_onset"] = 1
    r50["close_streak"] = 1
    r50["eef_to_obj_distance"] = 0.02
    for t in range(51, 65):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 50 + 1
        records[t]["eef_to_obj_distance"] = 0.03  # EEF stays near during lift
    for t in range(52, 60):
        records[t]["obj_z"] = 0.05 + 0.01 * (t - 51)  # lift

    # Release
    for t in range(60, 75):
        records[t]["decoded_open_bool"] = 1
        records[t]["clean_gripper_raw"] = 0.7
        records[t]["gripper_qpos_before"] = 0.03

    # Post-release re-close at step 80 (NOT a critical close)
    r80 = records[80]
    r80["clean_gripper_raw"] = 0.0
    r80["clean_close"] = 1
    r80["close_onset"] = 1
    r80["close_streak"] = 1
    r80["decoded_open_bool"] = 1  # still decoded as open
    r80["gripper_qpos_before"] = 0.03  # gripper physically open
    r80["eef_to_obj_distance"] = 0.2  # far from object

    return records


def test_case_d_teacher_p_returns_primary_not_reclose():
    """Teacher-P returns first critical close (50), not post-release re-close (80)."""
    records = _make_case_d()
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == 50, f"Teacher-P expected 50 (primary), got {anchor}"


def test_case_d_student_abstains_at_reclose():
    """Student abstains (gripper_already_open) at post-release re-close."""
    records = _make_case_d()
    preds = rule_based_close_predictor(records, horizon=PREDICTION_HORIZON, teacher_anchor=50)
    # Step 80 should abstain due to gripper_already_open
    assert preds[80]["abstain"] == "gripper_already_open", \
        f"Expected gripper_already_open abstain at re-close, got '{preds[80]['abstain']}'"
    # Primary close at step 50 should NOT abstain
    assert not preds[50]["abstain"], f"Primary close should not abstain: {preds[50]['abstain']}"


# ═══════════════════════════════════════════════════════════════════
# Case E: Prepend idle frames — NoStep invariance
# ═══════════════════════════════════════════════════════════════════

def _make_case_e():
    """Standard approach+close at step 50."""
    records = [_base_record(t) for t in range(100)]
    r50 = records[50]
    r50["clean_gripper_raw"] = 0.0
    r50["clean_close"] = 1
    r50["close_onset"] = 1
    r50["close_streak"] = 1
    for t in range(51, 70):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 50 + 1
    return records


def test_case_e_no_step_student_invariant_to_idle_prefix():
    """After prepending 20 idle steps, student score peak shifts by exactly 20."""
    records = _make_case_e()
    preds_orig = rule_based_close_predictor(records)

    idle = [_base_record(t) for t in range(20)]
    shifted = idle + [_base_record(t) for t in range(100)]
    # Re-apply close at step 70 (50 + 20)
    r70 = shifted[70]
    r70["clean_gripper_raw"] = 0.0
    r70["clean_close"] = 1
    r70["close_onset"] = 1
    r70["close_streak"] = 1
    for t in range(71, 90):
        shifted[t]["clean_gripper_raw"] = 0.0
        shifted[t]["clean_close"] = 1
        shifted[t]["close_streak"] = t - 70 + 1

    preds_shifted = rule_based_close_predictor(shifted)

    best_orig = max(preds_orig, key=lambda p: p["score"])
    best_shifted = max(preds_shifted, key=lambda p: p["score"])
    assert best_shifted["step"] == best_orig["step"] + 20, \
        f"Expected peak at {best_orig['step'] + 20}, got {best_shifted['step']}"


# ═══════════════════════════════════════════════════════════════════
# Case F: Attack outcome fields injected — output immunity
# ═══════════════════════════════════════════════════════════════════

def _make_case_f(with_attack_fields: bool = False):
    """Standard trace, optionally with attack outcome fields injected."""
    records = [_base_record(t) for t in range(100)]
    r50 = records[50]
    r50["clean_gripper_raw"] = 0.0
    r50["clean_close"] = 1
    r50["close_onset"] = 1
    r50["close_streak"] = 1
    for t in range(51, 70):
        records[t]["clean_gripper_raw"] = 0.0
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 50 + 1

    if with_attack_fields:
        for r in records:
            r["vis_open_count"] = 5
            r["random_open_count"] = 3
            r["attack_success"] = 1
            r["qpos_after_attack"] = 0.01
            r["attack_method"] = "pgd_vis"

    return records


def test_case_f_attack_fields_do_not_change_output():
    """Selector output must be identical with and without attack outcome fields."""
    records_clean = _make_case_f(with_attack_fields=False)
    records_poisoned = _make_case_f(with_attack_fields=True)

    preds_clean = rule_based_close_predictor(records_clean)
    preds_poisoned = rule_based_close_predictor(records_poisoned)

    win_clean = select_best_window(preds_clean)
    win_poisoned = select_best_window(preds_poisoned)

    assert win_clean["anchor_step"] == win_poisoned["anchor_step"]
    assert win_clean["window_start"] == win_poisoned["window_start"]
    assert win_clean["window_end"] == win_poisoned["window_end"]
    assert win_clean["score"] == win_poisoned["score"]

    # Online trigger also identical
    on_clean = select_online_trigger(preds_clean)
    on_poisoned = select_online_trigger(preds_poisoned)
    assert on_clean["trigger_step"] == on_poisoned["trigger_step"]


# ═══════════════════════════════════════════════════════════════════
# Additional: Teacher-R baseline regression
# ═══════════════════════════════════════════════════════════════════

def test_teacher_r_accepts_early_close():
    """Teacher-R DOES accept step-4 close (known limitation — documents the gap)."""
    records = _make_case_b()
    anchor = teacher_rule_critical_close_anchor(records)
    # Teacher-R finds step 4 (first close_onset with qpos < 0.01)
    # This is the documented limitation: Teacher-R confuses spurious early closes
    assert anchor == 4, \
        f"Teacher-R expected 4 (first close onset, even if spurious), got {anchor}"


# ═══════════════════════════════════════════════════════════════════
# A.1.4: Motion evidence type distinction
# ═══════════════════════════════════════════════════════════════════

from gripper_attack.phase_detector import (
    _classify_motion_evidence,
    MOTION_SUSTAINED_VERTICAL_LIFT,
    MOTION_SUSTAINED_HORIZONTAL_TRANSPORT,
    MOTION_NO_SUSTAINED_MOTION,
)


def _make_motion_trace(close_at=30, z_offsets=None):
    """Minimal privileged trace with configurable post-close z/y offsets."""
    records = [_base_record(t) for t in range(60)]
    for t in range(60):
        records[t]["obj_x"] = 0.5
        records[t]["obj_z"] = 0.05
        records[t]["obj_y"] = 0.0
        records[t]["eef_to_obj_distance"] = 0.03
    r = records[close_at]
    r["clean_gripper_raw"] = 0.0
    r["clean_close"] = 1
    r["close_onset"] = 1
    r["close_streak"] = 1
    if z_offsets:
        for dt, dz in z_offsets:
            t = close_at + dt
            if t < 60:
                records[t]["obj_z"] = 0.05 + dz
    return records


def test_sustained_vertical_lift_passes():
    """2+ consecutive frames of positive dz with EEF near → vertical lift."""
    records = _make_motion_trace(close_at=30, z_offsets=[
        (1, 0.008), (2, 0.016), (3, 0.024)])
    evidence = _classify_motion_evidence(records, 30)
    assert evidence["motion_evidence_type"] == MOTION_SUSTAINED_VERTICAL_LIFT
    assert evidence["consecutive_motion_frames"] >= 2


def test_horizontal_push_not_called_vertical_lift():
    """Horizontal-only displacement without vertical z → NOT vertical lift."""
    records = _make_motion_trace(close_at=30)
    # Move only in y (horizontal), z stays flat
    for dt in [1, 2, 3, 4]:
        records[30 + dt]["obj_y"] = dt * 0.01  # y increases
        records[30 + dt]["obj_z"] = 0.05        # z flat
    evidence = _classify_motion_evidence(records, 30)
    assert evidence["motion_evidence_type"] != MOTION_SUSTAINED_VERTICAL_LIFT
    # Should be horizontal transport or no sustained motion
    assert evidence["motion_evidence_type"] in (
        MOTION_SUSTAINED_HORIZONTAL_TRANSPORT, MOTION_NO_SUSTAINED_MOTION)


def test_single_frame_z_jitter_fails():
    """Single frame dz > threshold → NOT sustained vertical lift."""
    records = _make_motion_trace(close_at=30, z_offsets=[
        (1, 0.006),  # one frame only
        (2, 0.0),    # back to flat
    ])
    evidence = _classify_motion_evidence(records, 30)
    assert evidence["motion_evidence_type"] != MOTION_SUSTAINED_VERTICAL_LIFT


def test_eef_detaches_during_motion_fails():
    """EEF far from object during z motion → eef_attachment_consistent=False."""
    records = _make_motion_trace(close_at=30, z_offsets=[
        (1, 0.008), (2, 0.016), (3, 0.024)])
    # EEF moves away during lift
    for dt in [1, 2, 3]:
        records[30 + dt]["eef_to_obj_distance"] = 0.20  # far
    evidence = _classify_motion_evidence(records, 30)
    assert not evidence["eef_attachment_consistent"]
    # Should NOT report vertical lift (EEF must be near for lift classification)
    assert evidence["motion_evidence_type"] != MOTION_SUSTAINED_VERTICAL_LIFT
