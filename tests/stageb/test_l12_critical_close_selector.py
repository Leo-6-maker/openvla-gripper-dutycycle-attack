"""CPU tests for Layer2 critical-close selector."""

from gripper_attack.critical_close_selector import (
    extract_deployment_features,
    rule_based_close_predictor,
    select_best_window,
    select_online_trigger,
    build_clean_proposal,
)
from gripper_attack.phase_detector import (
    teacher_rule_phase_labels,
    teacher_rule_critical_close_anchor,
    teacher_privileged_critical_close_anchor,
)


def _make_trace(n_close_onset_at=50, n_steps=100):
    """Build a minimal clean trace for testing.

    Steps before close: gripper_raw=0.7 (OPEN), EEF moving.
    At close onset: gripper_raw=0.0 (CLOSE), close_onset=1, close_streak=1.
    After close: sustained CLOSE for 20 steps.
    After release: gripper OPEN, qpos > 0.01.
    """
    records = []
    for t in range(n_steps):
        rec = {
            "step": t,
            "clean_gripper_env": 1.0,
            "clean_gripper_raw": 0.7,   # OPEN before close
            "gripper_qpos_before": 0.0,
            "gripper_qpos_after": 0.0,
            "qpos_abs_before": 0.0,
            "qpos_abs_after": 0.0,
            "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.2,
            "clean_close": 0,
            "close_onset": 0,
            "close_streak": 0,
            "decoded_open_bool": 0,
        }
        if t == n_close_onset_at:
            rec["close_onset"] = 1
            rec["clean_close"] = 1
            rec["close_streak"] = 1
            rec["clean_gripper_env"] = 1.0
            rec["clean_gripper_raw"] = 0.0   # CLOSE command issued
        elif t > n_close_onset_at and t < n_close_onset_at + 20:
            rec["clean_close"] = 1
            rec["close_streak"] = t - n_close_onset_at + 1
            rec["clean_gripper_env"] = 1.0
            rec["clean_gripper_raw"] = 0.0   # sustained CLOSE
        elif t > n_close_onset_at + 50:
            rec["decoded_open_bool"] = 1
            rec["clean_close"] = 0
            rec["clean_gripper_raw"] = 0.7   # OPEN again
            rec["gripper_qpos_before"] = 0.03
            rec["qpos_abs_before"] = 0.03
        records.append(rec)
    return records


def test_feature_extraction_shape():
    records = _make_trace()
    feats = extract_deployment_features(records)
    assert feats.shape == (100, 13)
    assert feats.dtype == np.float32


def test_rule_predictor_returns_all_steps():
    records = _make_trace()
    preds = rule_based_close_predictor(records)
    assert len(preds) == 100
    for p in preds:
        assert "step" in p
        assert "score" in p
        assert "abstain" in p


def test_close_onset_gets_high_score():
    records = _make_trace(n_close_onset_at=50)
    preds = rule_based_close_predictor(records)
    # CLOSE onset + raw crossing should produce high score (>2.0)
    onset = preds[50]
    assert onset["score"] >= 2.0, f"score={onset['score']}"
    assert not onset["abstain"]


def test_post_release_abstains():
    records = _make_trace()
    preds = rule_based_close_predictor(records)
    # Step 90+ should abstain (gripper open)
    late = preds[95]
    assert late["abstain"] != ""


def test_best_window_selects_close_onset():
    records = _make_trace(n_close_onset_at=78)
    preds = rule_based_close_predictor(records)
    win = select_best_window(preds)
    assert win["anchor_step"] == 78
    assert win["window_start"] >= 76  # 78 - pre_offset(2)
    assert win["window_end"] == win["window_start"] + 10
    assert win["score"] >= 2.0, f"score={win['score']}"


def test_build_proposal():
    win = {"window_start": 70, "window_end": 80, "anchor_step": 78,
           "score": 3.5, "abstain_reason": ""}
    p = build_clean_proposal("tomato_sauce", 0, "/tmp/t.csv", "abc", "abc123", win)
    assert p.is_valid()
    assert p.uses_clean_only
    assert not p.uses_attack_outcome
    assert p.window_start == 70
    assert p.anchor_step == 78


def test_teacher_rule_anchor():
    records = _make_trace(n_close_onset_at=78)
    anchor = teacher_rule_critical_close_anchor(records)
    assert anchor == 78


def test_teacher_privileged_abstains_without_privileged_fields():
    """Teacher-P must abstain when trace lacks object/target pose fields."""
    records = _make_trace(n_close_onset_at=78)
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == -1  # abstain: no privileged fields in synthetic trace


def test_teacher_privileged_with_object_fields():
    """Teacher-P finds critical close when object is near and lift evidence exists."""
    records = _make_trace(n_close_onset_at=50)
    # Add privileged fields
    for t, r in enumerate(records):
        r["eef_to_obj_distance"] = 0.15  # far initially
        r["obj_to_target_distance"] = 0.3
        r["obj_x"] = 0.0
        r["obj_y"] = 0.0
        r["obj_z"] = 0.05
    # At close onset, EEF is near object
    records[50]["eef_to_obj_distance"] = 0.03  # 3cm = near
    # After close, EEF stays near and object lifts (sustained, 2+ consecutive frames)
    for t in range(51, 65):
        records[t]["eef_to_obj_distance"] = 0.04  # EEF stays near during lift
    for t in range(52, 60):
        records[t]["obj_z"] = 0.05 + 0.01 * (t - 51)  # rising (sustained lift)
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == 50


def test_teacher_privileged_rejects_early_far_close():
    """Teacher-P must NOT accept a close when EEF is far from object."""
    records = _make_trace(n_close_onset_at=4)  # early close like butter_s2
    for t, r in enumerate(records):
        r["eef_to_obj_distance"] = 0.5  # 50cm = far
        r["obj_to_target_distance"] = 0.3
        r["obj_x"] = 0.0
        r["obj_y"] = 0.0
        r["obj_z"] = 0.05
    # Later close at step 50 with EEF near object
    records[50]["close_onset"] = 1
    records[50]["clean_close"] = 1
    records[50]["close_streak"] = 1
    records[50]["eef_to_obj_distance"] = 0.02  # near
    records[50]["gripper_qpos_before"] = 0.0
    for t in range(51, 65):
        records[t]["clean_close"] = 1
        records[t]["close_streak"] = t - 50 + 1
        records[t]["eef_to_obj_distance"] = 0.03  # EEF stays near during lift
    for t in range(52, 60):
        records[t]["obj_z"] = 0.05 + 0.01 * (t - 51)  # sustained lift
    anchor = teacher_privileged_critical_close_anchor(records)
    assert anchor == 50  # step 50, NOT step 4


def test_teacher_phase_labels():
    records = _make_trace(n_close_onset_at=50)
    labels = teacher_rule_phase_labels(records)
    assert len(labels) == 100
    # grasp_close label at onset
    assert labels[50] == "grasp_close"


def test_horizon_labels_with_teacher_anchor():
    """Horizon labels correctly mark steps within H of teacher anchor."""
    records = _make_trace(n_close_onset_at=50)
    preds = rule_based_close_predictor(records, horizon=4, teacher_anchor=50)
    # Steps 47-49: within horizon (50 in [48, 51] for step 47, etc.)
    for p in preds:
        t = p["step"]
        if 46 <= t < 50:
            assert p["will_critical_close_within_horizon"], f"step {t} should be in horizon"
            assert p["predicted_close_horizon"] == 50 - t
        else:
            assert not p["will_critical_close_within_horizon"], f"step {t} should NOT be in horizon"


def test_student_is_invariant_to_idle_prefix_shift():
    """Student predictions should NOT systematically shift when idle steps are prepended."""
    records = _make_trace(n_close_onset_at=50)
    preds_orig = rule_based_close_predictor(records)

    # Prepend 20 idle steps (no CLOSE, no movement, just stationary OPEN)
    idle = []
    for t in range(20):
        idle.append({
            "step": t,
            "clean_gripper_env": 0.0,
            "clean_gripper_raw": 0.7,
            "gripper_qpos_before": 0.0,
            "qpos_abs_before": 0.0,
            "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.2,
            "clean_close": 0,
            "close_onset": 0,
            "close_streak": 0,
            "decoded_open_bool": 0,
        })
    shifted_records = idle + records
    preds_shifted = rule_based_close_predictor(shifted_records)

    # The highest-scoring step in the shifted trace should be at (20 + 50) = 70
    best_orig = max(preds_orig, key=lambda p: p["score"])
    best_shifted = max(preds_shifted, key=lambda p: p["score"])
    assert best_shifted["step"] == best_orig["step"] + 20, \
        f"Expected step {best_orig['step'] + 20}, got {best_shifted['step']}"


def test_student_has_no_absolute_step_feature():
    """Student scoring must NOT include absolute step thresholds (t<60, t>200)."""
    # Two identical traces, one starts at step 0, one at step 100
    records_early = _make_trace(n_close_onset_at=50, n_steps=100)
    # Simulate a late trace by making a trace where close onset is at step 50
    # but with high step numbers
    records_late = _make_trace(n_close_onset_at=50, n_steps=100)
    for r in records_late:
        r["step"] = r["step"] + 200  # shift all steps forward by 200

    preds_early = rule_based_close_predictor(records_early)
    preds_late = rule_based_close_predictor(records_late)

    # The score at the close onset step should be identical in both traces
    # (since the student has no absolute step thresholds)
    score_early = preds_early[50]["score"]
    score_late = preds_late[50]["score"]
    assert score_early == score_late, \
        f"Score depends on absolute step: early={score_early}, late={score_late}"


def test_online_trigger_fires_at_first_crossing():
    """Online trigger fires at first score threshold crossing with confirmation."""
    records = _make_trace(n_close_onset_at=50)
    preds = rule_based_close_predictor(records)
    win = select_online_trigger(preds, score_threshold=1.5, confirmation_steps=1)

    # Should trigger near the close onset (step 50)
    trigger = win.get("trigger_step", -1)
    assert trigger >= 45, f"Trigger too early: {trigger}"
    assert trigger <= 52, f"Trigger too late: {trigger}"
    assert win["window_start"] >= trigger - 2  # pre_offset
    assert win["abstain_reason"] == ""


def test_online_trigger_no_false_positive_on_idle():
    """Online trigger should NOT fire on idle traces with no close."""
    idle = []
    for t in range(50):
        idle.append({
            "step": t,
            "clean_gripper_env": 0.0,
            "clean_gripper_raw": 0.7,  # always OPEN
            "gripper_qpos_before": 0.0,
            "qpos_abs_before": 0.0,
            "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.2,
            "clean_close": 0,
            "close_onset": 0,
            "close_streak": 0,
            "decoded_open_bool": 0,
        })
    preds = rule_based_close_predictor(idle)
    win = select_online_trigger(preds, score_threshold=1.5)
    assert win["abstain_reason"] == "no_online_trigger"
    assert win["trigger_step"] == -1


def test_no_absolute_step_in_scoring():
    """Scores should only depend on relative physical features, not step number."""
    import numpy as np
    # Create trace with high step numbers but same physical features
    records_high_step = _make_trace(n_close_onset_at=50, n_steps=100)
    for r in records_high_step:
        r["step"] = r["step"] + 1000

    records_low_step = _make_trace(n_close_onset_at=50, n_steps=100)

    preds_high = rule_based_close_predictor(records_high_step)
    preds_low = rule_based_close_predictor(records_low_step)

    for t in range(100):
        assert abs(preds_high[t]["score"] - preds_low[t]["score"]) < 1e-6, \
            f"Step {t}: score_high={preds_high[t]['score']}, score_low={preds_low[t]['score']}"


# numpy import for tests that need it
import numpy as np
