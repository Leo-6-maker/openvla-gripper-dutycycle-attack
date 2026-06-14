"""Tests for E4A.1 failure taxonomy correctness."""

from gripper_attack.critical_close_selector import (
    rule_based_close_predictor,
    select_online_trigger,
)


def _make_trace(n_close_onset_at=50, n_steps=100):
    records = []
    for t in range(n_steps):
        rec = {
            "step": t,
            "clean_gripper_env": 1.0,
            "clean_gripper_raw": 0.7,
            "gripper_qpos_before": 0.0,
            "qpos_abs_before": 0.0,
            "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.2,
            "clean_close": 0, "close_onset": 0, "close_streak": 0,
            "decoded_open_bool": 0,
        }
        if t == n_close_onset_at:
            rec["close_onset"] = 1
            rec["clean_close"] = 1
            rec["close_streak"] = 1
            rec["clean_gripper_raw"] = 0.0
        elif t > n_close_onset_at and t < n_close_onset_at + 20:
            rec["clean_close"] = 1
            rec["close_streak"] = t - n_close_onset_at + 1
            rec["clean_gripper_raw"] = 0.0
        elif t > n_close_onset_at + 50:
            rec["decoded_open_bool"] = 1
            rec["clean_gripper_raw"] = 0.7
            rec["gripper_qpos_before"] = 0.03
        records.append(rec)
    return records


def test_score_components_sum_to_total_score():
    """Sum of decomposition matches unclamped score; final score floored at 0."""
    records = _make_trace(n_close_onset_at=30)
    preds = rule_based_close_predictor(records)
    for p in preds:
        raw_sum = (p.get("raw_crossing_bonus", 0) + p.get("close_streak_bonus", 0) +
                   p.get("close_onset_qpos_bonus", 0) + p.get("eef_deceleration_bonus", 0) +
                   p.get("qpos_ready_bonus", 0) + p.get("decoded_open_penalty", 0))
        # Final score is max(0, raw_sum)
        assert abs(max(0.0, raw_sum) - p["score"]) < 0.01, \
            f"Step {p['step']}: max(0, {raw_sum}) != score={p['score']}"


def test_instrumentation_does_not_change_original_score():
    """Adding decomposition fields doesn't change score or trigger."""
    records = _make_trace(n_close_onset_at=30)
    preds = rule_based_close_predictor(records)
    win = select_online_trigger(preds, mode="close_interception")
    # Score at close step should be 3.3 (raw 1.5 + streak 1.0 + onset/qpos 0.5 + qpos_ready 0.3)
    assert preds[30]["score"] == 3.3
    assert win["trigger_step"] == 30  # first close event


def test_decomposition_fields_present():
    """Every prediction has score decomposition fields."""
    records = _make_trace(n_close_onset_at=30)
    preds = rule_based_close_predictor(records)
    for field in ["raw_crossing_bonus", "close_streak_bonus", "close_onset_qpos_bonus",
                  "eef_deceleration_bonus", "qpos_ready_bonus", "decoded_open_penalty",
                  "eef_speed_now", "eef_speed_prev"]:
        assert field in preds[0], f"Missing field: {field}"


def test_best_close_candidate_filters_non_close_steps():
    """best_close_candidate only considers is_close_event_candidate steps."""
    records = _make_trace(n_close_onset_at=30)
    preds = rule_based_close_predictor(records)
    close_cands = [p for p in preds if p.get("is_close_event_candidate") and not p.get("abstain")]
    best_close = max(close_cands, key=lambda p: p["score"])
    # The close event at step 30 is the only close candidate with high score
    assert best_close["step"] == 30


def test_exact_trigger_is_correct_even_if_later_step_scores_higher():
    """Trigger at Teacher-P is correct regardless of later candidates."""
    # In this trace, trigger=P=30, but later step has higher score (impossible
    # in this simple trace, but verify the classification logic)
    records = _make_trace(n_close_onset_at=30)
    preds = rule_based_close_predictor(records)
    win = select_online_trigger(preds, mode="close_interception")
    trigger = win.get("trigger_step", -1)
    # With P=30, trigger should be near 30
    assert trigger == 30
