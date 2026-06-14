"""Tests for E4B.1 causal policy timing semantics."""

from scripts.stageb.run_l12_e4b_counterfactual import (
    policy_first_threshold,
    policy_bounded_peak_hold,
    policy_local_maximum,
    policy_non_causal_record_high,
    policy_non_causal_global_argmax,
)


def _make_trace(n_close_at=30, n_steps=100):
    records = []
    for t in range(n_steps):
        rec = {
            "step": t, "clean_gripper_env": 1.0, "clean_gripper_raw": 0.7,
            "gripper_qpos_before": 0.0, "qpos_abs_before": 0.0,
            "eef_x": 0.0, "eef_y": 0.0, "eef_z": 0.2,
            "clean_close": 0, "close_onset": 0, "close_streak": 0,
            "decoded_open_bool": 0,
        }
        if t == n_close_at:
            rec["close_onset"] = 1; rec["clean_close"] = 1
            rec["close_streak"] = 1; rec["clean_gripper_raw"] = 0.0
        elif t > n_close_at and t < n_close_at + 20:
            rec["clean_close"] = 1; rec["close_streak"] = t - n_close_at + 1
            rec["clean_gripper_raw"] = 0.0
        elif t > n_close_at + 50:
            rec["decoded_open_bool"] = 1; rec["clean_gripper_raw"] = 0.7
            rec["gripper_qpos_before"] = 0.03
        records.append(rec)
    return records


def _preds(records):
    from gripper_attack.critical_close_selector import rule_based_close_predictor
    return rule_based_close_predictor(records)


def test_first_threshold_decision_equals_selected():
    p = _preds(_make_trace(30))
    r = policy_first_threshold(p)
    assert r["causal"] is True
    assert r["selected_event_step"] == r["decision_step"] == r["actuation_step"]


def test_peak_hold_decision_is_after_full_hold():
    p = _preds(_make_trace(30))
    r = policy_bounded_peak_hold(p, hold_steps=4)
    assert r["causal"] is True
    # decision_step must be first_t + hold_steps
    first_t = r["first_threshold_step"]
    assert r["decision_step"] == first_t + 4
    assert r["actuation_step"] == r["decision_step"]


def test_peak_hold_never_retroactively_actuates():
    p = _preds(_make_trace(30))
    r = policy_bounded_peak_hold(p, hold_steps=8)
    # actuation >= selected_event (can't actuate before the event exists)
    assert r["actuation_step"] >= r["selected_event_step"]


def test_local_max_decision_uses_next_step():
    p = _preds(_make_trace(30))
    r = policy_local_maximum(p)
    if r["causal"]:
        assert r["decision_step"] == r["selected_event_step"] + 1


def test_noncausal_record_high_is_marked_noncausal():
    p = _preds(_make_trace(30))
    r = policy_non_causal_record_high(p)
    assert r["causal"] is False


def test_noncausal_global_argmax_is_marked_noncausal():
    p = _preds(_make_trace(30))
    r = policy_non_causal_global_argmax(p)
    assert r["causal"] is False
    assert r["policy"] == "noncausal_global_score_argmax"


def test_selected_event_and_actuation_metrics_are_separate():
    p = _preds(_make_trace(30))
    r = policy_bounded_peak_hold(p, hold_steps=4)
    assert "selected_event_step" in r
    assert "actuation_step" in r
    # selected_event_step is within [first, first+K]; actuation = first+K
    assert r["first_threshold_step"] <= r["selected_event_step"] <= r["actuation_step"]


def test_close_streak_field_populated():
    p = _preds(_make_trace(30))
    assert "close_streak_value" in p[0]
    assert p[30]["close_streak_value"] == 1
