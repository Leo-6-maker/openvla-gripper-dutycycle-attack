"""CPU tests for streaming replay state machine."""

from gripper_attack.streaming_state import CloseEventStreamingState
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
            "clean_close": 0,
            "close_onset": 0,
            "close_streak": 0,
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


def test_streaming_state_matches_batch():
    """Streaming state machine produces identical scores to batch predictor."""
    records = _make_trace(n_close_onset_at=50)
    preds_batch = rule_based_close_predictor(records)

    state = CloseEventStreamingState()
    for r in records:
        state.update(r)

    for t in range(len(records)):
        assert abs(preds_batch[t]["score"] - state.predictions[t]["score"]) < 1e-6, \
            f"Step {t}: batch={preds_batch[t]['score']}, streaming={state.predictions[t]['score']}"


def test_streaming_state_trigger_matches_batch():
    """Streaming online trigger matches batch online trigger."""
    records = _make_trace(n_close_onset_at=50)
    preds_batch = rule_based_close_predictor(records)
    win_batch = select_online_trigger(preds_batch, score_threshold=1.5, confirmation_steps=1)

    state = CloseEventStreamingState(score_threshold=1.5, confirmation_steps=1)
    for r in records:
        state.update(r)

    win_stream = state.online_window()
    assert win_batch["trigger_step"] == win_stream["trigger_step"], \
        f"Batch trigger={win_batch['trigger_step']}, stream={win_stream['trigger_step']}"


def test_streaming_state_causal_no_future_leak():
    """Streaming state never accesses future records."""
    records = _make_trace(n_close_onset_at=50)
    state = CloseEventStreamingState()

    for t in range(len(records)):
        # Only pass records[:t+1]
        state.update(records[t])

        # At step t, internal step count should be t+1
        # and predictions list should have t+1 entries
        assert len(state.predictions) == t + 1


def test_streaming_state_triggers_once():
    """Streaming state triggers at most once (no re-trigger within cooldown)."""
    records = _make_trace(n_close_onset_at=50)
    state = CloseEventStreamingState(score_threshold=1.5, confirmation_steps=1)
    trigger_count = 0
    for r in records:
        pred = state.update(r)
        if pred["triggered"]:
            trigger_count += 1
    assert trigger_count == 1
    assert state.triggered
    assert state.trigger_step >= 0


def test_streaming_state_no_trigger_on_idle():
    """Streaming state should not trigger on idle trace."""
    idle = []
    for t in range(50):
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
    state = CloseEventStreamingState()
    for r in idle:
        state.update(r)
    assert not state.triggered
    win = state.online_window()
    assert win["abstain_reason"] == "no_online_trigger"
