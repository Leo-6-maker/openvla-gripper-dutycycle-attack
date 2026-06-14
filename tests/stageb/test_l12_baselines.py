"""CPU tests for L12 baselines (repaired — no hardcoded table, no leakage)."""

import numpy as np

from gripper_attack.l12_baselines import (
    offline_time_only_diagnostic,
    online_safe_time_baseline,
    task_only_window,
    close_event_rule_baseline,
    label_shuffle_null,
    train_fold_prevalence,
    oracle_anchor_upper_bound,
    always_abstain_baseline,
)
from gripper_attack.critical_close_selector import WINDOW_LEN, PRE_OFFSET


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


# ── E1.1: TimeOnly ──

def test_offline_time_uses_quarter_trajectory():
    win = offline_time_only_diagnostic(n_steps=100)
    assert win["anchor_step"] == 25
    assert win["window_start"] == 23
    assert win["prediction_mode"] == "offline_time_only_diagnostic"


def test_online_time_no_future_episode_length():
    """Online time baseline uses only current step, not episode length."""
    win_early = online_safe_time_baseline(current_step=10)
    assert win_early["abstain_reason"] == "too_early_time_heuristic"
    assert win_early["anchor_step"] == -1

    win_late = online_safe_time_baseline(current_step=50)
    assert win_late["anchor_step"] == 50
    assert win_late["prediction_mode"] == "online_time_baseline"


# ── E1.2: TaskOnly (no hardcoded table) ──

def test_task_only_uses_train_fold():
    train_medians = {"butter": 5, "cream_cheese": 60}
    win = task_only_window("butter", train_fold_median_anchors=train_medians)
    assert win["anchor_step"] == 5
    assert win["prediction_mode"] == "task_only_baseline"


def test_task_only_unknown_task_falls_back_to_global():
    win = task_only_window("unknown_task", train_fold_median_anchors={},
                            global_train_median=40)
    assert win["anchor_step"] == 40


def test_task_only_never_uses_eval_trace_anchor():
    """TaskOnly does not receive eval trace data — only task key + train fold."""
    # No eval trace passed as argument — only task key and train fold stats
    win = task_only_window("cream_cheese",
                            train_fold_median_anchors={"cream_cheese": 59})
    assert win["anchor_step"] == 59
    # The trace itself is never inspected


# ── E1.3: LabelShuffle (evaluation null) ──

def test_label_shuffle_returns_n_results():
    records = _make_trace(n_close_onset_at=50)
    results = label_shuffle_null(records, teacher_p_anchor=50, n_shuffles=20)
    assert len(results) == 20
    for r in results:
        assert "shuffled_eval_target" in r
        assert "original_teacher_anchor" in r
        assert r["original_teacher_anchor"] == 50


def test_label_shuffle_targets_vary():
    records = _make_trace(n_close_onset_at=50)
    results = label_shuffle_null(records, teacher_p_anchor=50, n_shuffles=20)
    targets = [r["shuffled_eval_target"] for r in results]
    assert len(set(targets)) >= 10  # most seeds should shuffle differently


def test_label_shuffle_selector_output_unchanged():
    """Rule-based selector produces same window regardless of shuffled target."""
    records = _make_trace(n_close_onset_at=50)
    results = label_shuffle_null(records, teacher_p_anchor=50, n_shuffles=5)
    anchors = [r["anchor_step"] for r in results]
    # Rule-based selector doesn't use teacher, so all anchors identical
    assert len(set(anchors)) == 1
    assert anchors[0] == 50  # CloseEventRule finds the only close


# ── E1.4: Prevalence & Oracle ──

def test_train_fold_prevalence_uses_global_median():
    win = train_fold_prevalence(train_fold_global_median=48)
    assert win["anchor_step"] == 48
    assert win["prediction_mode"] == "train_fold_prevalence"


def test_oracle_upper_bound_not_reported_as_baseline():
    win = oracle_anchor_upper_bound(teacher_p_anchor=78)
    assert win["anchor_step"] == 78
    assert win["score"] == 5.0
    assert win["prediction_mode"] == "oracle_upper_bound"


def test_oracle_abstains_when_teacher_abstains():
    win = oracle_anchor_upper_bound(teacher_p_anchor=-1)
    assert win["anchor_step"] == -1
    assert win["abstain_reason"] == "teacher_abstained"


def test_always_abstain_baseline():
    win = always_abstain_baseline()
    assert win["anchor_step"] == -1
    assert win["abstain_reason"] == "always_abstain_baseline"


def test_close_event_rule_baseline_works():
    records = _make_trace(n_close_onset_at=50)
    win = close_event_rule_baseline(records)
    assert win["anchor_step"] == 50
    assert win["score"] >= 1.5
