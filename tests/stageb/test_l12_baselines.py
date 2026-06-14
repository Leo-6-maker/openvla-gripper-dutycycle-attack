"""CPU tests for L12 baselines."""

import numpy as np

from gripper_attack.l12_baselines import (
    time_only_window,
    task_only_window,
    close_event_rule_baseline,
    label_shuffle_baseline,
    prevalence_baseline,
    always_abstain_baseline,
    TASK_MEDIAN_ANCHORS,
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


def test_time_only_uses_quarter_trajectory():
    win = time_only_window(n_steps=100)
    assert win["anchor_step"] == 25  # 25% of 100
    assert win["window_start"] == 23  # 25 - 2
    assert win["window_end"] == 33    # 23 + 10


def test_time_only_zero_trajectory():
    win = time_only_window(n_steps=8)
    assert win["anchor_step"] == 2


def test_task_only_known_task():
    for task in TASK_MEDIAN_ANCHORS:
        win = task_only_window(task, n_steps=200)
        assert win["anchor_step"] == TASK_MEDIAN_ANCHORS[task]


def test_task_only_unknown_task():
    win = task_only_window("unknown_task", n_steps=200)
    assert win["anchor_step"] == 50  # 25% fallback
    assert win["score"] == 0.5


def test_close_event_rule_baseline_works():
    records = _make_trace(n_close_onset_at=50)
    win = close_event_rule_baseline(records)
    assert win["anchor_step"] == 50
    assert win["score"] >= 1.5


def test_label_shuffle_returns_n_results():
    records = _make_trace(n_close_onset_at=50)
    results = label_shuffle_baseline(records, teacher_p_anchor=50, n_shuffles=20)
    assert len(results) == 20
    for r in results:
        assert "shuffle_seed" in r
        assert "fake_anchor" in r


def test_label_shuffle_anchors_vary():
    records = _make_trace(n_close_onset_at=50)
    results = label_shuffle_baseline(records, teacher_p_anchor=50, n_shuffles=20)
    anchors = [r["fake_anchor"] for r in results]
    assert len(set(anchors)) >= 10  # most seeds should produce different anchors


def test_prevalence_baseline_returns_teacher():
    records = _make_trace(n_close_onset_at=78)
    win = prevalence_baseline(records, teacher_p_anchor=78)
    assert win["anchor_step"] == 78
    assert win["score"] == 5.0


def test_prevalence_baseline_abstains_when_teacher_abstains():
    win = prevalence_baseline([], teacher_p_anchor=-1)
    assert win["anchor_step"] == -1
    assert win["abstain_reason"] == "teacher_abstained"


def test_always_abstain_baseline():
    win = always_abstain_baseline()
    assert win["anchor_step"] == -1
    assert win["abstain_reason"] == "always_abstain_baseline"
