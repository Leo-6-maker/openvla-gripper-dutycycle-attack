from __future__ import annotations

import sys
from pathlib import Path

from gripper_attack.v5_scheduler import V5OneShotScheduler

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.detector_v5.evaluate_v5_causal_online import _select_working_point


def test_scheduler_replay_is_causal_min_dwell_and_one_shot():
    scheduler = V5OneShotScheduler()
    emitted = []
    for step in range(20):
        result = scheduler.update(
            step=step, candidate_close=True, valid=True,
            utility_probability=0.9, release_probability=0.0,
            regrasp_probability=0.0, uncertainty_probability=0.99,
        )
        emitted.append(result["emit"])
        assert result["teacher_inputs_consumed"] is False
    assert sum(emitted) == 1
    assert emitted[0] is False
    assert scheduler.emit_step >= 9


def test_working_point_uses_maximum_threshold_meeting_recall():
    result = _select_working_point([
        {"threshold": 0.5, "critical_window_recall": 0.96},
        {"threshold": 0.6, "critical_window_recall": 0.95},
        {"threshold": 0.7, "critical_window_recall": 0.94},
    ])
    assert result["status"] == "PASS"
    assert result["selected_threshold"] == 0.6


def test_working_point_holds_when_no_threshold_meets_recall():
    result = _select_working_point([
        {"threshold": 0.5, "critical_window_recall": 0.94},
    ])
    assert result["status"] == "HOLD"
    assert result["selected_threshold"] is None
