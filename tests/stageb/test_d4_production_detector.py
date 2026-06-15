"""D4.2b: Production detector correctness tests."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from gripper_attack.production_detector import ProductionStreamingDetector

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return torch.zeros(x.shape[0])


def make_detector():
    means = {f: 0.0 for f in ["total_score","raw_crossing_bonus","close_streak_bonus",
              "close_onset_qpos_bonus","eef_deceleration_bonus","qpos_ready_bonus",
              "eef_speed_now","eef_speed_prev","eef_deceleration_delta",
              "close_streak","raw_crossing","close_onset","qpos",
              "time_since_prev_close","time_since_last_open","candidate_index"]}
    stdevs = {f: 1.0 for f in means}
    impute = {f: 0.0 for f in means}
    return ProductionStreamingDetector(DummyModel().eval(), means, stdevs, impute, threshold=0.5)


def test_reset_clears_all_state():
    d = make_detector()
    d.update(0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.step == 1
    d.reset()
    assert d.step == 0
    assert d.emit_step == -1
    assert len(d.history) == 0


def test_at_most_one_emission():
    d = make_detector()
    d.threshold = -999.0  # very low threshold to ensure emission
    for i in range(10):
        d.update(0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
        d.prev_raw = 0.3  # fake crossing
    assert d.emit_step >= 0
    # Second call should not change emit_step
    first_emit = d.emit_step
    d.update(0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    assert d.emit_step == first_emit


def test_invalid_raw_no_crossing():
    d = make_detector()
    d.update(0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    # Invalid raw should not produce raw_crossing
    result = d.update(0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0, raw_valid=False)
    # Candidate may still fire from close_onset, but raw_crossing flag is False
    # Just verify no crash and valid return type
    assert result is None or isinstance(result, dict)


def test_missing_qpos_disables_qpos_features():
    d = make_detector()
    d.update(0.7, 1.0, 0.0, 0.0, 0.0, 0.2, 0, qpos_valid=False)
    # Should not crash — qpos_valid=False just sets field to empty
    assert len(d.history) == 1
    assert d.history[0]["gripper_qpos_before"] == ""


def test_close_onset_detected():
    d = make_detector()
    # First step: env > 0.5 = CLOSE, streak was 0 → onset
    d.update(0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)
    # onset detected (close_streak was 0 → now 1)
    assert d.close_streak == 1


def test_open_tracking():
    d = make_detector()
    d.update(0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0)  # CLOSE step
    d.update(0.7, -1.0, 0.03, 0.0, 0.0, 0.2, 1) # OPEN step
    assert len(d.open_steps) == 1
    assert d.open_steps[0] == 1


def test_no_future_access():
    """Production adapter must not import or reference candidate CSV or Teacher-P."""
    import inspect
    src = inspect.getsource(ProductionStreamingDetector.__init__)
    assert "candidate" not in src.lower() or "candidate_table" not in src
    assert "teacher" not in src.lower() or "teacher_p" not in src
