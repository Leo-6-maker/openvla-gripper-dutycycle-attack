"""CPU tests for Layer2 critical-close selector."""

from gripper_attack.critical_close_selector import (
    extract_deployment_features,
    rule_based_close_predictor,
    select_best_window,
    build_clean_proposal,
)
from gripper_attack.phase_detector import (
    teacher_phase_labels,
    teacher_critical_close_anchor,
)


def _make_trace(n_close_onset_at=50, n_steps=100):
    """Build a minimal clean trace for testing."""
    records = []
    for t in range(n_steps):
        rec = {
            "step": t,
            "clean_gripper_env": 1.0,
            "clean_gripper_raw": 0.0,
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
            rec["clean_gripper_raw"] = 0.0
        elif t > n_close_onset_at and t < n_close_onset_at + 20:
            rec["clean_close"] = 1
            rec["close_streak"] = t - n_close_onset_at + 1
            rec["clean_gripper_env"] = 1.0
        elif t > n_close_onset_at + 50:
            rec["decoded_open_bool"] = 1
            rec["clean_close"] = 0
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
    # The CLOSE onset step should have high score
    onset = preds[50]
    assert onset["score"] >= 3.0
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
    assert win["score"] >= 3.0


def test_build_proposal():
    win = {"window_start": 70, "window_end": 80, "anchor_step": 78,
           "score": 3.5, "abstain_reason": ""}
    p = build_clean_proposal("tomato_sauce", 0, "/tmp/t.csv", "abc", "abc123", win)
    assert p.is_valid()
    assert p.uses_clean_only
    assert not p.uses_attack_outcome
    assert p.window_start == 70
    assert p.anchor_step == 78


def test_teacher_anchor():
    records = _make_trace(n_close_onset_at=78)
    anchor = teacher_critical_close_anchor(records)
    assert anchor == 78


def test_teacher_phase_labels():
    records = _make_trace(n_close_onset_at=50)
    labels = teacher_phase_labels(records)
    assert len(labels) == 100
    # grasp_close label at onset
    assert labels[50] == "grasp_close"


# numpy import for tests that need it
import numpy as np
