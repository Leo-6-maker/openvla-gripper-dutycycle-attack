from __future__ import annotations

import pytest
import torch

from gripper_attack.v5_protocol import (
    V5ModelContract,
    V5Window,
    feature_order_sha,
    validate_phase_windows,
    validate_student_features,
    validate_teacher_row,
)
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker, v5_window_ranking_loss
from gripper_attack.v5_scheduler import V5OneShotScheduler
from gripper_attack.v5_teacher import convert_teacher_row


def test_v5_feature_order_is_name_bound_and_student_rejects_teacher_fields():
    contract = V5ModelContract("V5_D_PROPRIO_POLICY_INTENT_CAUSAL_VISUAL", visual_dim=8)
    assert contract.proprio_order_sha256 == feature_order_sha(contract.to_dict()["proprio_features"])
    row = {"features_25d": [0.0] * 25, "quality_valid": True}
    with pytest.raises(ValueError, match="forbidden"):
        validate_student_features(row)


def test_v5_teacher_known_supervision_requires_xor():
    row = {
        "canonical_parent_key": "libero_spatial/task_00/state_00",
        "step": 0,
        "event_id": 0,
        "phase_id": 0,
        "window_id": "0:0",
        "phase_name": "VALID_RETENTION",
        "window_start": 0,
        "window_end": 3,
        "candidate_close": True,
        "quality_valid": True,
        "veto_invalid": True,
        "release_imminent": False,
        "regrasp_or_unstable": False,
        "known_mask": True,
        "utility_tier": 3,
        "ranking_group": "libero_spatial/task_00/state_00",
    }
    with pytest.raises(ValueError, match="XOR"):
        validate_teacher_row(row)


def test_v5_teacher_utility_is_proxy_and_unknown_is_masked():
    row = convert_teacher_row(
        {
            "step": 0,
            "event_id": 3,
            "phase_segment_index": 1,
            "phase_name": "stable_carry",
            "window_start": 0,
            "window_end": 4,
            "candidate_close": True,
            "quality_valid": True,
            "veto_invalid": False,
            "known_mask": True,
            "retention_continuation_t10": True,
        },
        "libero_spatial/task_00/state_00",
        0,
    )
    assert row["utility_tier"] == 2
    high_value = convert_teacher_row(
        {
            "step": 0,
            "event_id": 3,
            "phase_segment_index": 1,
            "phase_name": "VALID_RETENTION",
            "window_start": 0,
            "window_end": 10,
            "candidate_close": True,
            "quality_valid": True,
            "veto_invalid": False,
            "known_mask": True,
        },
        "libero_spatial/task_00/state_00",
        0,
    )
    assert high_value["utility_tier"] == 3
    unknown = convert_teacher_row(
        {
            "step": 1,
            "event_id": -1,
            "phase_name": "abstain_unsupported",
            "candidate_close": False,
            "quality_valid": False,
            "veto_invalid": False,
            "known_mask": False,
        },
        "libero_spatial/task_00/state_00",
        1,
    )
    assert unknown["known_mask"] is False
    assert unknown["utility_tier"] is None


def test_v5_windows_do_not_collapse_phase_segments():
    windows = [
        V5Window("ep", "7:0", 2, 5, "PRE_SUPPORT", None, False, True),
        V5Window("ep", "7:1", 6, 10, "VALID_RETENTION", 3, True, True),
    ]
    validate_phase_windows(windows)


def test_v5_ranker_has_gradient_through_window_ranking():
    model = CausalMultimodalVulnerabilityRanker(V5ModelContract("V5_B_PROPRIO_POLICY_INTENT"))
    x = torch.randn(1, 4, 25)
    i = torch.randn(1, 4, 9)
    output = model.forward_sequence(x, i)
    windows = [
        {"episode_id": "ep", "known": True, "utility_tier": 3},
        {"episode_id": "ep", "known": True, "utility_tier": 1},
    ]
    loss = v5_window_ranking_loss(output["utility_logit"][0, :2], windows)
    loss.backward()
    assert model.utility_head.weight.grad is not None
    assert float(model.utility_head.weight.grad.abs().sum()) > 0.0


def test_v5_scheduler_is_candidate_gated_and_one_shot():
    scheduler = V5OneShotScheduler()
    for step in range(3):
        result = scheduler.update(
            step=step,
            candidate_close=True,
            valid=True,
            utility_probability=0.9,
            release_probability=0.0,
            regrasp_probability=0.0,
            uncertainty_probability=0.0,
        )
    assert result["emit"] is True
    assert result["emit_step"] == 2
    after = scheduler.update(
        step=3,
        candidate_close=True,
        valid=True,
        utility_probability=1.0,
        release_probability=0.0,
        regrasp_probability=0.0,
        uncertainty_probability=0.0,
    )
    assert after["emit"] is False
    assert after["one_shot_emitted"] is True


def test_v5_scheduler_abstains_on_veto():
    scheduler = V5OneShotScheduler()
    for step in range(5):
        result = scheduler.update(
            step=step,
            candidate_close=True,
            valid=True,
            utility_probability=0.99,
            release_probability=0.9,
            regrasp_probability=0.0,
            uncertainty_probability=0.0,
        )
    assert result["emit"] is False
    assert result["one_shot_emitted"] is False
