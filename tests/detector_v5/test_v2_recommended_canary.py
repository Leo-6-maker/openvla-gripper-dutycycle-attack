from __future__ import annotations

import torch

from gripper_attack.v5_factorized_dataset import FactorizedEpisode
from gripper_attack.v5_factorized_student_v2_recommended import (
    ExactCausalTCNEncoder,
    RecommendedEventBalancedLoss,
)


def _episode() -> FactorizedEpisode:
    T = 4
    return FactorizedEpisode(
        canonical_parent_key="suite/task/state",
        suite="suite",
        task_idx=0,
        state_id=0,
        mechanism_route="single_object_pick_place",
        route_supported=True,
        features_25d=torch.zeros(T, 25),
        valid_mask=torch.ones(T, dtype=torch.bool),
        grasp_target=torch.zeros(T, dtype=torch.bool),
        grasp_known_mask=torch.zeros(T, dtype=torch.bool),
        manipulation_target=torch.zeros(T, dtype=torch.bool),
        manipulation_known_mask=torch.zeros(T, dtype=torch.bool),
        release_target=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        release_known_mask=torch.ones(T, dtype=torch.bool),
        event_id=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        event_role=["primary"] * T,
        active_object_name=["obj"] * T,
        k10_feasible=torch.zeros(T, dtype=torch.bool),
        k10_known_mask=torch.ones(T, dtype=torch.bool),
        policy_intent_9d=torch.empty(0),
        policy_intent_valid_mask=torch.zeros(T, dtype=torch.bool),
    )


def test_exact_context_is_exact_for_frozen_values():
    for context in (16, 32, 64):
        encoder = ExactCausalTCNEncoder(25, 64, context, 0.0)
        assert encoder.actual_receptive_field == context
        x = torch.randn(2, 73, 25)
        y = encoder(x)
        assert y.shape == (2, 73, 64)


def test_route_class_weights_change_loss():
    episode = _episode()
    logits = {
        "grasp": torch.zeros(1, 4),
        "manipulation": torch.zeros(1, 4),
        "release": torch.zeros(1, 4),
    }
    valid = torch.ones(1, 4, dtype=torch.bool)
    loss_fn = RecommendedEventBalancedLoss()

    unweighted, _, _ = loss_fn(logits, [episode], valid_mask=valid, class_weights={})
    weighted, _, _ = loss_fn(
        logits,
        [episode],
        valid_mask=valid,
        class_weights={
            "release": {"pos_weight": 3.0, "neg_weight": 1.0},
        },
    )
    assert weighted.item() > unweighted.item()


def test_invalid_jitter_prefix_is_excluded_from_loss():
    episode = _episode()
    valid = torch.tensor([[False, False, True, True]])
    base = {
        "grasp": torch.zeros(1, 4),
        "manipulation": torch.zeros(1, 4),
        "release": torch.tensor([[0.0, 0.0, -1.0, -1.0]]),
    }
    changed_invalid_prefix = {
        "grasp": base["grasp"].clone(),
        "manipulation": base["manipulation"].clone(),
        "release": torch.tensor([[20.0, -20.0, -1.0, -1.0]]),
    }
    loss_fn = RecommendedEventBalancedLoss()
    first, _, _ = loss_fn(base, [episode], valid_mask=valid, class_weights={})
    second, _, _ = loss_fn(
        changed_invalid_prefix,
        [episode],
        valid_mask=valid,
        class_weights={},
    )
    assert torch.allclose(first, second, atol=1e-7)
