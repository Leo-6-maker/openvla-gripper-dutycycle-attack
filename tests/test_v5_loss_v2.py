from __future__ import annotations

import torch

from gripper_attack.v5_dataset import V5Episode
from gripper_attack.v5_protocol import V5ModelContract, V5Window
from gripper_attack.v5_ranker import CausalMultimodalVulnerabilityRanker, compute_v5_loss_v2


def _episode(tiers: tuple[int, ...]) -> V5Episode:
    windows = []
    start = 0
    for index, tier in enumerate(tiers):
        end = start + 2
        windows.append(V5Window("ep", f"0:{index}", start, end, "VALID_RETENTION", tier, True, True))
        start = end + 1
    length = start
    return V5Episode(
        canonical_parent_key="ep", suite="libero_object", task_idx=0, state_id=0,
        features_25d=torch.randn(length, 25), valid_mask=torch.ones(length, dtype=torch.bool),
        candidate_close=torch.ones(length, dtype=torch.bool), utility_tier=torch.tensor([3] * length),
        known_mask=torch.ones(length, dtype=torch.bool), release_imminent=torch.zeros(length, dtype=torch.bool),
        regrasp_or_unstable=torch.zeros(length, dtype=torch.bool), release_known_mask=torch.ones(length, dtype=torch.bool),
        regrasp_known_mask=torch.ones(length, dtype=torch.bool), windows=tuple(windows),
    )


def test_v5_loss_trains_all_active_heads_and_ranking_components():
    episode = _episode((3, 1))
    model = CausalMultimodalVulnerabilityRanker(V5ModelContract("V5_A_PROPRIO"))
    output = model.forward_sequence(episode.features_25d.unsqueeze(0))
    losses = compute_v5_loss_v2(output["utility_logit"][0], output["release_logit"][0], output["regrasp_logit"][0], episode)
    losses["total"].backward()
    assert float(losses["tier_pairwise"]) >= 0.0
    assert model.utility_head.weight.grad is not None and float(model.utility_head.weight.grad.abs().sum()) > 0.0
    assert model.release_head.weight.grad is not None and float(model.release_head.weight.grad.abs().sum()) > 0.0
    assert model.regrasp_head.weight.grad is not None and float(model.regrasp_head.weight.grad.abs().sum()) > 0.0
    assert model.support_head is None
    assert model.uncertainty_head is None


def test_pure_negative_loss_is_driven_by_highest_window():
    episode = _episode((1, 1))
    model = CausalMultimodalVulnerabilityRanker(V5ModelContract("V5_A_PROPRIO"))
    output = model.forward_sequence(episode.features_25d.unsqueeze(0))
    loss = compute_v5_loss_v2(output["utility_logit"][0], output["release_logit"][0], output["regrasp_logit"][0], episode)["pure_negative_abstention"]
    loss.backward()
    assert float(loss) > 0.0
    assert model.utility_head.weight.grad is not None and float(model.utility_head.weight.grad.abs().sum()) > 0.0
