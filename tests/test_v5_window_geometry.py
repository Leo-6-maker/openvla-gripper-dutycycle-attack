from __future__ import annotations

import torch

from gripper_attack.v5_dataset import V5Episode, aggregate_retrospective_window_scores, causal_window_anchor_scores, classify_v5_episode_windows
from gripper_attack.v5_protocol import V5Window


def _episode(length: int = 12) -> V5Episode:
    window = V5Window("ep", "0:0", 0, length - 1, "VALID_RETENTION", 3, True, True)
    return V5Episode(
        canonical_parent_key="ep",
        suite="libero_object",
        task_idx=0,
        state_id=0,
        features_25d=torch.zeros(length, 25),
        valid_mask=torch.ones(length, dtype=torch.bool),
        candidate_close=torch.ones(length, dtype=torch.bool),
        utility_tier=torch.full((length,), 3, dtype=torch.long),
        known_mask=torch.ones(length, dtype=torch.bool),
        release_imminent=torch.zeros(length, dtype=torch.bool),
        regrasp_or_unstable=torch.zeros(length, dtype=torch.bool),
        release_known_mask=torch.ones(length, dtype=torch.bool),
        regrasp_known_mask=torch.ones(length, dtype=torch.bool),
        windows=(window,),
    )


def test_causal_anchor_does_not_read_future_window_steps():
    episode = _episode()
    logits = torch.zeros(12)
    logits[10:] = 100.0
    retrospective, _ = aggregate_retrospective_window_scores(logits, episode)
    causal, rows = causal_window_anchor_scores(logits, episode)
    assert float(retrospective[0]) > 0.0
    assert float(causal[0]) == 0.0
    assert rows[0]["decision_anchor_step"] == 9


def test_short_window_is_not_minimum_dwell_candidate():
    window = V5Window("ep", "0:0", 0, 4, "VALID_RETENTION", 3, True, True)
    assert window.minimum_dwell_met is False
    assert window.decision_anchor_step == 4


def test_strict_episode_categories_do_not_call_positive_only_mixed():
    positive = V5Window("ep", "positive", 0, 9, "VALID_RETENTION", 3, True, True)
    negative = V5Window("ep", "negative", 10, 19, "PRE_SUPPORT", 1, True, True)
    assert classify_v5_episode_windows((positive, negative)) == "TRUE_MIXED"
    assert classify_v5_episode_windows((positive,)) == "POSITIVE_ONLY"
    assert classify_v5_episode_windows((negative,)) == "PURE_NEGATIVE"


def test_causal_category_uses_anchor_tier_and_excludes_short_windows():
    promoted = V5Window("ep", "promoted", 0, 9, "VALID_RETENTION", 3, True, True, causal_utility_tier=1)
    short = V5Window("ep", "short", 10, 12, "VALID_RETENTION", 3, True, True, causal_utility_tier=3)
    assert classify_v5_episode_windows((promoted,)) == "POSITIVE_ONLY"
    assert classify_v5_episode_windows((promoted,), causal=True) == "PURE_NEGATIVE"
    assert classify_v5_episode_windows((short,), causal=True) == "NO_CANDIDATE"
