"""Compact, frozen CLEAN features for the official OpenVLA collector.

The deployed detector may consume only causal 25D proprio/action features and
the 9D clean gripper-intent summary.  Privileged simulator fields stay in a
separate sidecar and are never part of the student feature vector.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .sc5_streaming_features_v2 import FEATURE_NAMES as CANONICAL_25D_FEATURES
from .sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2


CLEAN_POLICY_FEATURE_NAMES = (
    "clean_open_probability_mass",
    "clean_close_probability_mass",
    "clean_open_minus_close_log_mass",
    "clean_action_token_entropy_normalized",
    "clean_top1_probability",
    "clean_top1_is_open",
    "clean_top1_is_close",
    "clean_best_open_rank_normalized",
    "clean_best_close_rank_normalized",
)


def _validated_token_ids(ids: Iterable[int], *, vocab_size: int, name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in ids)
    if not values:
        raise ValueError(f"{name} token ids cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} token ids contain duplicates")
    if any(value < 0 or value >= vocab_size for value in values):
        raise ValueError(f"{name} token id outside vocabulary")
    return values


def summarize_clean_gripper_logits(
    logits: torch.Tensor,
    *,
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return the nine finite, deployment-safe gripper intent features."""

    if logits.ndim < 1 or logits.shape[-1] < 2 or not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point with shape [..., vocab]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    vocab_size = int(logits.shape[-1])
    open_ids = _validated_token_ids(open_token_ids, vocab_size=vocab_size, name="open")
    close_ids = _validated_token_ids(close_token_ids, vocab_size=vocab_size, name="close")
    if set(open_ids) & set(close_ids):
        raise ValueError("OPEN and CLOSE token sets must be disjoint")

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    open_index = torch.tensor(open_ids, device=logits.device)
    close_index = torch.tensor(close_ids, device=logits.device)
    open_log_mass = torch.logsumexp(log_probs.index_select(-1, open_index), dim=-1)
    close_log_mass = torch.logsumexp(log_probs.index_select(-1, close_index), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1) / math.log(vocab_size)
    top1_probability, top1_token = probs.max(dim=-1)

    open_membership = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    close_membership = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    open_membership[open_index] = True
    close_membership[close_index] = True
    descending = torch.argsort(logits, dim=-1, descending=True)
    inverse_rank = torch.argsort(descending, dim=-1)
    rank_denominator = float(max(1, vocab_size - 1))

    return {
        "clean_open_probability_mass": open_log_mass.exp(),
        "clean_close_probability_mass": close_log_mass.exp(),
        "clean_open_minus_close_log_mass": open_log_mass - close_log_mass,
        "clean_action_token_entropy_normalized": entropy,
        "clean_top1_probability": top1_probability,
        "clean_top1_is_open": open_membership[top1_token].to(logits.dtype),
        "clean_top1_is_close": close_membership[top1_token].to(logits.dtype),
        "clean_best_open_rank_normalized": inverse_rank.index_select(-1, open_index).min(dim=-1).values.to(logits.dtype) / rank_denominator,
        "clean_best_close_rank_normalized": inverse_rank.index_select(-1, close_index).min(dim=-1).values.to(logits.dtype) / rank_denominator,
    }


def policy_intent_9d(
    logits: torch.Tensor,
    *,
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
) -> list[float]:
    summary = summarize_clean_gripper_logits(
        logits,
        open_token_ids=open_token_ids,
        close_token_ids=close_token_ids,
    )
    values = [float(summary[name].detach().cpu()) for name in CLEAN_POLICY_FEATURE_NAMES]
    if len(values) != 9 or not np.isfinite(np.asarray(values, dtype=np.float32)).all():
        raise ValueError("clean policy intent must be a finite 9D vector")
    return values


def derive_gripper_token_semantics(model: Any, unnorm_key: str) -> dict[str, Any]:
    """Derive OPEN/CLOSE token sets from the pinned executable action decoder."""

    centers = np.asarray(model.bin_centers, dtype=np.float32).reshape(-1)
    stats = model.get_action_stats(unnorm_key)
    low = np.asarray(stats["q01"], dtype=np.float32).reshape(-1)
    high = np.asarray(stats["q99"], dtype=np.float32).reshape(-1)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool).reshape(-1)
    if low.size == 0 or high.shape != low.shape or mask.shape != low.shape:
        raise ValueError("invalid OpenVLA action statistics")
    index = low.size - 1
    decoded = 0.5 * (centers + 1.0) * (high[index] - low[index]) + low[index] if mask[index] else centers
    vocab_size = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    token_action_map = {int(vocab_size - i - 1): float(value) for i, value in enumerate(decoded)}
    open_ids = tuple(sorted(token for token, value in token_action_map.items() if value > 0.5))
    close_ids = tuple(sorted(token for token, value in token_action_map.items() if value <= 0.5))
    if not open_ids or not close_ids:
        raise ValueError("could not derive non-empty OPEN/CLOSE token sets")
    return {
        "open_token_ids": open_ids,
        "close_token_ids": close_ids,
        "token_action_map": token_action_map,
    }


def top_token_evidence(logits: torch.Tensor, *, top_k: int = 16) -> tuple[list[int], list[float]]:
    values, ids = torch.topk(logits.float().reshape(-1), k=min(int(top_k), int(logits.numel())))
    return ids.detach().cpu().tolist(), values.detach().cpu().tolist()


__all__ = [
    "CANONICAL_25D_FEATURES",
    "CLEAN_POLICY_FEATURE_NAMES",
    "SC5StreamingFeatureAdapterV2",
    "derive_gripper_token_semantics",
    "policy_intent_9d",
    "summarize_clean_gripper_logits",
    "top_token_evidence",
]
