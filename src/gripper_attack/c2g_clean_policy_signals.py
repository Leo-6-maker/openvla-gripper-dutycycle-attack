"""Clean OpenVLA gripper-policy signal extraction for Detector v2.

All features are computed from the clean forward pass before any visual attack or
adversarial re-decode. Token groups must be supplied by the repository's audited
OpenVLA action-token semantics; this module never guesses OPEN/CLOSE polarity.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
from torch import Tensor


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
    logits: Tensor,
    *,
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
) -> Dict[str, Tensor]:
    """Return deployment-safe clean-forward gripper statistics."""

    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [..., vocab] with vocab >= 2")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must be floating point")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    vocab_size = int(logits.shape[-1])
    open_ids = _validated_token_ids(open_token_ids, vocab_size=vocab_size, name="open")
    close_ids = _validated_token_ids(close_token_ids, vocab_size=vocab_size, name="close")
    overlap = set(open_ids) & set(close_ids)
    if overlap:
        raise ValueError("OPEN and CLOSE token sets must be disjoint")

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    open_index = torch.tensor(open_ids, device=logits.device)
    close_index = torch.tensor(close_ids, device=logits.device)
    open_log_mass = torch.logsumexp(log_probs.index_select(-1, open_index), dim=-1)
    close_log_mass = torch.logsumexp(log_probs.index_select(-1, close_index), dim=-1)
    open_mass = open_log_mass.exp()
    close_mass = close_log_mass.exp()
    entropy = -(probs * log_probs).sum(dim=-1) / math.log(vocab_size)
    top1_probability, top1_token = probs.max(dim=-1)

    open_membership = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    close_membership = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    open_membership[open_index] = True
    close_membership[close_index] = True
    top1_is_open = open_membership[top1_token].to(logits.dtype)
    top1_is_close = close_membership[top1_token].to(logits.dtype)

    descending = torch.argsort(logits, dim=-1, descending=True)
    inverse_rank = torch.argsort(descending, dim=-1)
    best_open_rank = inverse_rank.index_select(-1, open_index).min(dim=-1).values.to(logits.dtype)
    best_close_rank = inverse_rank.index_select(-1, close_index).min(dim=-1).values.to(logits.dtype)
    rank_denominator = float(max(1, vocab_size - 1))

    return {
        "clean_open_probability_mass": open_mass,
        "clean_close_probability_mass": close_mass,
        "clean_open_minus_close_log_mass": open_log_mass - close_log_mass,
        "clean_action_token_entropy_normalized": entropy,
        "clean_top1_probability": top1_probability,
        "clean_top1_is_open": top1_is_open,
        "clean_top1_is_close": top1_is_close,
        "clean_best_open_rank_normalized": best_open_rank / rank_denominator,
        "clean_best_close_rank_normalized": best_close_rank / rank_denominator,
    }


def clean_policy_feature_tensor(
    logits: Tensor,
    *,
    open_token_ids: Sequence[int],
    close_token_ids: Sequence[int],
) -> Tensor:
    """Stack clean policy statistics in the frozen feature order."""

    summary = summarize_clean_gripper_logits(
        logits,
        open_token_ids=open_token_ids,
        close_token_ids=close_token_ids,
    )
    return torch.stack([summary[name] for name in CLEAN_POLICY_FEATURE_NAMES], dim=-1)
