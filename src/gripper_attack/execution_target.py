from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .openvla_libero_exec_spec import raw_gripper_is_close, raw_gripper_is_open, raw_gripper_to_env_gripper


TARGET_31744 = 31744
TARGET_31744_EXECUTION_CLASS = "CLIP_MEDIATED_OPEN"


@dataclass(frozen=True)
class TokenExecution:
    token_id: int
    disc_before: int
    disc_after: int
    clipped: bool
    decoded_raw_gripper: float
    executed_env_gripper: float
    execution_class: str


def native_action_token_ids(*, vocab_eff: int, n_bins: int) -> set[int]:
    start = max(0, int(vocab_eff) - int(n_bins))
    return set(range(start, int(vocab_eff)))


def official_generation_token_ids(row_or_vocab_size: int | torch.Tensor) -> list[int]:
    if torch.is_tensor(row_or_vocab_size):
        size = int(row_or_vocab_size.numel())
    else:
        size = int(row_or_vocab_size)
    return list(range(max(size, 0)))


def classify_execution_token(
    token_id: int,
    *,
    vocab_eff: int,
    n_bins: int,
    bin_centers: Iterable[float],
    action_stats: Mapping[str, Any],
) -> TokenExecution:
    token_id = int(token_id)
    vocab_eff = int(vocab_eff)
    n_bins = int(n_bins)
    centers = np.asarray(list(bin_centers), dtype=np.float32)
    if centers.ndim != 1 or centers.size != n_bins:
        raise ValueError("bin_centers size must match n_bins")
    disc_before = int(vocab_eff - token_id - 1)
    disc_after = max(0, min(n_bins - 1, disc_before))
    clipped = disc_before != disc_after
    low = np.asarray(action_stats["q01"], dtype=np.float32)
    high = np.asarray(action_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    gripper_dim = len(low) - 1
    center = float(centers[disc_after])
    if bool(mask[gripper_dim]):
        raw = float(0.5 * (center + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim])
    else:
        raw = float(center)
    env = float(raw_gripper_to_env_gripper(raw))
    if clipped:
        if env < -0.5:
            execution_class = "CLIP_MEDIATED_OPEN"
        elif env > 0.5:
            execution_class = "CLIP_MEDIATED_CLOSE"
        else:
            execution_class = "CLIP_MEDIATED_NEUTRAL"
    elif abs(raw - 0.5) <= 1e-9:
        execution_class = "NATIVE_BOUNDARY"
    elif raw_gripper_is_open(raw):
        execution_class = "NATIVE_OPEN"
    elif raw_gripper_is_close(raw):
        execution_class = "NATIVE_CLOSE"
    else:
        execution_class = "NATIVE_UNCLASSIFIED"
    return TokenExecution(
        token_id=token_id,
        disc_before=disc_before,
        disc_after=disc_after,
        clipped=bool(clipped),
        decoded_raw_gripper=raw,
        executed_env_gripper=env,
        execution_class=execution_class,
    )


def validate_execution_target(
    *,
    token_id: int,
    expected_execution_class: str,
    vocab_eff: int,
    n_bins: int,
    bin_centers: Iterable[float],
    action_stats: Mapping[str, Any],
) -> TokenExecution:
    execution = classify_execution_token(
        token_id,
        vocab_eff=vocab_eff,
        n_bins=n_bins,
        bin_centers=bin_centers,
        action_stats=action_stats,
    )
    if execution.execution_class != str(expected_execution_class):
        raise ValueError(
            f"target token {int(token_id)} class {execution.execution_class}, "
            f"expected {expected_execution_class}"
        )
    if execution.execution_class.startswith("CLIP_MEDIATED") and int(token_id) in native_action_token_ids(vocab_eff=vocab_eff, n_bins=n_bins):
        raise ValueError(f"clip-mediated target token {int(token_id)} is inside native action-bin range")
    return execution


def target_token_cw_loss_and_stats(
    row: torch.Tensor,
    *,
    target_token_id: int,
    allowed_token_ids: Iterable[int] | None = None,
    margin: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    row = row.float()
    target = int(target_token_id)
    if target < 0 or target >= int(row.numel()):
        raise ValueError(f"target token {target} is outside score row length {int(row.numel())}")
    if allowed_token_ids is None:
        allowed = torch.arange(row.numel(), device=row.device, dtype=torch.long)
    else:
        allowed = torch.tensor([int(x) for x in allowed_token_ids], device=row.device, dtype=torch.long)
    if allowed.numel() == 0:
        raise ValueError("allowed_token_ids must not be empty")
    allowed = allowed[(allowed >= 0) & (allowed < row.numel())]
    if not torch.any(allowed == target):
        allowed = torch.cat([allowed, torch.tensor([target], device=row.device, dtype=torch.long)])
    competitor = allowed[allowed != target]
    if competitor.numel() == 0:
        raise ValueError("target-token CW loss needs at least one competitor")
    target_score = row[target]
    competitor_scores = row[competitor]
    best_score, best_rel = torch.max(competitor_scores, dim=0)
    best_token = int(competitor[int(best_rel.detach().cpu())].detach().cpu())
    loss = F.relu(best_score - target_score + float(margin))
    return loss, {
        "target_token_id": int(target),
        "target_token_score": float(target_score.detach().cpu()),
        "best_competitor_token_id": best_token,
        "best_competitor_score": float(best_score.detach().cpu()),
        "target_minus_best_competitor_margin": float((target_score - best_score).detach().cpu()),
        "target_objective_margin": float((target_score - best_score).detach().cpu()),
        "target_objective_margin_name": "target_minus_best_competitor_margin",
        "cw_margin_param": float(margin),
        "allowed_token_count": int(allowed.numel()),
    }


def target_token_logratio_loss_and_stats(
    row: torch.Tensor,
    *,
    target_token_id: int,
    allowed_token_ids: Iterable[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Non-saturating target-token objective.

    The loss is ``logsumexp(scores[j != target]) - score[target]`` over the
    official action-position score row. Unlike the CW hinge objective, this has
    no zero-loss plateau once the target clears a fixed margin.
    """

    row = row.float()
    target = int(target_token_id)
    if target < 0 or target >= int(row.numel()):
        raise ValueError(f"target token {target} is outside score row length {int(row.numel())}")
    if allowed_token_ids is None:
        allowed = torch.arange(row.numel(), device=row.device, dtype=torch.long)
    else:
        allowed = torch.tensor([int(x) for x in allowed_token_ids], device=row.device, dtype=torch.long)
    if allowed.numel() == 0:
        raise ValueError("allowed_token_ids must not be empty")
    allowed = allowed[(allowed >= 0) & (allowed < row.numel())]
    if not torch.any(allowed == target):
        allowed = torch.cat([allowed, torch.tensor([target], device=row.device, dtype=torch.long)])
    competitor = allowed[allowed != target]
    if competitor.numel() == 0:
        raise ValueError("target-token log-ratio loss needs at least one competitor")
    target_score = row[target]
    competitor_scores = row[competitor]
    competitor_logsumexp = torch.logsumexp(competitor_scores, dim=0)
    best_score, best_rel = torch.max(competitor_scores, dim=0)
    best_token = int(competitor[int(best_rel.detach().cpu())].detach().cpu())
    loss = competitor_logsumexp - target_score
    return loss, {
        "target_token_id": int(target),
        "target_token_score": float(target_score.detach().cpu()),
        "best_competitor_token_id": best_token,
        "best_competitor_score": float(best_score.detach().cpu()),
        "competitor_logsumexp_score": float(competitor_logsumexp.detach().cpu()),
        "target_minus_best_competitor_margin": float((target_score - best_score).detach().cpu()),
        "target_minus_competitor_logsumexp_margin": float((target_score - competitor_logsumexp).detach().cpu()),
        "target_objective_margin": float((target_score - competitor_logsumexp).detach().cpu()),
        "target_objective_margin_name": "target_minus_competitor_logsumexp_margin",
        "cw_margin_param": None,
        "allowed_token_count": int(allowed.numel()),
    }
