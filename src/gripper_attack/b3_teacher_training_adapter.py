"""Strict, offline-only conversion from B3 Teacher rows to masked targets.

This adapter is preparation code.  It does not train, select, calibrate, or
run a detector.  Missing heads and masks are hard errors; unknown T10 labels
are masked, never converted to negatives.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


HEADS = (
    "grasp_support",
    "retention_active",
    "retention_continuation_t10",
    "release_imminent",
)
MASKS = {
    "grasp_support": "grasp_support_mask",
    "retention_active": "retention_active_mask",
    "retention_continuation_t10": "retention_unknown_mask",
    "release_imminent": "release_imminent_mask",
}
REQUIRED_FIELDS = frozenset(HEADS) | frozenset(MASKS.values())


def _binary(value: Any, field: str) -> float:
    if value not in (False, True, 0, 1, 0.0, 1.0):
        raise ValueError(f"{field} must be binary, got {value!r}")
    return float(bool(value))


def _known(head: str, row: dict[str, Any]) -> bool:
    mask = MASKS[head]
    value = row[mask]
    if not isinstance(value, bool):
        raise ValueError(f"{mask} must be boolean")
    return not value if head == "retention_continuation_t10" else value


def _mask(value: torch.Tensor | None, shape: tuple[int, int], name: str) -> torch.Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value.to(dtype=torch.bool)


def adapt_teacher_batch(
    records: Sequence[Sequence[dict[str, Any] | None]],
    *,
    episode_valid_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return binary targets and effective masks for a rectangular batch.

    ``records[b][t] is None`` is permitted only where the combined episode or
    padding mask is false.  Every valid row must contain all four heads and
    all four masks.  T10 known is defined exactly as
    ``not retention_unknown_mask``.
    """

    batch = len(records)
    steps = max((len(episode) for episode in records), default=0)
    if batch < 1 or steps < 1 or any(len(episode) != steps for episode in records):
        raise ValueError("records must be a non-empty rectangular batch")
    shape = (batch, steps)
    valid = _mask(episode_valid_mask, shape, "episode_valid_mask") & _mask(padding_mask, shape, "padding_mask")
    targets = {head: torch.zeros(shape, dtype=torch.float32) for head in HEADS}
    masks = {head: torch.zeros(shape, dtype=torch.bool) for head in HEADS}

    for batch_index, episode in enumerate(records):
        for step_index, row in enumerate(episode):
            if row is None:
                if bool(valid[batch_index, step_index]):
                    raise ValueError(f"valid row is missing at [{batch_index}, {step_index}]")
                continue
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(f"Teacher row [{batch_index}, {step_index}] missing fields: {missing}")
            for head in HEADS:
                known = _known(head, row)
                value = row[head]
                if known:
                    target = _binary(value, head)
                    if valid[batch_index, step_index]:
                        targets[head][batch_index, step_index] = target
                elif value is not None:
                    raise ValueError(f"unknown {head} must be null")
                masks[head][batch_index, step_index] = bool(valid[batch_index, step_index] and known)
    return targets, masks


def effective_known_mask(
    head: str,
    row: dict[str, Any],
    *,
    episode_valid: bool = True,
    padding_valid: bool = True,
) -> bool:
    """Expose the exact mask semantics for audits and trainer integration."""

    if head not in HEADS:
        raise ValueError(f"unknown B3 head: {head}")
    if not episode_valid or not padding_valid:
        return False
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise ValueError(f"Teacher row missing fields: {missing}")
    return _known(head, row)


__all__ = ["HEADS", "MASKS", "REQUIRED_FIELDS", "adapt_teacher_batch", "effective_known_mask"]
