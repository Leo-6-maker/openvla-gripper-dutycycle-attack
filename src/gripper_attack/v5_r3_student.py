"""Small R3 Student training utilities with explicit UNKNOWN masking."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .v5_r3_teacher import HEADS


MODEL_HEAD_ALIASES = {"k10_feasibility": "k10_feasible"}


def _model_head_name(head: str) -> str:
    return MODEL_HEAD_ALIASES.get(head, head)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor | None:
    if logits.shape != targets.shape or logits.shape != known_mask.shape:
        raise ValueError("logits, targets and known_mask must have identical shapes")
    if not known_mask.any():
        return None
    return F.binary_cross_entropy_with_logits(
        logits[known_mask], targets[known_mask].to(dtype=logits.dtype), reduction="mean"
    )


def r3_multihead_loss(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    known_masks: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float | None]]:
    terms: list[torch.Tensor] = []
    diagnostics: dict[str, float | None] = {}
    for head in HEADS:
        model_head = _model_head_name(head)
        if model_head not in logits or head not in targets or head not in known_masks:
            raise ValueError(f"missing R3 head: {head}")
        loss = masked_bce_loss(logits[model_head], targets[head], known_masks[head])
        diagnostics[head] = None if loss is None else float(loss.detach().cpu())
        if loss is not None:
            terms.append(loss)
    if not terms:
        raise ValueError("all R3 heads are UNKNOWN/masked")
    return torch.stack(terms).mean(), diagnostics


def shuffle_known_targets(
    targets: Mapping[str, torch.Tensor],
    known_masks: Mapping[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    """Deterministic label-shuffle control; masks and UNKNOWN positions stay fixed."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shuffled = {head: value.clone() for head, value in targets.items()}
    for head in HEADS:
        mask = known_masks[head].bool()
        values = targets[head][mask].detach().cpu()
        if values.numel() > 1:
            order = torch.randperm(values.numel(), generator=generator)
            shuffled[head][mask] = values[order].to(device=shuffled[head].device, dtype=shuffled[head].dtype)
    return shuffled


def finite_head_outputs(logits: Mapping[str, torch.Tensor]) -> bool:
    return all(_model_head_name(head) in logits and torch.isfinite(logits[_model_head_name(head)]).all().item() for head in HEADS)
