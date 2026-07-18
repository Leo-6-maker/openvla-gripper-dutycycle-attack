"""Causal multimodal V5 ranker and FIT-only ranking losses.

The model consumes precomputed causal Student streams.  It has no interface
for Teacher fields, privileged state, future frames, or attack outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .v5_protocol import V5ModelContract, variant_uses_intent, variant_uses_visual


@dataclass(frozen=True)
class V5RankerConfig:
    contract: V5ModelContract
    pairwise_margin: float = 0.2
    hard_negative_weight: float = 1.0
    abstention_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.pairwise_margin <= 0 or self.hard_negative_weight < 0 or self.abstention_weight < 0:
            raise ValueError("V5 loss weights and margin must be non-negative")


class CausalMultimodalVulnerabilityRanker(nn.Module):
    """Small recurrent fusion model; hidden state is causal and mask-aware."""

    def __init__(self, config: V5ModelContract):
        super().__init__()
        self.config = config
        self.proprio_cell = nn.GRUCell(25, config.hidden_dim)
        self.intent_cell = nn.GRUCell(9, config.intent_hidden_dim) if variant_uses_intent(config.variant) else None
        self.visual_cell = nn.GRUCell(config.visual_dim, config.hidden_dim) if variant_uses_visual(config.variant) else None
        branch_count = 1 + int(self.intent_cell is not None) + int(self.visual_cell is not None)
        branch_dim = config.hidden_dim + (config.intent_hidden_dim if self.intent_cell is not None else 0) + (config.hidden_dim if self.visual_cell is not None else 0)
        self.branch_projection = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.hidden_dim)]
            + ([nn.Linear(config.intent_hidden_dim, config.hidden_dim)] if self.intent_cell is not None else [])
            + ([nn.Linear(config.hidden_dim, config.hidden_dim)] if self.visual_cell is not None else [])
        )
        self.gate = nn.Linear(branch_dim, branch_count)
        self.fusion = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh())
        self.utility_head = nn.Linear(config.hidden_dim, 1)
        self.release_head = nn.Linear(config.hidden_dim, 1)
        self.regrasp_head = nn.Linear(config.hidden_dim, 1)
        self.support_head = nn.Linear(config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)

    def initial_hidden(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor | None, Tensor | None]:
        h_p = torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=dtype)
        h_i = torch.zeros(batch_size, self.config.intent_hidden_dim, device=device, dtype=dtype) if self.intent_cell is not None else None
        h_v = torch.zeros(batch_size, self.config.hidden_dim, device=device, dtype=dtype) if self.visual_cell is not None else None
        return h_p, h_i, h_v

    @staticmethod
    def _check(value: Tensor, width: int, name: str) -> Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape [B,{width}], got {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
        return value

    def step(
        self,
        proprio: Tensor,
        intent: Tensor | None = None,
        visual: Tensor | None = None,
        hidden: tuple[Tensor, Tensor | None, Tensor | None] | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], tuple[Tensor, Tensor | None, Tensor | None]]:
        proprio = self._check(proprio, 25, "proprio")
        batch = proprio.shape[0]
        if self.intent_cell is not None:
            if intent is None:
                raise ValueError("this V5 variant requires policy-intent input")
            intent = self._check(intent, 9, "intent")
        elif intent is not None:
            raise ValueError("proprio-only V5 variant must not receive policy-intent input")
        if self.visual_cell is not None:
            if visual is None:
                raise ValueError("this V5 variant requires causal visual input")
            visual = self._check(visual, self.config.visual_dim, "visual")
        elif visual is not None:
            raise ValueError("non-visual V5 variant must not receive visual input")
        if hidden is None:
            hidden = self.initial_hidden(batch, device=proprio.device, dtype=proprio.dtype)
        h_p, h_i, h_v = hidden
        valid = torch.ones(batch, dtype=torch.bool, device=proprio.device) if valid_mask is None else valid_mask.to(device=proprio.device)
        if valid.ndim != 1 or valid.shape[0] != batch or valid.dtype != torch.bool:
            raise TypeError("valid_mask must be bool with shape [B]")
        new_p = self.proprio_cell(proprio, h_p)
        kept_p = torch.where(valid[:, None], new_p, h_p)
        reps = [kept_p]
        next_i = h_i
        if self.intent_cell is not None:
            assert h_i is not None and intent is not None
            new_i = self.intent_cell(intent, h_i)
            next_i = torch.where(valid[:, None], new_i, h_i)
            reps.append(next_i)
        next_v = h_v
        if self.visual_cell is not None:
            assert h_v is not None and visual is not None
            new_v = self.visual_cell(visual, h_v)
            next_v = torch.where(valid[:, None], new_v, h_v)
            reps.append(next_v)
        projected = [layer(value) for layer, value in zip(self.branch_projection, reps)]
        concat = torch.cat(reps, dim=-1)
        gates = torch.softmax(self.gate(concat), dim=-1)
        fused = sum(gates[:, index:index + 1] * projected[index] for index in range(len(projected)))
        fused = self.fusion(fused)
        return {
            "utility_logit": self.utility_head(fused).squeeze(-1),
            "release_logit": self.release_head(fused).squeeze(-1),
            "regrasp_logit": self.regrasp_head(fused).squeeze(-1),
            "support_logit": self.support_head(fused).squeeze(-1),
            "uncertainty_logit": self.uncertainty_head(fused).squeeze(-1),
        }, (kept_p, next_i, next_v)

    def forward_sequence(
        self,
        proprio: Tensor,
        intent: Tensor | None = None,
        visual: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 3 or proprio.shape[-1] != 25:
            raise ValueError("proprio must have shape [B,T,25]")
        batch, steps, _ = proprio.shape
        for name, value, width in (("intent", intent, 9), ("visual", visual, self.config.visual_dim)):
            required = (name == "intent" and self.intent_cell is not None) or (name == "visual" and self.visual_cell is not None)
            if required and value is None:
                raise ValueError(f"{name} stream is required")
            if value is not None and (value.ndim != 3 or tuple(value.shape[:2]) != (batch, steps) or value.shape[-1] != width):
                raise ValueError(f"{name} must have shape [B,T,{width}]")
        if valid_mask is None:
            valid_mask = torch.ones(batch, steps, dtype=torch.bool, device=proprio.device)
        outputs: dict[str, list[Tensor]] = defaultdict(list)
        hidden = None
        for step in range(steps):
            row, hidden = self.step(
                proprio[:, step],
                None if intent is None else intent[:, step],
                None if visual is None else visual[:, step],
                hidden,
                valid_mask[:, step],
            )
            for name, value in row.items():
                outputs[name].append(value)
        return {name: torch.stack(values, dim=1) for name, values in outputs.items()}


def _group_windows(windows: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        groups[str(window["episode_id"])].append(index)
    return groups


def v5_window_ranking_loss(
    utility_logits: Tensor,
    windows: Sequence[Mapping[str, object]],
    *,
    margin: float = 0.2,
    abstention_weight: float = 1.0,
) -> Tensor:
    """Differentiable within-episode utility ranking with pure-negative abstention."""

    if utility_logits.ndim != 1 or utility_logits.shape[0] != len(windows):
        raise ValueError("utility_logits must be [window_count]")
    if margin <= 0:
        raise ValueError("margin must be positive")
    losses: list[Tensor] = []
    for indices in _group_windows(windows).values():
        known = [i for i in indices if bool(windows[i].get("known", False))]
        positive = [i for i in known if int(windows[i].get("utility_tier", 0)) >= 2]
        negative = [i for i in known if int(windows[i].get("utility_tier", 0)) <= 1]
        if positive and negative:
            pos = utility_logits[positive].max()
            neg = utility_logits[negative].max()
            losses.append(F.relu(torch.as_tensor(margin, device=utility_logits.device, dtype=utility_logits.dtype) - pos + neg))
        elif not positive and known:
            # A pure-negative episode must have no high utility window.
            losses.append(abstention_weight * F.softplus(utility_logits[known]).mean())
    if not losses:
        return utility_logits.sum() * 0.0
    return torch.stack(losses).mean()


__all__ = ["V5RankerConfig", "CausalMultimodalVulnerabilityRanker", "v5_window_ranking_loss"]
