"""Recommended sidecar Student V2B implementation.

This module is intentionally separate from the frozen 864-job Stage-1 code.
It provides:
- exact causal context (W=16/32/64 means exactly that many steps),
- event-balanced release loss with top-k auxiliary,
- route-specific class weights that are actually consumed,
- valid-mask intersection so temporally jittered prefixes are excluded.

Artifacts produced with this module are engineering sidecar evidence and must not
be mixed into the frozen Stage-1 selection namespace.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from .v5_factorized_dataset import FactorizedEpisode
from .v5_factorized_student_v2 import FactorizedStudentV2


class ExactCausalTCNEncoder(nn.Module):
    """Causal TCN whose receptive field equals ``context_steps`` exactly.

    For power-of-two W, kernel_size=2 and dilations 1,2,...,W/2 give
    receptive field 1 + sum(dilations) = W.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64,
                 context_steps: int = 32, dropout: float = 0.1):
        super().__init__()
        if context_steps not in {16, 32, 64}:
            raise ValueError(f"context_steps must be one of 16/32/64, got {context_steps}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_steps = context_steps

        layers: list[nn.Module] = []
        dilation = 1
        current_rf = 1
        while current_rf < context_steps:
            layers.append(nn.Conv1d(
                hidden_dim if layers else input_dim,
                hidden_dim,
                kernel_size=2,
                dilation=dilation,
                padding=0,
            ))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_rf += dilation
            dilation *= 2

        if current_rf != context_steps:
            raise AssertionError(f"exact receptive field construction failed: {current_rf}")
        self.actual_receptive_field = current_rf
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], got {tuple(x.shape)}")
        T = x.shape[1]
        out = x.transpose(1, 2)
        for module in self.conv_stack:
            if isinstance(module, nn.Conv1d):
                pad = (module.kernel_size[0] - 1) * module.dilation[0]
                out = nn.functional.pad(out, (pad, 0))
            out = module(out)
        out = out.transpose(1, 2)
        if out.shape[1] != T:
            raise AssertionError(f"causal length changed: {out.shape[1]} != {T}")
        return out


class RecommendedFactorizedStudentV2(FactorizedStudentV2):
    """V2B Student with exact causal context and frozen 25D-only input."""

    def __init__(self, input_dim_25d: int = 25, hidden_dim: int = 64,
                 context_steps: int = 32, dropout: float = 0.1):
        super().__init__(
            input_dim_25d=input_dim_25d,
            hidden_dim=hidden_dim,
            receptive_field=context_steps,
            encoder_type="tcn",
            dropout=dropout,
            use_9d=False,
        )
        self.encoder_25d = ExactCausalTCNEncoder(
            input_dim=input_dim_25d,
            hidden_dim=hidden_dim,
            context_steps=context_steps,
            dropout=dropout,
        )
        self.receptive_field = context_steps
        self.encoder_type = "exact_tcn"


def _event_mean_bce(
    bce_fn: nn.Module,
    logits: Tensor,
    targets: Tensor,
    known_mask: Tensor,
    event_ids: Tensor,
    pos_weight: float | None,
    neg_weight: float | None,
) -> Tensor:
    bce = bce_fn(logits, targets.float())
    event_losses: list[Tensor] = []
    for eid in event_ids[known_mask].unique().tolist():
        em = known_mask & (event_ids == eid)
        if not em.any():
            continue
        has_pos = bool((targets[em] > 0.5).any())
        if has_pos and pos_weight is not None:
            weight = float(pos_weight)
        elif (not has_pos) and neg_weight is not None:
            weight = float(neg_weight)
        else:
            weight = 1.0
        event_losses.append((bce[em] * weight).mean())
    if not event_losses:
        return logits.new_tensor(0.0)
    return torch.stack(event_losses).mean()


class RecommendedEventBalancedLoss(nn.Module):
    """Corrected V2B loss for the recommended sidecar canary.

    ``class_weights`` is the route-specific head map:
    ``{"grasp": {...}, "manipulation": {...}, "release": {...}}``.
    ``valid_mask`` is intersected with every Teacher known mask.
    """

    DURATION_BUCKETS = ((0, 15), (15, 30), (30, 50), (50, 100), (100, 99999))
    TOP_K = 3
    AUX_WEIGHT = 0.3

    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    @staticmethod
    def _weights(class_weights: dict | None, head: str) -> tuple[float | None, float | None]:
        head_weights = (class_weights or {}).get(head, {})
        return head_weights.get("pos_weight"), head_weights.get("neg_weight")

    def _release_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        known_mask: Tensor,
        event_ids: Tensor,
        pos_weight: float | None,
        neg_weight: float | None,
    ) -> tuple[Tensor, Tensor, dict]:
        bce = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        event_losses: list[Tensor] = []
        aux_losses: list[Tensor] = []
        audit = defaultdict(lambda: {"event_count": 0, "loss_sum": 0.0})

        for eid in event_ids[known_mask].unique().tolist():
            em = known_mask & (event_ids == eid)
            if not em.any():
                continue
            has_pos = bool((targets[em] > 0.5).any())
            if has_pos and pos_weight is not None:
                weight = float(pos_weight)
            elif (not has_pos) and neg_weight is not None:
                weight = float(neg_weight)
            else:
                weight = 1.0

            event_bce = (bce[em] * weight).mean()
            event_losses.append(event_bce)

            event_probs = probs[em]
            k = min(self.TOP_K, int(event_probs.numel()))
            top_k_mean = torch.topk(event_probs, k).values.mean()
            event_target = logits.new_tensor(1.0 if has_pos else 0.0)
            aux_losses.append(nn.functional.binary_cross_entropy(
                top_k_mean.clamp(1e-7, 1 - 1e-7), event_target))

            duration = int(em.sum().item())
            for lo, hi in self.DURATION_BUCKETS:
                if lo <= duration < hi:
                    key = f"{lo}_{hi}"
                    audit[key]["event_count"] += 1
                    audit[key]["loss_sum"] += float(event_bce.item())
                    break

        event_mean = torch.stack(event_losses).mean() if event_losses else logits.new_tensor(0.0)
        aux_mean = torch.stack(aux_losses).mean() if aux_losses else logits.new_tensor(0.0)
        return event_mean, aux_mean, dict(audit)

    def forward(
        self,
        logits: dict[str, Tensor],
        episodes: list[FactorizedEpisode],
        valid_mask: Tensor | None = None,
        class_weights: dict | None = None,
        identity_weights: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float], dict]:
        B = len(episodes)
        device = logits["grasp"].device
        total_loss = logits["grasp"].new_tensor(0.0)
        metrics = {
            "grasp": 0.0,
            "manipulation": 0.0,
            "release": 0.0,
            "release_aux": 0.0,
            "consistency": 0.0,
        }
        audit = {
            "duration_buckets": {
                f"{lo}_{hi}": {"event_count": 0, "loss_sum": 0.0}
                for lo, hi in self.DURATION_BUCKETS
            },
            "identity_losses": [],
        }

        for b, ep in enumerate(episodes):
            T = len(ep.features_25d)
            vm = (valid_mask[b, :T].to(device) if valid_mask is not None
                  else torch.ones(T, dtype=torch.bool, device=device))
            eids = ep.event_id.to(device)
            g_target = ep.grasp_target.to(device)
            m_target = ep.manipulation_target.to(device)
            r_target = ep.release_target.to(device)
            g_mask = ep.grasp_known_mask.to(device) & vm
            m_mask = ep.manipulation_known_mask.to(device) & vm
            r_mask = ep.release_known_mask.to(device) & vm

            g_pw, g_nw = self._weights(class_weights, "grasp")
            m_pw, m_nw = self._weights(class_weights, "manipulation")
            r_pw, r_nw = self._weights(class_weights, "release")

            g_logits = logits["grasp"][b, :T]
            m_logits = logits["manipulation"][b, :T]
            r_logits = logits["release"][b, :T]

            g_loss = _event_mean_bce(self.bce, g_logits, g_target, g_mask, eids, g_pw, g_nw)
            m_loss = _event_mean_bce(self.bce, m_logits, m_target, m_mask, eids, m_pw, m_nw)
            r_loss, r_aux, bucket_audit = self._release_loss(
                r_logits, r_target, r_mask, eids, r_pw, r_nw)

            consistency_mask = g_mask & m_mask
            if consistency_mask.any():
                consistency = torch.relu(
                    torch.sigmoid(m_logits) - torch.sigmoid(g_logits)
                )[consistency_mask].mean()
            else:
                consistency = logits["grasp"].new_tensor(0.0)

            ep_loss = (
                g_loss + m_loss + r_loss
                + self.AUX_WEIGHT * r_aux
                + self.consistency_weight * consistency
            )
            if identity_weights is not None:
                ep_loss = ep_loss * identity_weights[b]
            total_loss = total_loss + ep_loss

            metrics["grasp"] += float(g_loss.item())
            metrics["manipulation"] += float(m_loss.item())
            metrics["release"] += float(r_loss.item())
            metrics["release_aux"] += float(r_aux.item())
            metrics["consistency"] += float(consistency.item())
            audit["identity_losses"].append(float(ep_loss.item()))
            for key, value in bucket_audit.items():
                audit["duration_buckets"][key]["event_count"] += value["event_count"]
                audit["duration_buckets"][key]["loss_sum"] += value["loss_sum"]

        total_loss = total_loss / max(1, B)
        for key in metrics:
            metrics[key] /= max(1, B)
        return total_loss, metrics, audit


__all__ = [
    "ExactCausalTCNEncoder",
    "RecommendedFactorizedStudentV2",
    "RecommendedEventBalancedLoss",
]
