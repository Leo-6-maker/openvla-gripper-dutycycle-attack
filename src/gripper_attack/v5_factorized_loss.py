"""Factorized Student loss (Gate S4).

Masked BCE per head, aggregated: event → episode → route.
Consistency: ReLU(p_manipulation - p_grasp).
All class weights computed from current training fold only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .v5_factorized_dataset import FactorizedEpisode, SUPPORTED_ROUTES


class FactorizedLoss(nn.Module):
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    def forward(
        self,
        logits: dict[str, Tensor],  # grasp/manipulation/release each [T] or [B,T]
        episode: FactorizedEpisode,
    ) -> tuple[Tensor, dict[str, float]]:
        g_logits = logits["grasp"].squeeze(0) if logits["grasp"].ndim == 2 else logits["grasp"]
        m_logits = logits["manipulation"].squeeze(0) if logits["manipulation"].ndim == 2 else logits["manipulation"]
        r_logits = logits["release"].squeeze(0) if logits["release"].ndim == 2 else logits["release"]

        g_loss = self._head_loss(g_logits, episode.grasp_target, episode.grasp_known_mask,
                                  episode.event_id)
        m_loss = self._head_loss(m_logits, episode.manipulation_target, episode.manipulation_known_mask,
                                  episode.event_id)
        r_loss = self._head_loss(r_logits, episode.release_target, episode.release_known_mask,
                                  episode.event_id)

        p_g = torch.sigmoid(g_logits)
        p_m = torch.sigmoid(m_logits)
        consistency = torch.relu(p_m - p_g)[episode.grasp_known_mask].mean()

        total = g_loss + m_loss + r_loss + self.consistency_weight * consistency
        return total, {
            "grasp": g_loss.item(), "manipulation": m_loss.item(),
            "release": r_loss.item(), "consistency": consistency.item(),
        }

    def _head_loss(
        self, logits: Tensor, targets: Tensor, known_mask: Tensor, event_ids: Tensor,
    ) -> Tensor:
        """Masked BCE, averaged: event-mean → episode-mean.

        Each event contributes equally regardless of duration.
        Background known negatives (event_id=-1) contribute as one unit.
        """
        bce = self.bce(logits, targets.float()) * known_mask.float()
        unique_events = event_ids[known_mask].unique()
        if len(unique_events) == 0:
            return torch.tensor(0.0, device=logits.device)

        event_losses = []
        for eid in unique_events.tolist():
            if eid < 0:
                # Background negatives → single unit
                bg_mask = known_mask & (event_ids == eid)
                if bg_mask.any():
                    event_losses.append(bce[bg_mask].mean())
            else:
                event_mask = known_mask & (event_ids == eid)
                if event_mask.any():
                    event_losses.append(bce[event_mask].mean())
        return torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=logits.device)


__all__ = ["FactorizedLoss"]
