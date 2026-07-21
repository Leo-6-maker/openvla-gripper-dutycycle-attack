"""Factorized Student loss (Gate S4.1).

Batched masked BCE per head: step masked → event mean → episode mean → route mean.
Each episode computed independently; padding does not affect results.
Consistency: ReLU(p_manipulation - p_grasp) with NaN-safe mean.
All class weights from current training fold only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .v5_factorized_dataset import FactorizedEpisode


class FactorizedLoss(nn.Module):
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    def forward(
        self,
        logits: dict[str, Tensor],       # each [B, T]
        episodes: list[FactorizedEpisode],  # len = B
        valid_mask: Tensor | None = None,   # [B, T] bool, optional padding mask
    ) -> tuple[Tensor, dict[str, float]]:
        B = len(episodes)
        if logits["grasp"].shape[0] != B:
            raise ValueError(f"batch size mismatch: logits={logits['grasp'].shape[0]} episodes={B}")

        total_loss = torch.tensor(0.0, device=logits["grasp"].device)
        metrics = {"grasp": 0.0, "manipulation": 0.0, "release": 0.0, "consistency": 0.0}

        device = logits["grasp"].device
        for b in range(B):
            ep = episodes[b]
            T_ep = len(ep.features_25d)

            g_logits = logits["grasp"][b, :T_ep]
            m_logits = logits["manipulation"][b, :T_ep]
            r_logits = logits["release"][b, :T_ep]

            g_target = ep.grasp_target.to(device)
            g_mask = ep.grasp_known_mask.to(device)
            m_target = ep.manipulation_target.to(device)
            m_mask = ep.manipulation_known_mask.to(device)
            r_target = ep.release_target.to(device)
            r_mask = ep.release_known_mask.to(device)
            eids = ep.event_id.to(device)

            g_loss = self._head_loss(g_logits, g_target, g_mask, eids)
            m_loss = self._head_loss(m_logits, m_target, m_mask, eids)
            r_loss = self._head_loss(r_logits, r_target, r_mask, eids)

            p_g = torch.sigmoid(g_logits)
            p_m = torch.sigmoid(m_logits)
            consistency = torch.relu(p_m - p_g)[g_mask].mean() if g_mask.any() else torch.tensor(0.0, device=device)

            ep_loss = g_loss + m_loss + r_loss + self.consistency_weight * consistency
            total_loss = total_loss + ep_loss
            metrics["grasp"] += g_loss.item()
            metrics["manipulation"] += m_loss.item()
            metrics["release"] += r_loss.item()
            metrics["consistency"] += consistency.item()

        total_loss = total_loss / B
        for k in metrics:
            metrics[k] /= B
        return total_loss, metrics

    def _head_loss(
        self, logits: Tensor, targets: Tensor, known_mask: Tensor, event_ids: Tensor,
    ) -> Tensor:
        """Per-episode: step masked BCE → event-mean → episode-mean."""
        bce = self.bce(logits, targets.float()) * known_mask.float()
        unique_events = event_ids[known_mask].unique()
        if len(unique_events) == 0:
            return torch.tensor(0.0, device=logits.device)

        event_losses = []
        for eid in unique_events.tolist():
            if eid < 0:
                bg_mask = known_mask & (event_ids == eid)
                if bg_mask.any():
                    event_losses.append(bce[bg_mask].mean())
            else:
                event_mask = known_mask & (event_ids == eid)
                if event_mask.any():
                    event_losses.append(bce[event_mask].mean())
        return torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=logits.device)


__all__ = ["FactorizedLoss"]
