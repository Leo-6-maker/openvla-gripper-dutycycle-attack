"""Factorized Student loss (Gate S4.2).

Batched masked BCE per head, event-mean → episode-mean → route-mean.
Supports fold-only class weights per route per head.
Consistency: ReLU(p_manipulation - p_grasp), NaN-safe.
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
        logits: dict[str, Tensor],
        episodes: list[FactorizedEpisode],
        valid_mask: Tensor | None = None,
        class_weights: dict[str, dict[str, float]] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """class_weights: {head_name: {"pos_weight": float, "neg_weight": float}}"""
        B = len(episodes)
        if logits["grasp"].shape[0] != B:
            raise ValueError(f"batch size mismatch: logits={logits['grasp'].shape[0]} episodes={B}")

        device = logits["grasp"].device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {"grasp": 0.0, "manipulation": 0.0, "release": 0.0, "consistency": 0.0}

        for b in range(B):
            ep = episodes[b]
            T_ep = len(ep.features_25d)

            g_logits = logits["grasp"][b, :T_ep]
            m_logits = logits["manipulation"][b, :T_ep]
            r_logits = logits["release"][b, :T_ep]

            g_target = ep.grasp_target.to(device); g_mask = ep.grasp_known_mask.to(device)
            m_target = ep.manipulation_target.to(device); m_mask = ep.manipulation_known_mask.to(device)
            r_target = ep.release_target.to(device); r_mask = ep.release_known_mask.to(device)
            eids = ep.event_id.to(device)

            g_pw = class_weights.get("grasp", {}).get("pos_weight") if class_weights else None
            g_nw = class_weights.get("grasp", {}).get("neg_weight") if class_weights else None
            m_pw = class_weights.get("manipulation", {}).get("pos_weight") if class_weights else None
            m_nw = class_weights.get("manipulation", {}).get("neg_weight") if class_weights else None
            r_pw = class_weights.get("release", {}).get("pos_weight") if class_weights else None
            r_nw = class_weights.get("release", {}).get("neg_weight") if class_weights else None

            g_loss = self._head_loss(g_logits, g_target, g_mask, eids, g_pw, g_nw)
            m_loss = self._head_loss(m_logits, m_target, m_mask, eids, m_pw, m_nw)
            r_loss = self._head_loss(r_logits, r_target, r_mask, eids, r_pw, r_nw)

            p_g = torch.sigmoid(g_logits); p_m = torch.sigmoid(m_logits)
            consistency = torch.relu(p_m - p_g)[g_mask].mean() if g_mask.any() else torch.tensor(0.0, device=device)

            ep_loss = g_loss + m_loss + r_loss + self.consistency_weight * consistency
            total_loss = total_loss + ep_loss
            metrics["grasp"] += g_loss.item(); metrics["manipulation"] += m_loss.item()
            metrics["release"] += r_loss.item(); metrics["consistency"] += consistency.item()

        total_loss = total_loss / B
        for k in metrics: metrics[k] /= B
        return total_loss, metrics

    def _head_loss(
        self, logits: Tensor, targets: Tensor, known_mask: Tensor, event_ids: Tensor,
        pos_weight: float | None = None, neg_weight: float | None = None,
    ) -> Tensor:
        """Per-episode: step masked BCE → weighted event-mean → episode-mean."""
        bce = self.bce(logits, targets.float())
        unique_events = event_ids[known_mask].unique()
        if len(unique_events) == 0:
            return torch.tensor(0.0, device=logits.device)

        event_losses = []
        for eid in unique_events.tolist():
            em = known_mask & (event_ids == eid)
            if not em.any():
                continue
            has_pos = (targets[em] > 0.5).any()
            w = pos_weight if (has_pos and pos_weight is not None) else (neg_weight if neg_weight is not None else 1.0)
            event_losses.append((bce[em] * w).mean())

        return torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=logits.device)


__all__ = ["FactorizedLoss"]
