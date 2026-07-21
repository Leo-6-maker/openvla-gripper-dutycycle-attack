"""V2 Factorized Student loss functions.

V2A: Same as V1 (masked step BCE per head, event-mean -> episode-mean)
V2B/V2C: Event-balanced release loss + top-k event auxiliary + duration buckets.

All variants retain V1 grasp/manipulation loss and ReLU consistency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from collections import defaultdict

from .v5_factorized_dataset import FactorizedEpisode


class FactorizedLossV2A(nn.Module):
    """V2A loss: same as V1 — masked step BCE, event-mean -> episode-mean."""

    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    def forward(self, logits: dict[str, Tensor], episodes: list[FactorizedEpisode],
                valid_mask: Tensor | None = None,
                class_weights: dict[str, dict[str, float]] | None = None,
                identity_weights: Tensor | None = None,
                ) -> tuple[Tensor, dict[str, float]]:
        B = len(episodes)
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

            cw = class_weights.get(ep.mechanism_route, {}) if class_weights else {}
            g_pw = cw.get("grasp", {}).get("pos_weight")
            g_nw = cw.get("grasp", {}).get("neg_weight")
            m_pw = cw.get("manipulation", {}).get("pos_weight")
            m_nw = cw.get("manipulation", {}).get("neg_weight")
            r_pw = cw.get("release", {}).get("pos_weight")
            r_nw = cw.get("release", {}).get("neg_weight")

            g_loss = self._head_loss(g_logits, g_target, g_mask, eids, g_pw, g_nw)
            m_loss = self._head_loss(m_logits, m_target, m_mask, eids, m_pw, m_nw)
            r_loss = self._head_loss(r_logits, r_target, r_mask, eids, r_pw, r_nw)

            p_g = torch.sigmoid(g_logits); p_m = torch.sigmoid(m_logits)
            consistency = torch.relu(p_m - p_g)[g_mask].mean() if g_mask.any() else torch.tensor(0.0, device=device)

            ep_loss = g_loss + m_loss + r_loss + self.consistency_weight * consistency
            if identity_weights is not None:
                ep_loss = ep_loss * identity_weights[b]
            total_loss = total_loss + ep_loss
            metrics["grasp"] += g_loss.item(); metrics["manipulation"] += m_loss.item()
            metrics["release"] += r_loss.item(); metrics["consistency"] += consistency.item()

        total_loss = total_loss / B
        for k in metrics: metrics[k] /= B
        return total_loss, metrics

    def _head_loss(self, logits, targets, known_mask, event_ids,
                   pos_weight=None, neg_weight=None):
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


class FactorizedLossV2B(nn.Module):
    """V2B/V2C loss: event-balanced release + top-k auxiliary + duration buckets.

    Release loss = event_mean(step_bce) * 0.7 + top_k_aux * 0.3
    Grasp and manipulation unchanged from V1.
    """

    DURATION_BUCKETS = [(0, 15), (15, 30), (30, 50), (50, 100), (100, 99999)]
    TOP_K = 3
    AUX_WEIGHT = 0.3  # frozen per errata E5

    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.consistency_weight = consistency_weight

    def forward(self, logits: dict[str, Tensor], episodes: list[FactorizedEpisode],
                valid_mask: Tensor | None = None,
                class_weights: dict[str, dict[str, float]] | None = None,
                identity_weights: Tensor | None = None,
                ) -> tuple[Tensor, dict[str, float], dict]:
        B = len(episodes)
        device = logits["grasp"].device
        total_loss = torch.tensor(0.0, device=device)
        metrics = {"grasp": 0.0, "manipulation": 0.0, "release": 0.0,
                   "release_aux": 0.0, "consistency": 0.0}
        audit = {"duration_buckets": {f"{lo}_{hi}": {"event_count": 0, "loss_sum": 0.0}
                                      for lo, hi in self.DURATION_BUCKETS},
                 "identity_losses": []}

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

            cw = class_weights.get(ep.mechanism_route, {}) if class_weights else {}
            g_pw = cw.get("grasp", {}).get("pos_weight")
            g_nw = cw.get("grasp", {}).get("neg_weight")
            m_pw = cw.get("manipulation", {}).get("pos_weight")
            m_nw = cw.get("manipulation", {}).get("neg_weight")
            r_pw = cw.get("release", {}).get("pos_weight")
            r_nw = cw.get("release", {}).get("neg_weight")

            # Grasp and manipulation: unchanged V1 per-event BCE
            g_loss = FactorizedLossV2A._head_loss(None, g_logits, g_target, g_mask, eids, g_pw, g_nw)
            m_loss = FactorizedLossV2A._head_loss(None, m_logits, m_target, m_mask, eids, m_pw, m_nw)

            # Release: event-balanced + top-k auxiliary
            r_loss, r_aux, bucket_audit = self._release_loss(
                r_logits, r_target, r_mask, eids, r_pw, r_nw)

            # Consistency
            p_g = torch.sigmoid(g_logits); p_m = torch.sigmoid(m_logits)
            consistency = torch.relu(p_m - p_g)[g_mask].mean() if g_mask.any() else torch.tensor(0.0, device=device)

            ep_loss = g_loss + m_loss + r_loss + self.AUX_WEIGHT * r_aux + self.consistency_weight * consistency
            if identity_weights is not None:
                ep_loss = ep_loss * identity_weights[b]

            total_loss = total_loss + ep_loss
            metrics["grasp"] += g_loss.item()
            metrics["manipulation"] += m_loss.item()
            metrics["release"] += r_loss.item()
            metrics["release_aux"] += r_aux.item()
            metrics["consistency"] += consistency.item()
            audit["identity_losses"].append(float(ep_loss.item()))

            for bk, bv in bucket_audit.items():
                if bk in audit["duration_buckets"]:
                    audit["duration_buckets"][bk]["event_count"] += bv["event_count"]
                    audit["duration_buckets"][bk]["loss_sum"] += bv["loss_sum"]

        total_loss = total_loss / B
        for k in metrics: metrics[k] /= B
        return total_loss, metrics, audit

    def _release_loss(self, logits, targets, known_mask, event_ids,
                      pos_weight=None, neg_weight=None):
        """Event-balanced release loss with top-k auxiliary."""
        bce = self.bce(logits, targets.float())
        probs = torch.sigmoid(logits)
        unique_events = event_ids[known_mask].unique()
        if len(unique_events) == 0:
            return (torch.tensor(0.0, device=logits.device),
                    torch.tensor(0.0, device=logits.device),
                    {})

        # Per-event step BCE
        event_losses = []
        aux_losses = []
        bucket_audit = defaultdict(lambda: {"event_count": 0, "loss_sum": 0.0})

        for eid in unique_events.tolist():
            em = known_mask & (event_ids == eid)
            if not em.any():
                continue
            has_pos = (targets[em] > 0.5).any()
            w = pos_weight if (has_pos and pos_weight is not None) else (neg_weight if neg_weight is not None else 1.0)

            # Step BCE for this event
            event_bce = (bce[em] * w).mean()
            event_losses.append(event_bce)

            # Top-k auxiliary: mean of top-k probs within known mask
            event_probs = probs[em]
            k = min(self.TOP_K, len(event_probs))
            top_k_vals = torch.topk(event_probs, k).values
            top_k_mean = top_k_vals.mean()
            # Binary target: is this event release-positive?
            event_target = torch.tensor(1.0 if has_pos else 0.0, device=logits.device)
            aux_bce = nn.functional.binary_cross_entropy(
                top_k_mean.clamp(1e-7, 1 - 1e-7), event_target)
            aux_losses.append(aux_bce)

            # Duration bucket
            dur = int(em.sum().item())
            for lo, hi in FactorizedLossV2B.DURATION_BUCKETS:
                if lo <= dur < hi:
                    bk = f"{lo}_{hi}"
                    bucket_audit[bk]["event_count"] += 1
                    bucket_audit[bk]["loss_sum"] += float(event_bce.item())
                    break

        # Event-balanced: mean over events (not steps)
        event_mean = torch.stack(event_losses).mean() if event_losses else torch.tensor(0.0, device=logits.device)
        aux_mean = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=logits.device)

        return event_mean, aux_mean, dict(bucket_audit)


__all__ = ["FactorizedLossV2A", "FactorizedLossV2B"]
