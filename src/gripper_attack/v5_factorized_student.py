"""Factorized Student model (Gate S3.1).

Narrow adapter on mature B3 causal GRU encoder.
Routes: single_object (3 heads), multi_object (3 heads), unsupported (abstain).

Key contracts:
  - unsupported route → zero probabilities (explicit mask, not logit trick)
  - route-homogeneous batches (single route per batch)
  - batch forward == step-by-step streaming
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

SUPPORTED_ROUTES = {"single_object_pick_place", "multi_object_transfer"}


class FactorizedStudent(nn.Module):
    def __init__(
        self,
        input_dim_25d: int = 25,
        input_dim_9d: int = 9,
        hidden_dim: int = 128,
        use_9d: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_9d = use_9d

        self.gru_25d = nn.GRUCell(input_dim_25d, hidden_dim)
        if use_9d:
            self.gru_9d = nn.GRUCell(input_dim_9d, hidden_dim)
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.heads_single = nn.ModuleDict({
            "grasp": nn.Linear(hidden_dim, 1),
            "manipulation": nn.Linear(hidden_dim, 1),
            "release": nn.Linear(hidden_dim, 1),
        })
        self.heads_multi = nn.ModuleDict({
            "grasp": nn.Linear(hidden_dim, 1),
            "manipulation": nn.Linear(hidden_dim, 1),
            "release": nn.Linear(hidden_dim, 1),
        })
        self._route_heads = {
            "single_object_pick_place": self.heads_single,
            "multi_object_transfer": self.heads_multi,
        }

    def _hidden_init(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor | None]:
        h_25d = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        h_9d = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype) if self.use_9d else None
        return h_25d, h_9d

    def initial_hidden(self, batch_size: int = 1, device: torch.device | str = "cpu") -> dict[str, Tensor | None]:
        if isinstance(device, str):
            device = torch.device(device)
        h_25d, h_9d = self._hidden_init(batch_size, device)
        return {"h_25d": h_25d, "h_9d": h_9d}

    def step(
        self,
        x_25d: Tensor,
        x_9d: Tensor | None,
        mask_25d: Tensor,
        mask_9d: Tensor | None,
        hidden: dict[str, Tensor | None],
        route: str,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | None]]:
        """Single-step streaming. Returns zero PROBABILITIES for unsupported routes."""
        h_25d = hidden["h_25d"]
        h_9d = hidden.get("h_9d")

        new_h_25d = torch.where(
            mask_25d.unsqueeze(-1), self.gru_25d(x_25d, h_25d), h_25d,
        )
        new_h_9d = h_9d
        if self.use_9d and h_9d is not None and x_9d is not None and mask_9d is not None:
            new_h_9d = torch.where(
                mask_9d.unsqueeze(-1), self.gru_9d(x_9d, h_9d), h_9d,
            )

        if self.use_9d and new_h_9d is not None:
            fused = self.fusion(torch.cat([new_h_25d, new_h_9d], dim=-1))
        else:
            fused = new_h_25d

        return self._route_probs(fused, route), {"h_25d": new_h_25d, "h_9d": new_h_9d}

    def forward_sequence(
        self,
        x_25d: Tensor,
        x_9d: Tensor | None,
        mask_25d: Tensor,
        mask_9d: Tensor | None,
        route: str,
    ) -> dict[str, Tensor]:
        """Batch forward [B,T,*]. Route-homogeneous batch."""
        B, T, _ = x_25d.shape
        device = x_25d.device
        h_25d, h_9d = self._hidden_init(B, device, x_25d.dtype)

        g_probs, m_probs, r_probs = [], [], []
        for t in range(T):
            m25 = mask_25d[:, t]
            m9 = mask_9d[:, t] if mask_9d is not None else None
            x9 = x_9d[:, t] if x_9d is not None else None

            h_25d = torch.where(m25.unsqueeze(-1), self.gru_25d(x_25d[:, t], h_25d), h_25d)
            if self.use_9d and h_9d is not None and x9 is not None and m9 is not None:
                h_9d = torch.where(m9.unsqueeze(-1), self.gru_9d(x9, h_9d), h_9d)

            if self.use_9d and h_9d is not None:
                fused = self.fusion(torch.cat([h_25d, h_9d], dim=-1))
            else:
                fused = h_25d

            pt = self._route_probs(fused, route)
            g_probs.append(pt["grasp"]); m_probs.append(pt["manipulation"]); r_probs.append(pt["release"])

        return {
            "grasp": torch.stack(g_probs, dim=1),
            "manipulation": torch.stack(m_probs, dim=1),
            "release": torch.stack(r_probs, dim=1),
        }

    def _route_logits(self, fused: Tensor, route: str) -> dict[str, Tensor]:
        """Raw logits for training (BCEWithLogitsLoss)."""
        B = fused.shape[0]
        if route not in SUPPORTED_ROUTES:
            z = fused.new_full((B,), -1e4)
            return {"grasp": z, "manipulation": z, "release": z}
        heads = self._route_heads[route]
        return {
            "grasp": heads["grasp"](fused).squeeze(-1),
            "manipulation": heads["manipulation"](fused).squeeze(-1),
            "release": heads["release"](fused).squeeze(-1),
        }

    def _route_probs(self, fused: Tensor, route: str) -> dict[str, Tensor]:
        """Sigmoid probabilities — zero for unsupported routes."""
        logits = self._route_logits(fused, route)
        if route not in SUPPORTED_ROUTES:
            return logits  # already near-zero probs via large negative logits
        return {
            "grasp": torch.sigmoid(logits["grasp"]),
            "manipulation": torch.sigmoid(logits["manipulation"]),
            "release": torch.sigmoid(logits["release"]),
        }

    def forward_logits(
        self, x_25d: Tensor, x_9d: Tensor | None,
        mask_25d: Tensor, mask_9d: Tensor | None, route: str,
    ) -> dict[str, Tensor]:
        """Training forward: returns logits [B,T] for BCEWithLogitsLoss."""
        B, T, _ = x_25d.shape
        device = x_25d.device
        h_25d, h_9d = self._hidden_init(B, device, x_25d.dtype)
        g_logits, m_logits, r_logits = [], [], []
        for t in range(T):
            m25 = mask_25d[:, t]; m9 = mask_9d[:, t] if mask_9d is not None else None
            x9 = x_9d[:, t] if x_9d is not None else None
            h_25d = torch.where(m25.unsqueeze(-1), self.gru_25d(x_25d[:, t], h_25d), h_25d)
            if self.use_9d and h_9d is not None and x9 is not None and m9 is not None:
                h_9d = torch.where(m9.unsqueeze(-1), self.gru_9d(x9, h_9d), h_9d)
            fused = self.fusion(torch.cat([h_25d, h_9d], dim=-1)) if (self.use_9d and h_9d is not None) else h_25d
            lt = self._route_logits(fused, route)
            g_logits.append(lt["grasp"]); m_logits.append(lt["manipulation"]); r_logits.append(lt["release"])
        return {
            "grasp": torch.stack(g_logits, dim=1),
            "manipulation": torch.stack(m_logits, dim=1),
            "release": torch.stack(r_logits, dim=1),
        }


__all__ = ["FactorizedStudent", "SUPPORTED_ROUTES"]
