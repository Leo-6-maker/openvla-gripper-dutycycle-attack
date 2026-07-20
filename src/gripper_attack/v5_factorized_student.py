"""Factorized Student model (Gate S3).

Narrow adapter on mature B3 causal GRU encoder.
Routes: single_object (3 heads), multi_object (3 heads), unsupported (abstain).

Key invariants:
  - batch forward_sequence == step-by-step streaming
  - hidden_state reset per episode
  - unsupported route → deterministic zero output
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

        # Single-object heads
        self.heads_single = nn.ModuleDict({
            "grasp": nn.Linear(hidden_dim, 1),
            "manipulation": nn.Linear(hidden_dim, 1),
            "release": nn.Linear(hidden_dim, 1),
        })
        # Multi-object heads
        self.heads_multi = nn.ModuleDict({
            "grasp": nn.Linear(hidden_dim, 1),
            "manipulation": nn.Linear(hidden_dim, 1),
            "release": nn.Linear(hidden_dim, 1),
        })

        # Route → heads mapping
        self._route_heads = {
            "single_object_pick_place": self.heads_single,
            "multi_object_transfer": self.heads_multi,
        }

    def _hidden_init(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor | None]:
        h_25d = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_9d = torch.zeros(batch_size, self.hidden_dim, device=device) if self.use_9d else None
        return h_25d, h_9d

    def initial_hidden(self, batch_size: int = 1) -> dict[str, Tensor | None]:
        h_25d, h_9d = self._hidden_init(batch_size, torch.device("cpu"))
        return {"h_25d": h_25d, "h_9d": h_9d}

    def step(
        self,
        x_25d: Tensor,          # [B, 25]
        x_9d: Tensor | None,    # [B, 9] or None
        mask_25d: Tensor,       # [B] bool
        mask_9d: Tensor | None, # [B] bool or None
        hidden: dict[str, Tensor | None],
        route: str,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor | None]]:
        """Single-step streaming inference. Must match forward_sequence output."""
        h_25d = hidden["h_25d"]
        h_9d = hidden.get("h_9d")

        new_h_25d = torch.where(
            mask_25d.unsqueeze(-1),
            self.gru_25d(x_25d, h_25d),
            h_25d,
        )
        new_h_9d = h_9d
        if self.use_9d and h_9d is not None and x_9d is not None and mask_9d is not None:
            new_h_9d = torch.where(
                mask_9d.unsqueeze(-1),
                self.gru_9d(x_9d, h_9d),
                h_9d,
            )

        if self.use_9d and new_h_9d is not None:
            fused = self.fusion(torch.cat([new_h_25d, new_h_9d], dim=-1))
        else:
            fused = new_h_25d

        logits = self._route_logits(fused, route)
        return logits, {"h_25d": new_h_25d, "h_9d": new_h_9d}

    def forward_sequence(
        self,
        x_25d: Tensor,          # [B, T, 25]
        x_9d: Tensor | None,    # [B, T, 9] or None
        mask_25d: Tensor,       # [B, T] bool
        mask_9d: Tensor | None, # [B, T] bool or None
        route: str,
    ) -> dict[str, Tensor]:
        """Batch forward. Output keys: grasp, manipulation, release. Each [B, T]."""
        B, T, _ = x_25d.shape
        device = x_25d.device
        h_25d, h_9d = self._hidden_init(B, device)

        g_logits, m_logits, r_logits = [], [], []
        for t in range(T):
            m25 = mask_25d[:, t]
            m9 = mask_9d[:, t] if mask_9d is not None else None
            x9 = x_9d[:, t] if x_9d is not None else None

            new_h_25d = torch.where(
                m25.unsqueeze(-1), self.gru_25d(x_25d[:, t], h_25d), h_25d
            )
            h_25d = new_h_25d
            if self.use_9d and h_9d is not None and x9 is not None and m9 is not None:
                h_9d = torch.where(m9.unsqueeze(-1), self.gru_9d(x9, h_9d), h_9d)

            if self.use_9d and h_9d is not None:
                fused = self.fusion(torch.cat([h_25d, h_9d], dim=-1))
            else:
                fused = h_25d

            lt = self._route_logits(fused, route)
            g_logits.append(lt["grasp"])
            m_logits.append(lt["manipulation"])
            r_logits.append(lt["release"])

        return {
            "grasp": torch.stack(g_logits, dim=1).squeeze(-1),
            "manipulation": torch.stack(m_logits, dim=1).squeeze(-1),
            "release": torch.stack(r_logits, dim=1).squeeze(-1),
        }

    def _route_logits(self, fused: Tensor, route: str) -> dict[str, Tensor]:
        if route not in SUPPORTED_ROUTES:
            B = fused.shape[0]
            return {"grasp": fused.new_zeros(B), "manipulation": fused.new_zeros(B),
                    "release": fused.new_zeros(B)}
        heads = self._route_heads[route]
        return {
            "grasp": heads["grasp"](fused).squeeze(-1),
            "manipulation": heads["manipulation"](fused).squeeze(-1),
            "release": heads["release"](fused).squeeze(-1),
        }


__all__ = ["FactorizedStudent", "SUPPORTED_ROUTES"]
