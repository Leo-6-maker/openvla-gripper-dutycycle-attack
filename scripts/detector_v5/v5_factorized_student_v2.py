"""V2 Factorized Student models (causal TCN and windowed GRU).

Candidates:
  V2A: Causal TCN encoder + step BCE loss
  V2B: Causal TCN encoder + event-balanced release loss
  V2C: Windowed GRU encoder + event-balanced release loss

All candidates:
  - Use V1 25D features, Teacher targets, known masks
  - Use V1 factorized route heads (single/multi)
  - Abstain on unsupported routes (zero probabilities)
  - No persistent hidden state across full episode
  - ~50K parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

SUPPORTED_ROUTES = {"single_object_pick_place", "multi_object_transfer"}


def _causal_pad_1d(x: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    """Left-only padding for causal 1D convolution. [B, C, T]"""
    pad = (kernel_size - 1) * dilation
    return nn.functional.pad(x, (pad, 0))


class CausalTCNEncoder(nn.Module):
    """Causal temporal convolution encoder with fixed receptive field.

    Uses dilated causal convolutions. Receptive field = sum of (kernel_size-1)*dilation + 1.
    No future information. Output length = input length.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64,
                 receptive_field: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.receptive_field = receptive_field

        # Build enough layers to achieve the target receptive field
        # Each layer: kernel=3, dilation doubles each time: 1,2,4,8,16,...
        layers = []
        current_rf = 1
        dilation = 1
        while current_rf < receptive_field:
            layers.append(nn.Conv1d(hidden_dim if layers else input_dim, hidden_dim,
                                    kernel_size=3, dilation=dilation, padding=0))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_rf += 2 * dilation
            dilation *= 2

        self.conv_stack = nn.Sequential(*layers)
        self.actual_receptive_field = current_rf

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, input_dim] -> [B, T, hidden_dim]"""
        B, T, C = x.shape
        x_t = x.transpose(1, 2)  # [B, C, T]

        # Apply causal convolutions layer by layer with left padding
        out = x_t
        for module in self.conv_stack:
            if isinstance(module, nn.Conv1d):
                pad = (module.kernel_size[0] - 1) * module.dilation[0]
                out = nn.functional.pad(out, (pad, 0))
                out = module(out)
            else:
                out = module(out)

        out = out.transpose(1, 2)  # [B, T, hidden_dim]
        if out.shape[1] > T:
            out = out[:, :T, :]
        elif out.shape[1] < T:
            pad_t = T - out.shape[1]
            out = nn.functional.pad(out, (0, 0, pad_t, 0))

        return out


class WindowedGRUEncoder(nn.Module):
    """GRU encoder with periodic hidden state reset every W steps.

    Hidden state is zeroed at the start of each non-overlapping window of length W.
    No state crosses window boundaries.
    """

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64,
                 window_size: int = 32, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, T, input_dim] -> [B, T, hidden_dim]"""
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        outputs = []

        for t in range(T):
            # Reset hidden state at window boundaries
            if t % self.window_size == 0:
                h = torch.zeros(B, self.hidden_dim, device=device)
            h = self.gru(x[:, t], h)
            h_out = self.dropout(h)
            outputs.append(h_out)

        return torch.stack(outputs, dim=1)


class FactorizedStudentV2(nn.Module):
    """V2 Factorized Student with bounded-context encoder.

    encoder_type: 'tcn' (V2A/V2B) or 'windowed_gru' (V2C)
    """

    def __init__(self, input_dim_25d: int = 25, hidden_dim: int = 64,
                 receptive_field: int = 32, encoder_type: str = 'tcn',
                 dropout: float = 0.0, use_9d: bool = False,
                 input_dim_9d: int = 9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.receptive_field = receptive_field
        self.encoder_type = encoder_type
        self.use_9d = use_9d

        if encoder_type == 'tcn':
            self.encoder_25d = CausalTCNEncoder(
                input_dim_25d, hidden_dim, receptive_field, dropout)
        elif encoder_type == 'windowed_gru':
            self.encoder_25d = WindowedGRUEncoder(
                input_dim_25d, hidden_dim, receptive_field, dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.encoder_9d = None
        self.fusion = None
        if use_9d:
            if encoder_type == 'tcn':
                self.encoder_9d = CausalTCNEncoder(
                    input_dim_9d, hidden_dim, receptive_field, dropout)
            else:
                self.encoder_9d = WindowedGRUEncoder(
                    input_dim_9d, hidden_dim, receptive_field, dropout)
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

    def _route_logits(self, fused: Tensor, route: str) -> dict[str, Tensor]:
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
        logits = self._route_logits(fused, route)
        if route not in SUPPORTED_ROUTES:
            return {"grasp": torch.zeros_like(logits["grasp"]),
                    "manipulation": torch.zeros_like(logits["manipulation"]),
                    "release": torch.zeros_like(logits["release"])}
        return {
            "grasp": torch.sigmoid(logits["grasp"]),
            "manipulation": torch.sigmoid(logits["manipulation"]),
            "release": torch.sigmoid(logits["release"]),
        }

    def forward_logits(self, x_25d: Tensor, x_9d: Tensor | None,
                       mask_25d: Tensor, mask_9d: Tensor | None,
                       route: str) -> dict[str, Tensor]:
        """Training forward: returns logits [B, T]."""
        enc_25d = self.encoder_25d(x_25d)  # [B, T, H]

        if self.use_9d and x_9d is not None and self.encoder_9d is not None:
            enc_9d = self.encoder_9d(x_9d)
            fused = self.fusion(torch.cat([enc_25d, enc_9d], dim=-1))
        else:
            fused = enc_25d

        B, T, H = fused.shape
        fused_flat = fused.reshape(B * T, H)
        logits = self._route_logits(fused_flat, route)

        return {
            "grasp": logits["grasp"].reshape(B, T),
            "manipulation": logits["manipulation"].reshape(B, T),
            "release": logits["release"].reshape(B, T),
        }

    def forward_sequence(self, x_25d: Tensor, x_9d: Tensor | None,
                         mask_25d: Tensor, mask_9d: Tensor | None,
                         route: str) -> dict[str, Tensor]:
        """Inference forward: returns probabilities [B, T]."""
        logits = self.forward_logits(x_25d, x_9d, mask_25d, mask_9d, route)
        if route not in SUPPORTED_ROUTES:
            B, T = x_25d.shape[:2]
            return {
                "grasp": torch.zeros(B, T, device=x_25d.device),
                "manipulation": torch.zeros(B, T, device=x_25d.device),
                "release": torch.zeros(B, T, device=x_25d.device),
            }
        return {
            "grasp": torch.sigmoid(logits["grasp"]),
            "manipulation": torch.sigmoid(logits["manipulation"]),
            "release": torch.sigmoid(logits["release"]),
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["FactorizedStudentV2", "CausalTCNEncoder", "WindowedGRUEncoder",
           "SUPPORTED_ROUTES"]
