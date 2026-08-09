"""V2 Gripper-Critical Trigger Student.

Architecture (config-driven, not hardcoded):
  Input:
    - 25D proprio/action causal history (required)
    - Policy intent 9d (optional, raw bypass)
    - Gripper token features 9d (optional, raw bypass)
    - Instruction embedding (optional, frozen)

  Encoder:
    - CausalTCN over 25D sequence → temporal hidden [B, T, H]
    - Raw bypass MLP over policy/gripper → bypass_hidden [B, T, H]
    - Concat + fusion MLP → fused [B, T, H]

  Heads (configurable):
    - critical_prob: P(gripper in critical stage)
    - release_safety: P(near intentional release)
    - trigger_score: P(critical) × (1-P(release)) × (optional policy_flip)
    - (auxiliary heads via config)

Design constraint: predicts clean-rollout opportunity timing, NOT attack outcome.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class CausalTCNEncoder(nn.Module):
    """Causal temporal convolution encoder. Same as V1 backbone."""

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64,
                 receptive_field: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        current_dim = input_dim
        # Build dilation stack: 1, 2, 4, 8, 16 → RF ≈ 32
        dilation = 1
        while dilation < receptive_field:
            layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=2,
                                    dilation=dilation, padding=0))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            dilation *= 2
            if dilation * 2 > receptive_field * 2:
                break
        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → [B, T, H]
        x_t = x.transpose(1, 2)  # [B, D, T]
        B, D, T = x_t.shape
        # Left-pad to maintain causality
        total_pad = 0
        for m in self.conv_stack:
            if isinstance(m, nn.Conv1d):
                total_pad += m.dilation[0] * (m.kernel_size[0] - 1)
        if total_pad > 0:
            x_t = F.pad(x_t, (total_pad, 0))
        out = self.conv_stack(x_t)
        # Trim to match input length T
        if out.shape[2] > T:
            out = out[:, :, -T:]
        return out.transpose(1, 2)  # [B, T, H]


class RawBypassEncoder(nn.Module):
    """Small MLP over raw policy/gripper features. Independent of TCN bottleneck."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → [B, T, H]
        return self.net(x)


class FusionMLP(nn.Module):
    """Fuse base encoder + bypass + context into unified hidden."""

    def __init__(self, total_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        x = torch.cat(tensors, dim=-1)
        return self.net(x)


class HeadRegistry(nn.Module):
    """Config-driven head collection. Each head is nn.Linear(H, 1) + optional activation."""

    def __init__(self, hidden_dim: int, head_names: List[str]):
        super().__init__()
        self.head_names = head_names
        for name in head_names:
            setattr(self, name, nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, H] → {name: [B, T, 1]}
        return {name: getattr(self, name)(x) for name in self.head_names}


class CriticalTriggerStudentV2(nn.Module):
    """Gripper-Critical Trigger Student V2.

    Config-driven architecture. Freeze by instantiating with explicit config,
    not by guessing head count.
    """

    def __init__(self,
                 # Core encoder
                 input_dim_25d: int = 25,
                 hidden_dim: int = 64,
                 receptive_field: int = 32,
                 dropout: float = 0.1,
                 # Optional bypass
                 use_policy_bypass: bool = True,
                 policy_bypass_dim: int = 9,
                 use_gripper_bypass: bool = True,
                 gripper_bypass_dim: int = 9,
                 # Optional context
                 use_instruction_context: bool = False,
                 instruction_embed_dim: int = 0,
                 # Heads (config-driven)
                 head_names: Optional[List[str]] = None,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_policy_bypass = use_policy_bypass
        self.use_gripper_bypass = use_gripper_bypass
        self.use_instruction_context = use_instruction_context

        # Base TCN encoder for 25D
        self.encoder_25d = CausalTCNEncoder(
            input_dim_25d, hidden_dim, receptive_field, dropout)

        # Optional raw bypass encoders
        total_fusion_dim = hidden_dim  # from TCN

        if use_policy_bypass:
            self.policy_bypass = RawBypassEncoder(policy_bypass_dim, hidden_dim, dropout)
            total_fusion_dim += hidden_dim
        else:
            self.policy_bypass = None

        if use_gripper_bypass:
            self.gripper_bypass = RawBypassEncoder(gripper_bypass_dim, hidden_dim, dropout)
            total_fusion_dim += hidden_dim
        else:
            self.gripper_bypass = None

        if use_instruction_context:
            assert instruction_embed_dim > 0
            self.instruction_proj = nn.Linear(instruction_embed_dim, hidden_dim)
            total_fusion_dim += hidden_dim
        else:
            self.instruction_proj = None

        # Fusion
        self.fusion = FusionMLP(total_fusion_dim, hidden_dim, dropout)

        # Heads
        if head_names is None:
            head_names = ['critical_prob', 'release_safety']
        self.head_names = head_names
        self.heads = HeadRegistry(hidden_dim, head_names)

    def forward(self,
                x_25d: torch.Tensor,
                x_policy: Optional[torch.Tensor] = None,
                x_gripper: Optional[torch.Tensor] = None,
                instruction_embed: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_25d:      [B, T, 25]  causal proprio/action history
            x_policy:   [B, T, 9]   policy intent features (optional)
            x_gripper:  [B, T, 9]   gripper token features (optional)
            instruction_embed: [B, D_inst] or [B, 1, D_inst] instruction embedding (optional)
            mask:       [B, T]      valid step mask

        Returns:
            Dict of head_name → [B, T, 1] raw logits
        """
        B, T, _ = x_25d.shape

        # Base TCN encoding
        enc_25d = self.encoder_25d(x_25d)  # [B, T, H]

        components = [enc_25d]

        # Policy bypass
        if self.use_policy_bypass and self.policy_bypass is not None:
            if x_policy is None:
                x_policy = torch.zeros(B, T, 9, device=x_25d.device, dtype=x_25d.dtype)
            pol_h = self.policy_bypass(x_policy)  # [B, T, H]
            components.append(pol_h)

        # Gripper bypass
        if self.use_gripper_bypass and self.gripper_bypass is not None:
            if x_gripper is None:
                x_gripper = torch.zeros(B, T, 9, device=x_25d.device, dtype=x_25d.dtype)
            grp_h = self.gripper_bypass(x_gripper)  # [B, T, H]
            components.append(grp_h)

        # Instruction context (tile across time)
        if self.use_instruction_context and self.instruction_proj is not None:
            if instruction_embed is not None:
                if instruction_embed.dim() == 2:
                    instruction_embed = instruction_embed.unsqueeze(1)  # [B, 1, D]
                inst_h = self.instruction_proj(instruction_embed)  # [B, 1, H]
                inst_h = inst_h.expand(-1, T, -1)  # [B, T, H]
                components.append(inst_h)

        # Fusion
        fused = self.fusion(*components)  # [B, T, H]

        # Heads
        logits = self.heads(fused)  # {name: [B, T, 1]}

        return logits

    def compute_trigger_score(self, logits: Dict[str, torch.Tensor],
                               policy_flip_prob: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        """Compute trigger score S_t = P(C_t) × (1-P(R_t)) × P(policy_flip).

        Args:
            logits: head outputs from forward()
            policy_flip_prob: [B, T, 1] optional policy flip probability.
                              If None, uses only critical × (1-release).

        Returns:
            trigger_score: [B, T, 1] in [0, 1]
        """
        crit_prob = torch.sigmoid(logits.get('critical_prob',
                                              torch.zeros_like(list(logits.values())[0])))
        release_prob = torch.sigmoid(logits.get('release_safety',
                                                 torch.zeros_like(crit_prob)))
        trigger = crit_prob * (1.0 - release_prob)

        if policy_flip_prob is not None:
            trigger = trigger * policy_flip_prob

        return trigger

    def get_hidden(self, x_25d: torch.Tensor,
                   x_policy: Optional[torch.Tensor] = None,
                   x_gripper: Optional[torch.Tensor] = None,
                   ) -> torch.Tensor:
        """Extract hidden state before heads (for probing)."""
        B, T, _ = x_25d.shape
        enc_25d = self.encoder_25d(x_25d)
        components = [enc_25d]

        if self.use_policy_bypass and self.policy_bypass is not None:
            if x_policy is None:
                x_policy = torch.zeros(B, T, 9, device=x_25d.device, dtype=x_25d.dtype)
            components.append(self.policy_bypass(x_policy))

        if self.use_gripper_bypass and self.gripper_bypass is not None:
            if x_gripper is None:
                x_gripper = torch.zeros(B, T, 9, device=x_25d.device, dtype=x_25d.dtype)
            components.append(self.gripper_bypass(x_gripper))

        return self.fusion(*components)

    @property
    def config(self) -> dict:
        """Serializable architecture config for SHA binding."""
        return {
            'model_class': 'CriticalTriggerStudentV2',
            'hidden_dim': self.hidden_dim,
            'use_policy_bypass': self.use_policy_bypass,
            'use_gripper_bypass': self.use_gripper_bypass,
            'use_instruction_context': self.use_instruction_context,
            'head_names': self.head_names,
        }


# ── Pre-built recommended configs ──

def build_v2_recommended(head_names: Optional[List[str]] = None,
                          use_instruction: bool = False,
                          instruction_dim: int = 0) -> CriticalTriggerStudentV2:
    """Build recommended V2 config: 25D TCN + policy bypass + gripper bypass."""
    if head_names is None:
        head_names = ['critical_prob', 'release_safety']
    return CriticalTriggerStudentV2(
        input_dim_25d=25,
        hidden_dim=64,
        receptive_field=32,
        dropout=0.1,
        use_policy_bypass=True,
        policy_bypass_dim=9,
        use_gripper_bypass=True,
        gripper_bypass_dim=9,
        use_instruction_context=use_instruction,
        instruction_embed_dim=instruction_dim,
        head_names=head_names,
    )


def build_v2_minimal(head_names: Optional[List[str]] = None) -> CriticalTriggerStudentV2:
    """Minimal V2: 25D TCN only, no bypass. For ablation baseline."""
    if head_names is None:
        head_names = ['critical_prob', 'release_safety']
    return CriticalTriggerStudentV2(
        use_policy_bypass=False,
        use_gripper_bypass=False,
        use_instruction_context=False,
        head_names=head_names,
    )
