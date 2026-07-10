"""Clean-only C2g gripper-critical window detector and fixed-burst scheduler.

The model predicts when sustained gripper closure is task critical from the clean
causal stream. It does not predict attacked-rollout success and must never consume
post-intervention state or attack outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping

import math
import torch
from torch import Tensor, nn

from .c2g_causal_vulnerability_detector import (
    LanguageQueryPatchPool,
    first_trigger_episode_losses,
    masked_bce,
)


HEAD_NAMES = (
    "critical_window",
    "contact_grasp",
    "close_intent",
    "transport_constraint",
    "release_safe",
    "grounding_confidence",
    "window_start",
    "window_active",
)

FORBIDDEN_TARGET_TOKENS = (
    "vulnerability",
    "cmdopen",
    "attack_outcome",
    "counterfactual",
    "post_intervention",
    "success_flip",
    "object_drop_after",
    "qpos_delta_after",
)


@dataclass(frozen=True)
class C2gDetectorConfig:
    visual_dim: int
    language_dim: int
    policy_intent_dim: int = 9
    hidden: int = 128
    dropout: float = 0.1
    patch_dim: int | None = None
    use_policy_intent: bool = True
    use_visual: bool = True
    use_language_conditioning: bool = True

    def validate(self) -> None:
        for name in ("visual_dim", "language_dim", "policy_intent_dim", "hidden"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.patch_dim is not None and self.patch_dim <= 0:
            raise ValueError("patch_dim must be positive")
        if self.patch_dim is not None and not self.use_visual:
            raise ValueError("patch attention requires use_visual=true")


class C2gGripperCriticalWindowDetector(nn.Module):
    """Causal multi-stream detector with no task-index input."""

    def __init__(self, config: C2gDetectorConfig):
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden
        self.proprio_encoder = nn.GRU(25, hidden, batch_first=True)
        self.policy_encoder = (
            nn.GRU(config.policy_intent_dim, hidden, batch_first=True)
            if config.use_policy_intent
            else None
        )
        self.visual_projection = (
            nn.Sequential(nn.Linear(config.visual_dim, hidden), nn.GELU())
            if config.use_visual
            else None
        )
        self.patch_pool = (
            LanguageQueryPatchPool(config.patch_dim, config.language_dim, hidden)
            if config.patch_dim is not None
            else None
        )
        branch_count = 1 + int(config.use_policy_intent) + int(config.use_visual)
        self.fusion = nn.Sequential(
            nn.Linear(branch_count * hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        if config.use_language_conditioning:
            self.language_film = nn.Linear(config.language_dim, hidden * 2)
            self.language_gate = nn.Linear(config.language_dim, hidden)
        else:
            self.language_film = None
            self.language_gate = None
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in HEAD_NAMES})

    @staticmethod
    def _validate_history(name: str, value: Tensor, batch: int, time: int, dim: int) -> None:
        if value.ndim != 3 or value.shape != (batch, time, dim):
            raise ValueError(f"{name} must have shape [batch,time,{dim}]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

    def _visual_sequence(
        self,
        siglip_visual: Tensor | None,
        language: Tensor,
        *,
        batch: int,
        time: int,
        patch_tokens: Tensor | None,
        patch_token_mask: Tensor | None,
    ) -> Tensor:
        if not self.config.use_visual:
            if siglip_visual is not None or patch_tokens is not None or patch_token_mask is not None:
                raise ValueError("visual inputs supplied while use_visual=false")
            raise RuntimeError("_visual_sequence called with visual branch disabled")
        if patch_tokens is not None:
            if self.patch_pool is None:
                raise ValueError("patch_dim must be configured before patch_tokens can be used")
            return self.patch_pool(
                patch_tokens,
                language,
                time_steps=time,
                patch_token_mask=patch_token_mask,
            )
        if patch_token_mask is not None:
            raise ValueError("patch_token_mask requires patch_tokens")
        if siglip_visual is None:
            raise ValueError("siglip_visual is required when patch_tokens are absent")
        if siglip_visual.ndim == 2 and siglip_visual.shape == (batch, self.config.visual_dim):
            return self.visual_projection(siglip_visual).unsqueeze(1).expand(-1, time, -1)
        if siglip_visual.ndim == 3 and siglip_visual.shape == (batch, time, self.config.visual_dim):
            return self.visual_projection(siglip_visual)
        raise ValueError("siglip_visual must be [batch,visual_dim] or [batch,time,visual_dim]")

    def forward(
        self,
        proprio_25d: Tensor,
        language: Tensor,
        *,
        policy_intent: Tensor | None = None,
        siglip_visual: Tensor | None = None,
        patch_tokens: Tensor | None = None,
        patch_token_mask: Tensor | None = None,
        return_sequence: bool = False,
    ) -> Dict[str, Tensor]:
        if proprio_25d.ndim != 3 or proprio_25d.shape[-1] != 25:
            raise ValueError("proprio_25d must have shape [batch,time,25]")
        if not torch.isfinite(proprio_25d).all():
            raise ValueError("proprio_25d must be finite")
        batch, time, _ = proprio_25d.shape
        if language.ndim != 2 or language.shape != (batch, self.config.language_dim):
            raise ValueError("language must have shape [batch,language_dim]")
        if not torch.isfinite(language).all():
            raise ValueError("language must be finite")

        branches: list[Tensor] = []
        proprio_sequence, _ = self.proprio_encoder(proprio_25d)
        branches.append(proprio_sequence)

        if self.config.use_policy_intent:
            if policy_intent is None:
                raise ValueError("policy_intent is required by this detector configuration")
            self._validate_history(
                "policy_intent",
                policy_intent,
                batch,
                time,
                self.config.policy_intent_dim,
            )
            policy_sequence, _ = self.policy_encoder(policy_intent)
            branches.append(policy_sequence)
        elif policy_intent is not None:
            raise ValueError("policy_intent supplied while use_policy_intent=false")

        if self.config.use_visual:
            branches.append(
                self._visual_sequence(
                    siglip_visual,
                    language,
                    batch=batch,
                    time=time,
                    patch_tokens=patch_tokens,
                    patch_token_mask=patch_token_mask,
                )
            )
        elif siglip_visual is not None or patch_tokens is not None or patch_token_mask is not None:
            raise ValueError("visual inputs supplied while use_visual=false")

        fused = self.fusion(torch.cat(branches, dim=-1))
        if self.config.use_language_conditioning:
            gamma, beta = self.language_film(language).chunk(2, dim=-1)
            gate = torch.sigmoid(self.language_gate(language)).unsqueeze(1)
            conditioned = (1.0 + gamma.unsqueeze(1)) * fused + beta.unsqueeze(1)
            fused = gate * conditioned + (1.0 - gate) * fused
        fused = self.dropout(fused if return_sequence else fused[:, -1])
        return {name: head(fused).squeeze(-1) for name, head in self.heads.items()}


def _reject_outcome_targets(targets: Mapping[str, Tensor]) -> None:
    forbidden = sorted(
        key for key in targets if any(token in key.lower() for token in FORBIDDEN_TARGET_TOKENS)
    )
    if forbidden:
        raise ValueError(
            "post-attack/counterfactual targets are forbidden in clean-window training: "
            + ", ".join(forbidden)
        )


def _temporal_smoothness(probability: Tensor, mask: Tensor) -> Tensor:
    if probability.ndim != 2 or mask.shape != probability.shape:
        raise ValueError("temporal smoothness expects matching [episode,time] tensors")
    if probability.shape[1] < 2:
        return probability.sum() * 0.0
    adjacent = mask[:, 1:].bool() & mask[:, :-1].bool()
    if not adjacent.any():
        return probability.sum() * 0.0
    return (probability[:, 1:] - probability[:, :-1]).abs()[adjacent].mean()


def clean_window_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
    *,
    sample_weight: Tensor | None = None,
    auxiliary_weight: float = 0.2,
    start_weight: float = 0.4,
    active_weight: float = 0.2,
    early_weight: float = 0.25,
    miss_weight: float = 0.5,
    negative_episode_weight: float = 0.5,
    release_safe_episode_weight: float = 0.5,
    smoothness_weight: float = 0.05,
    include_episode_losses: bool | None = None,
) -> Dict[str, Tensor]:
    """Clean-label-only loss aligned to a fixed 2-of-3 online trigger."""

    _reject_outcome_targets(targets)
    required = set(HEAD_NAMES)
    missing_outputs = sorted(required - set(outputs))
    missing_targets = sorted(required - set(targets))
    missing_masks = sorted(required - set(masks))
    if missing_outputs or missing_targets or missing_masks:
        raise ValueError(
            f"missing detector fields outputs={missing_outputs} "
            f"targets={missing_targets} masks={missing_masks}"
        )

    primary = masked_bce(
        outputs["critical_window"],
        targets["critical_window"],
        masks["critical_window"],
        sample_weight,
    )
    auxiliaries = [
        masked_bce(outputs[name], targets[name], masks[name], sample_weight)
        for name in (
            "contact_grasp",
            "close_intent",
            "transport_constraint",
            "release_safe",
            "grounding_confidence",
        )
    ]
    auxiliary = torch.stack(auxiliaries).sum()
    start = masked_bce(
        outputs["window_start"],
        targets["window_start"],
        masks["window_start"],
        sample_weight,
    )
    active = masked_bce(
        outputs["window_active"],
        targets["window_active"],
        masks["window_active"],
        sample_weight,
    )

    critical_logits = outputs["critical_window"]
    if include_episode_losses is None:
        include_episode_losses = critical_logits.ndim == 2
    if include_episode_losses:
        if critical_logits.ndim != 2:
            raise ValueError("episode losses require sequence outputs; call return_sequence=True")
        episode = first_trigger_episode_losses(
            critical_logits,
            targets["critical_window"],
            masks["critical_window"],
            masks.get("episode_fully_known_negative"),
            targets["release_safe"],
            masks["release_safe"],
            return_diagnostics=True,
        )
        smoothness = _temporal_smoothness(
            torch.sigmoid(critical_logits), masks["critical_window"]
        )
    else:
        zero = critical_logits.sum() * 0.0
        episode = {
            "early_emit": zero,
            "episode_miss": zero,
            "negative_episode_any_emit": zero,
            "release_safe_emit": zero,
            "positive_episode_count": zero,
            "triggerable_positive_episode_count": zero,
            "untriggerable_positive_episode_count": zero,
            "persistent_positive_window_count": zero,
        }
        smoothness = zero

    total = (
        primary
        + auxiliary_weight * auxiliary
        + start_weight * start
        + active_weight * active
        + early_weight * episode["early_emit"]
        + miss_weight * episode["episode_miss"]
        + negative_episode_weight * episode["negative_episode_any_emit"]
        + release_safe_episode_weight * episode["release_safe_emit"]
        + smoothness_weight * smoothness
    )
    return {
        "total": total,
        "critical_window": primary,
        "auxiliary": auxiliary,
        "window_start": start,
        "window_active": active,
        "temporal_smoothness": smoothness,
        **episode,
    }


class SchedulerState(str, Enum):
    IDLE = "IDLE"
    BURST = "BURST"
    DONE = "DONE"


@dataclass(frozen=True)
class SchedulerDecision:
    state: SchedulerState
    trigger_started: bool
    attack_active: bool
    attack_index: int | None
    attacked_frames_emitted: int
    gate_now: bool


class FixedBurstTriggerScheduler:
    """Stateful 2-of-3 start gate followed by an immutable fixed-length burst."""

    def __init__(
        self,
        *,
        burst_length: int,
        tau_critical: float,
        tau_release: float,
        tau_ground: float,
        persistence_window: int = 3,
        persistence_required: int = 2,
        one_shot: bool = True,
    ):
        if burst_length <= 0:
            raise ValueError("burst_length must be positive")
        if persistence_required < 1 or persistence_window < persistence_required:
            raise ValueError("persistence requires 1 <= required <= window")
        for name, value in (
            ("tau_critical", tau_critical),
            ("tau_release", tau_release),
            ("tau_ground", tau_ground),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        self.burst_length = int(burst_length)
        self.tau_critical = float(tau_critical)
        self.tau_release = float(tau_release)
        self.tau_ground = float(tau_ground)
        self.persistence_window = int(persistence_window)
        self.persistence_required = int(persistence_required)
        self.one_shot = bool(one_shot)
        self.reset()

    def reset(self) -> None:
        self.state = SchedulerState.IDLE
        self._gate_history: list[bool] = []
        self._remaining = 0
        self._emitted = 0

    @staticmethod
    def _probability(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1]")
        return value

    def update(
        self,
        *,
        critical_probability: float,
        release_safe_probability: float,
        grounding_confidence_probability: float,
        valid: bool = True,
    ) -> SchedulerDecision:
        critical = self._probability("critical_probability", critical_probability)
        release = self._probability("release_safe_probability", release_safe_probability)
        grounding = self._probability(
            "grounding_confidence_probability", grounding_confidence_probability
        )
        gate_now = bool(
            valid
            and critical >= self.tau_critical
            and release < self.tau_release
            and grounding >= self.tau_ground
        )

        trigger_started = False
        attack_active = False
        attack_index: int | None = None

        if self.state == SchedulerState.IDLE:
            self._gate_history.append(gate_now)
            self._gate_history = self._gate_history[-self.persistence_window :]
            persistent = (
                len(self._gate_history) == self.persistence_window
                and sum(self._gate_history) >= self.persistence_required
            )
            if persistent:
                self.state = SchedulerState.BURST
                self._remaining = self.burst_length
                trigger_started = True

        if self.state == SchedulerState.BURST:
            attack_active = True
            attack_index = self._emitted
            self._emitted += 1
            self._remaining -= 1
            if self._remaining == 0:
                self.state = SchedulerState.DONE if self.one_shot else SchedulerState.IDLE
                if not self.one_shot:
                    self._gate_history.clear()

        return SchedulerDecision(
            state=self.state,
            trigger_started=trigger_started,
            attack_active=attack_active,
            attack_index=attack_index,
            attacked_frames_emitted=self._emitted,
            gate_now=gate_now,
        )
