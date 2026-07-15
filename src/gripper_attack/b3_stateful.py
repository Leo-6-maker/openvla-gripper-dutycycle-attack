"""Engineering-only stateful B3 retention detector.

This module implements the fixed model/runtime contract described by the B3
retention protocol.  It deliberately contains no Official Teacher loading,
threshold selection, rollout code, or attack logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .sc5_detector_runtime import SC5_FEATURES


B3_FEATURES_25D = tuple(SC5_FEATURES)
B3_POLICY_INTENT_FEATURES_9D = (
    "clean_open_probability_mass",
    "clean_close_probability_mass",
    "clean_open_minus_close_log_mass",
    "clean_action_token_entropy_normalized",
    "clean_top1_probability",
    "clean_top1_is_open",
    "clean_top1_is_close",
    "clean_best_open_rank_normalized",
    "clean_best_close_rank_normalized",
)
B3_HEADS = (
    "grasp_support",
    "retention_active",
    "retention_continuation_t10",
    "release_imminent",
)
B3_VARIANTS = ("B3_25D", "B3_25D9D")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class B3Normalization:
    """Frozen input normalization metadata for a checkpoint/runtime."""

    mean_25d: tuple[float, ...] = (0.0,) * 25
    std_25d: tuple[float, ...] = (1.0,) * 25
    mean_9d: tuple[float, ...] = (0.0,) * 9
    std_9d: tuple[float, ...] = (1.0,) * 9

    def __post_init__(self) -> None:
        for name, values, size in (
            ("mean_25d", self.mean_25d, 25),
            ("std_25d", self.std_25d, 25),
            ("mean_9d", self.mean_9d, 9),
            ("std_9d", self.std_9d, 9),
        ):
            if len(values) != size:
                raise ValueError(f"{name} must have length {size}")
            if not all(torch.isfinite(torch.tensor(values, dtype=torch.float64)).tolist()):
                raise ValueError(f"{name} contains non-finite values")
        if not all(value > 0.0 for value in self.std_25d + self.std_9d):
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean_25d": list(self.mean_25d),
            "std_25d": list(self.std_25d),
            "mean_9d": list(self.mean_9d),
            "std_9d": list(self.std_9d),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "B3Normalization":
        return cls(
            mean_25d=tuple(float(v) for v in value["mean_25d"]),
            std_25d=tuple(float(v) for v in value["std_25d"]),
            mean_9d=tuple(float(v) for v in value["mean_9d"]),
            std_9d=tuple(float(v) for v in value["std_9d"]),
        )

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class B3ModelConfig:
    variant: str = "B3_25D"
    hidden_dim: int = 128
    feature_names_25d: tuple[str, ...] = B3_FEATURES_25D
    policy_intent_feature_names_9d: tuple[str, ...] = B3_POLICY_INTENT_FEATURES_9D
    head_names: tuple[str, ...] = B3_HEADS
    status: str = "ENGINEERING_SMOKE_ONLY"
    formal_model: bool = False

    def __post_init__(self) -> None:
        if self.variant not in B3_VARIANTS:
            raise ValueError(f"unsupported B3 variant: {self.variant}")
        if self.hidden_dim != 128:
            raise ValueError("B3 engineering contract fixes hidden_dim=128")
        if len(self.feature_names_25d) != 25 or tuple(self.feature_names_25d) != B3_FEATURES_25D:
            raise ValueError("25D feature order does not match OFFICIAL_25D_V1")
        if tuple(self.policy_intent_feature_names_9d) != B3_POLICY_INTENT_FEATURES_9D:
            raise ValueError("9D policy-intent feature order is not frozen")
        if tuple(self.head_names) != B3_HEADS:
            raise ValueError("B3 head order is not frozen")
        if self.status != "ENGINEERING_SMOKE_ONLY" or self.formal_model:
            raise ValueError("B3 smoke checkpoints must not be marked formal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "feature_names_25d": list(self.feature_names_25d),
            "policy_intent_feature_names_9d": list(self.policy_intent_feature_names_9d),
            "head_names": list(self.head_names),
            "status": self.status,
            "formal_model": self.formal_model,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "B3ModelConfig":
        return cls(
            variant=str(value["variant"]),
            hidden_dim=int(value["hidden_dim"]),
            feature_names_25d=tuple(value["feature_names_25d"]),
            policy_intent_feature_names_9d=tuple(value["policy_intent_feature_names_9d"]),
            head_names=tuple(value["head_names"]),
            status=str(value["status"]),
            formal_model=bool(value["formal_model"]),
        )

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @property
    def feature_order_sha256(self) -> str:
        return _sha256_json({"features_25d": list(self.feature_names_25d), "features_9d": list(self.policy_intent_feature_names_9d)})

    @property
    def head_order_sha256(self) -> str:
        return _sha256_json(list(self.head_names))


HiddenState: TypeAlias = Tensor | tuple[Tensor, Tensor]


class B3StatefulGRU(nn.Module):
    """Fixed stateful GRU with explicit per-step and sequence APIs."""

    def __init__(self, config: B3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gru_25d = nn.GRU(25, config.hidden_dim, batch_first=True)
        if config.variant == "B3_25D9D":
            self.gru_9d: nn.GRU | None = nn.GRU(9, config.hidden_dim, batch_first=True)
            self.fusion: nn.Linear | None = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        else:
            self.gru_9d = None
            self.fusion = None
        self.heads = nn.ModuleDict({name: nn.Linear(config.hidden_dim, 1) for name in B3_HEADS})

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> HiddenState:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        h25 = torch.zeros(1, batch_size, self.config.hidden_dim, device=device, dtype=dtype)
        if self.gru_9d is None:
            return h25
        h9 = torch.zeros_like(h25)
        return h25, h9

    @staticmethod
    def _batch_features(value: Tensor, width: int, name: str) -> Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape [B, {width}], got {tuple(value.shape)}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
        return value

    @staticmethod
    def _valid_mask(value: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if value.ndim == 0:
            value = value.reshape(1)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"valid_mask must have shape [{batch_size}], got {tuple(value.shape)}")
        return value.to(device=device, dtype=torch.bool)

    @staticmethod
    def _select_hidden(old: Tensor, new: Tensor, valid_mask: Tensor) -> Tensor:
        selector = valid_mask.view(1, -1, 1)
        return torch.where(selector, new, old)

    def _representation(
        self,
        x25: Tensor,
        x9: Tensor | None,
        hidden: HiddenState | None,
        valid_mask: Tensor,
    ) -> tuple[Tensor, HiddenState]:
        batch_size = x25.shape[0]
        if hidden is None:
            hidden = self.initial_hidden(batch_size, device=x25.device, dtype=x25.dtype)
        if self.gru_9d is None:
            if isinstance(hidden, tuple):
                raise ValueError("B3_25D cannot receive dual hidden state")
            _, new25 = self.gru_25d(x25.unsqueeze(1), hidden)
            selected25 = self._select_hidden(hidden, new25, valid_mask)
            return selected25[-1], selected25

        if x9 is None:
            raise ValueError("B3_25D9D requires 9D policy-intent input")
        if not isinstance(hidden, tuple):
            raise ValueError("B3_25D9D requires dual hidden state")
        old25, old9 = hidden
        _, new25 = self.gru_25d(x25.unsqueeze(1), old25)
        _, new9 = self.gru_9d(x9.unsqueeze(1), old9)
        selected25 = self._select_hidden(old25, new25, valid_mask)
        selected9 = self._select_hidden(old9, new9, valid_mask)
        assert self.fusion is not None
        representation = self.fusion(torch.cat((selected25[-1], selected9[-1]), dim=-1))
        return representation, (selected25, selected9)

    def step(
        self,
        x25: Tensor,
        x9: Tensor | None = None,
        hidden: HiddenState | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], HiddenState]:
        x25 = self._batch_features(x25, 25, "x25")
        if self.gru_9d is not None:
            x9 = self._batch_features(x9 if x9 is not None else torch.empty(0), 9, "x9")
            if x9.shape[0] != x25.shape[0]:
                raise ValueError("x25 and x9 batch sizes differ")
        valid = self._valid_mask(valid_mask, x25.shape[0], x25.device)
        representation, next_hidden = self._representation(x25, x9, hidden, valid)
        return {f"{name}_logit": head(representation).squeeze(-1) for name, head in self.heads.items()}, next_hidden

    def forward_sequence(
        self,
        x25: Tensor,
        x9: Tensor | None = None,
        hidden: HiddenState | None = None,
        mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], HiddenState]:
        if x25.ndim == 2:
            x25 = x25.unsqueeze(0)
        if x25.ndim != 3 or x25.shape[-1] != 25:
            raise ValueError(f"x25 must have shape [B, T, 25], got {tuple(x25.shape)}")
        batch_size, steps, _ = x25.shape
        if self.gru_9d is not None:
            if x9 is None:
                raise ValueError("B3_25D9D requires 9D policy-intent input")
            if x9.ndim == 2:
                x9 = x9.unsqueeze(0)
            if x9.ndim != 3 or tuple(x9.shape[:2]) != (batch_size, steps) or x9.shape[-1] != 9:
                raise ValueError(f"x9 must have shape [{batch_size}, {steps}, 9], got {tuple(x9.shape)}")
        if mask is None:
            mask = torch.ones(batch_size, steps, dtype=torch.bool, device=x25.device)
        elif mask.ndim == 1 and batch_size == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or tuple(mask.shape) != (batch_size, steps):
            raise ValueError(f"mask must have shape [{batch_size}, {steps}], got {tuple(mask.shape)}")

        rows: dict[str, list[Tensor]] = {f"{name}_logit": [] for name in B3_HEADS}
        current = hidden
        for step_index in range(steps):
            step_logits, current = self.step(
                x25[:, step_index],
                None if x9 is None else x9[:, step_index],
                current,
                mask[ :, step_index],
            )
            for name, value in step_logits.items():
                rows[name].append(value)
        return {name: torch.stack(values, dim=1) for name, values in rows.items()}, current

    def forward(
        self,
        x25: Tensor,
        x9: Tensor | None = None,
        hidden: HiddenState | None = None,
        mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], HiddenState]:
        return self.forward_sequence(x25, x9=x9, hidden=hidden, mask=mask)


class B3_25D(B3StatefulGRU):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__(B3ModelConfig(variant="B3_25D", hidden_dim=hidden_dim))


class B3_25D9D(B3StatefulGRU):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__(B3ModelConfig(variant="B3_25D9D", hidden_dim=hidden_dim))


def build_b3_model(config: B3ModelConfig) -> B3StatefulGRU:
    return B3_25D(config.hidden_dim) if config.variant == "B3_25D" else B3_25D9D(config.hidden_dim)


def masked_bce_loss(logit: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """BCE that excludes unknown labels and remains finite when all are masked."""

    target = target.to(device=logit.device, dtype=logit.dtype)
    mask = mask.to(device=logit.device, dtype=torch.bool) & torch.isfinite(target)
    if not bool(mask.any()):
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[mask], target[mask])


def compute_b3_loss(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
) -> Tensor:
    """Compute the four-head masked smoke loss; unknown labels never become negatives."""

    weights = {
        "grasp_support": 0.5,
        "retention_active": 1.0,
        "retention_continuation_t10": 1.0,
        "release_imminent": 0.5,
    }
    first = next(iter(logits.values()))
    total = first.sum() * 0.0
    weight_sum = 0.0
    for head in B3_HEADS:
        target = targets.get(head)
        mask = masks.get(head)
        if target is None or mask is None:
            continue
        term = masked_bce_loss(logits[f"{head}_logit"], target, mask)
        total = total + weights[head] * term
        if bool((mask.to(dtype=torch.bool) & torch.isfinite(target)).any()):
            weight_sum += weights[head]
    return total / weight_sum if weight_sum else total


def _checkpoint_payload(
    model: B3StatefulGRU,
    normalization: B3Normalization,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = model.config
    payload: dict[str, Any] = {
        "schema": "c2g.b3_stateful_checkpoint.v1",
        "status": "ENGINEERING_SMOKE_ONLY",
        "formal_model": False,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "eligible_for_model_selection": False,
        "config": config.to_dict(),
        "config_hash": config.sha256,
        "feature_order_hash": config.feature_order_sha256,
        "head_order_hash": config.head_order_sha256,
        "normalization": normalization.to_dict(),
        "normalization_hash": normalization.sha256,
        "model_state": model.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_b3_checkpoint(
    path: str | Path,
    model: B3StatefulGRU,
    normalization: B3Normalization | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    normalization = normalization or B3Normalization()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, normalization, extra), destination)


def load_b3_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[B3StatefulGRU, B3ModelConfig, B3Normalization, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema") != "c2g.b3_stateful_checkpoint.v1":
        raise ValueError("unexpected B3 checkpoint schema")
    if payload.get("status") != "ENGINEERING_SMOKE_ONLY" or payload.get("formal_model") is not False:
        raise ValueError("formal B3 checkpoint cannot be loaded by the smoke runtime")
    if any(payload.get(name) is not False for name in (
        "formal_training_ready",
        "formal_attack_ready",
        "eligible_for_model_selection",
    )):
        raise ValueError("B3 smoke checkpoint has an unsafe eligibility flag")
    config = B3ModelConfig.from_dict(payload["config"])
    normalization = B3Normalization.from_dict(payload["normalization"])
    if payload.get("config_hash") != config.sha256:
        raise ValueError("B3 config hash mismatch")
    if payload.get("feature_order_hash") != config.feature_order_sha256:
        raise ValueError("B3 feature-order hash mismatch")
    if payload.get("head_order_hash") != config.head_order_sha256:
        raise ValueError("B3 head-order hash mismatch")
    if payload.get("normalization_hash") != normalization.sha256:
        raise ValueError("B3 normalization hash mismatch")
    model = build_b3_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, config, normalization, payload


class B3StatefulRuntime:
    """Small online wrapper whose only reset operation is episode reset."""

    def __init__(self, model: B3StatefulGRU, normalization: B3Normalization | None = None) -> None:
        self.model = model.eval()
        self.normalization = normalization or B3Normalization()
        self.hidden: HiddenState | None = None

    def reset_episode(self) -> None:
        self.hidden = None

    def _normalize(self, value: Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
        mean_tensor = torch.tensor(mean, device=value.device, dtype=value.dtype)
        std_tensor = torch.tensor(std, device=value.device, dtype=value.dtype)
        return (value - mean_tensor) / std_tensor

    @torch.no_grad()
    def step(
        self,
        x25: Tensor,
        x9: Tensor | None = None,
        *,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        x25 = self._normalize(x25, self.normalization.mean_25d, self.normalization.std_25d)
        if x9 is not None:
            x9 = self._normalize(x9, self.normalization.mean_9d, self.normalization.std_9d)
        logits, self.hidden = self.model.step(x25, x9, self.hidden, valid_mask)
        return logits


__all__ = [
    "B3_FEATURES_25D",
    "B3_POLICY_INTENT_FEATURES_9D",
    "B3_HEADS",
    "B3ModelConfig",
    "B3Normalization",
    "B3StatefulGRU",
    "B3StatefulRuntime",
    "B3_25D",
    "B3_25D9D",
    "build_b3_model",
    "compute_b3_loss",
    "load_b3_checkpoint",
    "masked_bce_loss",
    "save_b3_checkpoint",
]
