"""Official V3 B3 detector model and training contracts.

This module is preparation code.  It contains the stateful GRU, strict masked
loss, normalization, and checkpoint contracts, but it does not authorize a
real run by itself.  A checkpoint is formal only when a sealed training
authorization is supplied by the later V3 evidence gates.
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


FORMAL_CHECKPOINT_SCHEMA = "c2g.b3.official_v3.detector_checkpoint.v1"
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


def json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_tensor(value: Tensor, name: str) -> Tensor:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")
    return value


@dataclass(frozen=True)
class B3Normalization:
    """Normalization computed only from the training fold."""

    mean_25d: tuple[float, ...]
    std_25d: tuple[float, ...]
    mean_9d: tuple[float, ...] = (0.0,) * 9
    std_9d: tuple[float, ...] = (1.0,) * 9

    def __post_init__(self) -> None:
        for name, values, width in (
            ("mean_25d", self.mean_25d, 25),
            ("std_25d", self.std_25d, 25),
            ("mean_9d", self.mean_9d, 9),
            ("std_9d", self.std_9d, 9),
        ):
            if len(values) != width or not all(torch.isfinite(torch.tensor(values, dtype=torch.float64)).tolist()):
                raise ValueError(f"{name} must contain {width} finite values")
        if not all(value > 0.0 for value in self.std_25d + self.std_9d):
            raise ValueError("normalization standard deviations must be positive")

    @classmethod
    def identity(cls) -> "B3Normalization":
        return cls((0.0,) * 25, (1.0,) * 25)

    @property
    def sha256(self) -> str:
        return json_sha(self.to_dict())

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
            tuple(float(v) for v in value["mean_25d"]),
            tuple(float(v) for v in value["std_25d"]),
            tuple(float(v) for v in value.get("mean_9d", (0.0,) * 9)),
            tuple(float(v) for v in value.get("std_9d", (1.0,) * 9)),
        )


@dataclass(frozen=True)
class B3ModelConfig:
    variant: str = "B3_25D"
    hidden_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in B3_VARIANTS:
            raise ValueError(f"unsupported B3 variant: {self.variant}")
        if self.hidden_dim != 128 or self.dropout != 0.0:
            raise ValueError("Official V3 fixes hidden_dim=128 and dropout=0")

    @property
    def feature_order_sha256(self) -> str:
        return json_sha(list(B3_FEATURES_25D))

    @property
    def policy_intent_order_sha256(self) -> str:
        return json_sha(list(B3_POLICY_INTENT_FEATURES_9D))

    @property
    def head_order_sha256(self) -> str:
        return json_sha(list(B3_HEADS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "B3_OFFICIAL_V3_MODEL_CONFIG_V1",
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "feature_names_25d": list(B3_FEATURES_25D),
            "policy_intent_feature_names_9d": list(B3_POLICY_INTENT_FEATURES_9D),
            "head_names": list(B3_HEADS),
            "feature_order_sha256": self.feature_order_sha256,
            "policy_intent_order_sha256": self.policy_intent_order_sha256,
            "head_order_sha256": self.head_order_sha256,
        }

    @property
    def sha256(self) -> str:
        return json_sha(self.to_dict())


HiddenState: TypeAlias = Tensor | tuple[Tensor, Tensor]


class B3OfficialStatefulGRU(nn.Module):
    """Causal stateful GRU; hidden state resets only at episode boundaries."""

    def __init__(self, config: B3ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gru_25d = nn.GRU(25, config.hidden_dim, batch_first=True)
        self.gru_9d: nn.GRU | None = None
        self.fusion: nn.Linear | None = None
        if config.variant == "B3_25D9D":
            self.gru_9d = nn.GRU(9, config.hidden_dim, batch_first=True)
            self.fusion = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.heads = nn.ModuleDict({head: nn.Linear(config.hidden_dim, 1) for head in B3_HEADS})

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
        return h25, torch.zeros_like(h25)

    @staticmethod
    def _step_input(value: Tensor, width: int, name: str) -> Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape [B, {width}], got {tuple(value.shape)}")
        return _finite_tensor(value, name)

    @staticmethod
    def _valid_mask(mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            mask = mask.reshape(1)
        if mask.ndim != 1 or mask.shape[0] != batch_size or mask.dtype != torch.bool:
            raise TypeError(f"valid_mask must be bool with shape [{batch_size}]")
        return mask.to(device=device)

    @staticmethod
    def _keep_valid(old: Tensor, new: Tensor, valid: Tensor) -> Tensor:
        return torch.where(valid.view(1, -1, 1), new, old)

    def step(
        self,
        x25: Tensor,
        x9: Tensor | None = None,
        hidden: HiddenState | None = None,
        valid_mask: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], HiddenState]:
        x25 = self._step_input(x25, 25, "x25")
        batch_size = x25.shape[0]
        valid = self._valid_mask(valid_mask, batch_size, x25.device)
        if hidden is None:
            hidden = self.initial_hidden(batch_size, device=x25.device, dtype=x25.dtype)
        if self.gru_9d is None:
            if isinstance(hidden, tuple) or x9 is not None:
                raise ValueError("B3_25D accepts only a 25D input and one hidden state")
            _, candidate = self.gru_25d(x25.unsqueeze(1), hidden)
            kept = self._keep_valid(hidden, candidate, valid)
            representation = kept[-1]
            next_hidden: HiddenState = kept
        else:
            if x9 is None or not isinstance(hidden, tuple):
                raise ValueError("B3_25D9D requires 9D input and dual hidden state")
            x9 = self._step_input(x9, 9, "x9")
            if x9.shape[0] != batch_size:
                raise ValueError("x25 and x9 batch sizes differ")
            old25, old9 = hidden
            _, candidate25 = self.gru_25d(x25.unsqueeze(1), old25)
            assert self.gru_9d is not None and self.fusion is not None
            _, candidate9 = self.gru_9d(x9.unsqueeze(1), old9)
            kept25 = self._keep_valid(old25, candidate25, valid)
            kept9 = self._keep_valid(old9, candidate9, valid)
            representation = self.fusion(torch.cat((kept25[-1], kept9[-1]), dim=-1))
            next_hidden = kept25, kept9
        return {f"{head}_logit": layer(representation).squeeze(-1) for head, layer in self.heads.items()}, next_hidden

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
        batch, steps, _ = x25.shape
        if self.gru_9d is not None:
            if x9 is None:
                raise ValueError("B3_25D9D requires x9")
            if x9.ndim == 2:
                x9 = x9.unsqueeze(0)
            if x9.ndim != 3 or tuple(x9.shape[:2]) != (batch, steps) or x9.shape[-1] != 9:
                raise ValueError(f"x9 must have shape [{batch}, {steps}, 9]")
        if mask is None:
            mask = torch.ones(batch, steps, dtype=torch.bool, device=x25.device)
        if mask.ndim == 1 and batch == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or tuple(mask.shape) != (batch, steps) or mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool with shape [{batch}, {steps}]")
        outputs: dict[str, list[Tensor]] = {f"{head}_logit": [] for head in B3_HEADS}
        current = hidden
        for index in range(steps):
            step_out, current = self.step(
                x25[:, index], None if x9 is None else x9[:, index], current, mask[:, index]
            )
            for name, value in step_out.items():
                outputs[name].append(value)
        return {name: torch.stack(values, dim=1) for name, values in outputs.items()}, current

    def forward(self, x25: Tensor, x9: Tensor | None = None, hidden: HiddenState | None = None, mask: Tensor | None = None):
        return self.forward_sequence(x25, x9=x9, hidden=hidden, mask=mask)


def build_b3_model(config: B3ModelConfig) -> B3OfficialStatefulGRU:
    return B3OfficialStatefulGRU(config)


def _check_head_maps(logits: Mapping[str, Tensor], targets: Mapping[str, Tensor], masks: Mapping[str, Tensor]) -> None:
    expected_logits = {f"{head}_logit" for head in B3_HEADS}
    if set(logits) != expected_logits or set(targets) != set(B3_HEADS) or set(masks) != set(B3_HEADS):
        raise ValueError("all four B3 heads, targets, and masks are required")


def compute_b3_loss(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
    *,
    episode_valid_mask: Tensor | None = None,
    padding_mask: Tensor | None = None,
) -> Tensor:
    """Strict four-head loss; unknown and padding rows never become negatives."""

    _check_head_maps(logits, targets, masks)
    weights = {"grasp_support": 0.5, "retention_active": 1.0, "retention_continuation_t10": 1.0, "release_imminent": 0.5}
    first = logits["grasp_support_logit"]
    # Keep a zero-valued graph path for every head when the batch is fully
    # unknown.  This makes the no-label case a valid optimizer step instead
    # of silently detaching three heads from the graph.
    total = sum((logits[f"{head}_logit"].sum() * 0.0 for head in B3_HEADS), first.sum() * 0.0)
    weight_sum = 0.0
    combined = None
    for extra in (episode_valid_mask, padding_mask):
        if extra is None:
            continue
        if extra.shape != first.shape or extra.dtype != torch.bool:
            raise TypeError("episode_valid_mask and padding_mask must be bool and match logits")
        combined = extra if combined is None else (combined & extra)
    for head in B3_HEADS:
        logit = logits[f"{head}_logit"]
        target = targets[head]
        mask = masks[head]
        if logit.shape != target.shape or logit.shape != mask.shape or mask.dtype != torch.bool:
            raise TypeError(f"shape/dtype mismatch for {head}")
        effective = mask if combined is None else (mask & combined)
        if bool(effective.any()):
            if not bool(torch.isfinite(target[effective]).all()):
                raise ValueError(f"known target is non-finite for {head}")
            term = F.binary_cross_entropy_with_logits(logit[effective], target[effective].to(logit.dtype))
            total = total + weights[head] * term
            weight_sum += weights[head]
    return total / weight_sum if weight_sum else total


def validate_training_authorization(value: Mapping[str, Any]) -> None:
    required = (
        "schema", "authorization_status", "formal_fit_ready", "s1_materialization_status",
        "teacher_aggregate_status", "formal_training_authorized", "formal_attack_authorized",
        "formal_fit_registry_sha256", "s1_corpus_sha256", "teacher_aggregate_sha256", "runner_head",
    )
    missing = [name for name in required if name not in value]
    if (
        missing
        or value.get("schema") != "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_V1"
        or value.get("authorization_status") != "PASS"
        or value.get("formal_fit_ready") is not True
        or value.get("s1_materialization_status") != "PASS"
        or value.get("teacher_aggregate_status") != "PASS"
        or value.get("formal_training_authorized") is not True
        or value.get("formal_attack_authorized") is not False
    ):
        raise ValueError(f"formal training authorization is missing or unsafe: {missing}")
    for name in ("formal_fit_registry_sha256", "s1_corpus_sha256", "teacher_aggregate_sha256"):
        if not isinstance(value.get(name), str) or len(value[name]) != 64:
            raise ValueError(f"invalid authorization SHA: {name}")
    if not isinstance(value.get("runner_head"), str) or len(value["runner_head"]) != 40:
        raise ValueError("authorization runner_head must be a full Git SHA")


def checkpoint_payload(
    model: B3OfficialStatefulGRU,
    normalization: B3Normalization,
    *,
    authorization: Mapping[str, Any] | None = None,
    training_complete: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    formal = False
    if authorization is not None:
        validate_training_authorization(authorization)
        formal = bool(training_complete)
    payload: dict[str, Any] = {
        "schema": FORMAL_CHECKPOINT_SCHEMA,
        "status": "FORMAL_TRAINED" if formal else "ENGINEERING_SMOKE_ONLY",
        "formal_model": formal,
        "formal_training_ready": formal,
        "formal_attack_ready": False,
        "eligible_for_model_selection": formal,
        "config": model.config.to_dict(),
        "config_sha256": model.config.sha256,
        "normalization": normalization.to_dict(),
        "normalization_sha256": normalization.sha256,
        "authorization": dict(authorization) if authorization is not None else None,
        "model_state": model.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_b3_checkpoint(
    path: str | Path,
    model: B3OfficialStatefulGRU,
    normalization: B3Normalization,
    *,
    authorization: Mapping[str, Any] | None = None,
    training_complete: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, normalization, authorization=authorization, training_complete=training_complete, extra=extra), destination)


def load_b3_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu", require_formal: bool = False):
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema") != FORMAL_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected Official V3 checkpoint schema")
    config = B3ModelConfig(**{name: payload["config"][name] for name in ("variant", "hidden_dim", "dropout")})
    normalization = B3Normalization.from_dict(payload["normalization"])
    if payload.get("config_sha256") != config.sha256 or payload.get("normalization_sha256") != normalization.sha256:
        raise ValueError("checkpoint metadata hash mismatch")
    formal = payload.get("formal_model") is True and payload.get("formal_training_ready") is True
    if require_formal and not formal:
        raise ValueError("checkpoint is not a formally authorized model")
    if payload.get("formal_attack_ready") is not False:
        raise ValueError("checkpoint cannot authorize attack")
    model = build_b3_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, config, normalization, payload


__all__ = [
    "FORMAL_CHECKPOINT_SCHEMA", "B3_FEATURES_25D", "B3_POLICY_INTENT_FEATURES_9D", "B3_HEADS",
    "B3_VARIANTS", "B3Normalization", "B3ModelConfig", "B3OfficialStatefulGRU", "build_b3_model",
    "compute_b3_loss", "checkpoint_payload", "save_b3_checkpoint", "load_b3_checkpoint",
    "validate_training_authorization", "json_sha",
]
