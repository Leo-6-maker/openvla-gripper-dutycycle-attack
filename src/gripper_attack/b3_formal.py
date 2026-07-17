"""Official V3 B3 detector model and training contracts.

This module is preparation code.  It contains the stateful GRU, strict masked
loss, normalization, and checkpoint contracts, but it does not authorize a
real run by itself.  A checkpoint is formal only when a sealed training
authorization is supplied by the later V3 evidence gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
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
AUTHORIZATION_INPUT_NAMES = (
    "formal_fit_registry_sha256",
    "formal_registry_summary_sha256",
    "formal_registry_root_sha256",
    "s1_corpus_sha256",
    "s1_root_audit_sha256",
    "teacher_aggregate_sha256",
    "training_protocol_sha256",
    "source_contract_sha256",
    "protocol_sha256",
    "feature_rebuilder_sha256",
    "normalization_bundle_sha256",
    "normalization_sha256",
    "fold_manifest_sha256",
)
FORMAL_CHECKPOINT_STATUSES = (
    "ENGINEERING_SMOKE_ONLY",
    "FIT_FOLD_TRAINED_CANDIDATE",
    "FIT_VIABILITY_PASS",
    "FULL_FIT_REFIT_CANDIDATE",
    "FIT_DEV_SELECTED",
    "CALIBRATED",
    "CHECK_PASS",
    "ATTACK_CANARY_PASS",
)
MODEL_SELECTION_STATUSES = frozenset({"FIT_DEV_SELECTED", "CALIBRATED", "CHECK_PASS", "ATTACK_CANARY_PASS"})


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
        *AUTHORIZATION_INPUT_NAMES, "runner_head", "runner_binding", "authorization_generation",
        "authorization_payload_sha256", "input_snapshots", "verification", "variant", "fit_scope", "fold_id", "seed",
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
    for name in (*AUTHORIZATION_INPUT_NAMES, "authorization_payload_sha256"):
        if not isinstance(value.get(name), str) or len(value[name]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value[name]):
            raise ValueError(f"invalid authorization SHA: {name}")
    if not isinstance(value.get("runner_head"), str) or len(value["runner_head"]) != 40:
        raise ValueError("authorization runner_head must be a full Git SHA")
    if value.get("variant") not in B3_VARIANTS or value.get("seed") not in (20260717, 20260718, 20260719):
        raise ValueError("authorization matrix coordinates are invalid")
    if value.get("fit_scope") == "FIT_FOLD":
        if value.get("fold_id") not in range(4):
            raise ValueError("fold authorization must name one of four folds")
    elif value.get("fit_scope") == "FULL_FIT":
        if value.get("fold_id") != "FULL_FIT":
            raise ValueError("full-FIT authorization must use fold_id=FULL_FIT")
    else:
        raise ValueError("authorization fit_scope is invalid")
    generation = value.get("authorization_generation")
    if not isinstance(generation, Mapping) or generation.get("schema") != "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_GENERATOR_V1":
        raise ValueError("authorization is not machine-generated by the V3 builder")
    if (
        generation.get("generator_worktree_clean") is not True
        or generation.get("generator_script_tracked") is not True
        or generation.get("semantic_inputs_verified") is not True
        or generation.get("generator_entrypoint") != "build_b3_v3_training_authorization.py"
    ):
        raise ValueError("authorization generator provenance is incomplete")
    if not isinstance(generation.get("generator_script_sha256"), str) or len(generation["generator_script_sha256"]) != 64:
        raise ValueError("authorization generator SHA is missing")
    if not isinstance(generation.get("generator_script_git_blob_sha1"), str) or len(generation["generator_script_git_blob_sha1"]) != 40:
        raise ValueError("authorization generator Git blob is missing")
    if not isinstance(generation.get("generator_head"), str) or len(generation["generator_head"]) != 40:
        raise ValueError("authorization generator HEAD is missing")
    binding = value.get("runner_binding")
    if not isinstance(binding, Mapping) or binding.get("status") != "PASS" or binding.get("runner_worktree_clean") is not True:
        raise ValueError("authorization runner binding is not PASS")
    binding_sha = binding.get("runner_binding_sha256")
    binding_body = {key: item for key, item in binding.items() if key != "runner_binding_sha256"}
    if binding_sha != json_sha(binding_body):
        raise ValueError("authorization runner binding SHA is invalid")
    snapshots = value.get("input_snapshots")
    if not isinstance(snapshots, Mapping):
        raise ValueError("authorization input snapshots are missing")
    snapshot_names = set(AUTHORIZATION_INPUT_NAMES)
    if set(snapshots) != snapshot_names:
        raise ValueError("authorization input snapshot set is not exactly frozen")
    for name in snapshot_names:
        if snapshots.get(name) != value.get(name):
            raise ValueError(f"authorization input snapshot mismatch: {name}")
    verification = value.get("verification")
    if not isinstance(verification, Mapping) or verification.get("status") != "PASS" or verification.get("semantic_inputs_verified") is not True:
        raise ValueError("authorization semantic verification is missing or unsafe")
    if verification.get("normalization_recomputed") is not True or verification.get("runner_binding_measured") is not True:
        raise ValueError("authorization normalization/runner verification is incomplete")
    body = dict(value)
    body.pop("authorization_payload_sha256", None)
    if value.get("authorization_payload_sha256") != json_sha(body):
        raise ValueError("authorization payload SHA is invalid")


def checkpoint_payload(
    model: B3OfficialStatefulGRU,
    normalization: B3Normalization,
    *,
    authorization: Mapping[str, Any] | None = None,
    training_complete: bool = False,
    checkpoint_status: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status = "ENGINEERING_SMOKE_ONLY"
    formal = False
    if authorization is not None:
        validate_training_authorization(authorization)
        if training_complete:
            status = checkpoint_status or "FIT_FOLD_TRAINED_CANDIDATE"
            if status not in FORMAL_CHECKPOINT_STATUSES or status == "ENGINEERING_SMOKE_ONLY":
                raise ValueError(f"invalid formal checkpoint status: {status}")
            formal = True
        elif checkpoint_status not in (None, "ENGINEERING_SMOKE_ONLY"):
            raise ValueError("incomplete training cannot claim a formal checkpoint status")
    payload: dict[str, Any] = {
        "schema": FORMAL_CHECKPOINT_SCHEMA,
        "status": status,
        "checkpoint_status": status,
        "formal_model": formal,
        "formal_training_ready": formal,
        "formal_attack_ready": False,
        "eligible_for_model_selection": status in MODEL_SELECTION_STATUSES,
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
    checkpoint_status: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model, normalization, authorization=authorization, training_complete=training_complete,
            checkpoint_status=checkpoint_status, extra=extra,
        ),
        destination,
    )


def save_b3_checkpoint_bundle(
    output_root: str | Path,
    model: B3OfficialStatefulGRU,
    normalization: B3Normalization,
    *,
    authorization: Mapping[str, Any],
    checkpoint_status: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a non-overwrite formal checkpoint bundle with a full file seal."""

    from .b3_training_protocol import seal_directory, sha256_file

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint bundle: {destination}")
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        model_path = staging / "checkpoint.pt"
        save_b3_checkpoint(
            model_path, model, normalization, authorization=authorization, training_complete=True,
            checkpoint_status=checkpoint_status, extra=extra,
        )
        model_sha = sha256_file(model_path)
        manifest = {
            "schema": "B3_OFFICIAL_V3_CHECKPOINT_BUNDLE_V1",
            "checkpoint_file": model_path.name,
            "checkpoint_sha256": model_sha,
            "checkpoint_status": checkpoint_status,
            "authorization_payload_sha256": authorization["authorization_payload_sha256"],
            "variant": model.config.variant,
            "normalization_sha256": normalization.sha256,
            "formal_training_authorized": True,
            "formal_attack_authorized": False,
        }
        manifest_path = staging / "checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.with_name(manifest_path.name + ".sha256").write_text(
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8",
        )
        seal_directory(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_b3_checkpoint_bundle(
    root: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    require_formal: bool = False,
):
    from .b3_training_protocol import verify_sealed_directory, sha256_file

    bundle = Path(root)
    verify_sealed_directory(bundle)
    manifest = json.loads((bundle / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "B3_OFFICIAL_V3_CHECKPOINT_BUNDLE_V1":
        raise ValueError("unexpected checkpoint bundle schema")
    model_path = bundle / str(manifest.get("checkpoint_file", ""))
    if not model_path.is_file() or sha256_file(model_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint bundle model checksum mismatch")
    model, config, normalization, payload = load_b3_checkpoint(model_path, map_location=map_location, require_formal=require_formal)
    if payload.get("checkpoint_status") != manifest.get("checkpoint_status") or payload.get("authorization", {}).get("authorization_payload_sha256") != manifest.get("authorization_payload_sha256"):
        raise ValueError("checkpoint bundle manifest binding mismatch")
    return model, config, normalization, payload, manifest


def load_b3_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu", require_formal: bool = False):
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema") != FORMAL_CHECKPOINT_SCHEMA:
        raise ValueError("unexpected Official V3 checkpoint schema")
    config = B3ModelConfig(**{name: payload["config"][name] for name in ("variant", "hidden_dim", "dropout")})
    normalization = B3Normalization.from_dict(payload["normalization"])
    if payload.get("config_sha256") != config.sha256 or payload.get("normalization_sha256") != normalization.sha256:
        raise ValueError("checkpoint metadata hash mismatch")
    formal = payload.get("formal_model") is True and payload.get("formal_training_ready") is True and payload.get("checkpoint_status") in FORMAL_CHECKPOINT_STATUSES[1:]
    if require_formal and not formal:
        raise ValueError("checkpoint is not a formally authorized model")
    if formal:
        validate_training_authorization(payload.get("authorization", {}))
    if payload.get("formal_attack_ready") is not False:
        raise ValueError("checkpoint cannot authorize attack")
    model = build_b3_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, config, normalization, payload


__all__ = [
    "FORMAL_CHECKPOINT_SCHEMA", "FORMAL_CHECKPOINT_STATUSES", "B3_FEATURES_25D", "B3_POLICY_INTENT_FEATURES_9D", "B3_HEADS",
    "B3_VARIANTS", "AUTHORIZATION_INPUT_NAMES", "B3Normalization", "B3ModelConfig", "B3OfficialStatefulGRU", "build_b3_model",
    "compute_b3_loss", "checkpoint_payload", "save_b3_checkpoint", "load_b3_checkpoint",
    "save_b3_checkpoint_bundle", "load_b3_checkpoint_bundle",
    "validate_training_authorization", "json_sha",
]
