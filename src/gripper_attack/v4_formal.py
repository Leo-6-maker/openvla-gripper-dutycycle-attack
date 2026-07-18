"""Formal, FIT-only V4 model and objective contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .v4_contract import VIEW_FEATURE_COUNTS, VIEW_FEATURE_NAMES, json_sha, sha256_file

V4_CHECKPOINT_SCHEMA = "c2g.v4.corrected.official_v3.quality_detector_checkpoint.v2"
V4_CHECKPOINT_STATUS = "FIT_FOLD_TRAINED_CANDIDATE"


@dataclass(frozen=True)
class V4Normalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_count: int
    view: str

    def __post_init__(self) -> None:
        if self.view not in VIEW_FEATURE_COUNTS or self.feature_count != VIEW_FEATURE_COUNTS[self.view]:
            raise ValueError(f"invalid normalization view/count: {self.view}/{self.feature_count}")
        if len(self.mean) != self.feature_count or len(self.std) != self.feature_count:
            raise ValueError("normalization length mismatch")
        if not all(torch.isfinite(torch.tensor(v)).item() for v in self.mean + self.std):
            raise ValueError("normalization contains NaN/Inf")
        if not all(v > 0.0 for v in self.std):
            raise ValueError("normalization std must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"mean": list(self.mean), "std": list(self.std), "feature_count": self.feature_count, "view": self.view}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V4Normalization":
        return cls(
            tuple(float(v) for v in value["mean"]),
            tuple(float(v) for v in value["std"]),
            int(value["feature_count"]),
            str(value["view"]),
        )

    @property
    def sha256(self) -> str:
        return json_sha(self.to_dict())

    def normalize(self, features: Tensor) -> Tensor:
        mean = torch.tensor(self.mean, dtype=features.dtype, device=features.device)
        std = torch.tensor(self.std, dtype=features.dtype, device=features.device)
        return (features - mean) / std


class V4StatefulQualityGRU(nn.Module):
    """GRUCell whose hidden state changes only on valid, non-padding steps."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, aux_release: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.aux_release = bool(aux_release)
        self.gru = nn.GRUCell(self.input_dim, self.hidden_dim)
        self.quality_head = nn.Linear(self.hidden_dim, 1)
        self.release_head = nn.Linear(self.hidden_dim, 1) if self.aux_release else None

    def forward(self, x: Tensor, step_valid_mask: Tensor, episode_boundaries: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3 or step_valid_mask.shape != x.shape[:2] or episode_boundaries.shape != x.shape[:2]:
            raise ValueError("V4 GRU input shape mismatch")
        batch, steps, _ = x.shape
        hidden = torch.zeros(batch, self.hidden_dim, dtype=x.dtype, device=x.device)
        quality = []
        release = []
        for step in range(steps):
            boundary = episode_boundaries[:, step]
            valid = step_valid_mask[:, step]
            hidden = torch.where(boundary[:, None], torch.zeros_like(hidden), hidden)
            proposed = self.gru(x[:, step], hidden)
            hidden = torch.where(valid[:, None], proposed, hidden)
            quality.append(self.quality_head(hidden).squeeze(-1))
            if self.release_head is not None:
                release.append(self.release_head(hidden).squeeze(-1))
        result = {"quality": torch.stack(quality, dim=1)}
        if self.release_head is not None:
            result["release"] = torch.stack(release, dim=1)
        return result


def compute_v4_loss(
    logits: dict[str, Tensor],
    quality_target: Tensor,
    quality_mask: Tensor,
    *,
    window_id: Optional[Tensor] = None,
    release_target: Optional[Tensor] = None,
    release_mask: Optional[Tensor] = None,
    release_weight: float = 0.3,
    ranking_weight: float = 0.5,
    ranking_margin: float = 0.3,
    hard_negative_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Masked BCE plus differentiable window-level ranking and release BCE."""
    q_logits = logits["quality"]
    mask = quality_mask.to(dtype=torch.bool)
    q_target = quality_target.clamp(0.0, 1.0)
    q_raw = F.binary_cross_entropy_with_logits(q_logits, q_target, reduction="none")
    n_quality = mask.sum().clamp_min(1)
    quality_term = (q_raw * mask.float()).sum() / n_quality
    total = quality_term
    details: dict[str, float] = {"quality_bce": float(quality_term.detach())}

    positive_windows: list[Tensor] = []
    negative_windows: list[Tensor] = []
    if window_id is not None:
        for batch_index in range(q_logits.shape[0]):
            for wid in torch.unique(window_id[batch_index]):
                if int(wid) < 0:
                    continue
                window_mask = mask[batch_index] & (window_id[batch_index] == wid)
                if not bool(window_mask.any()):
                    continue
                positive_mask = window_mask & (q_target[batch_index] > 0.5)
                negative_mask = window_mask & (q_target[batch_index] < 0.5)
                if bool(positive_mask.any()):
                    positive_windows.append(q_logits[batch_index][positive_mask].mean())
                if bool(negative_mask.any()):
                    negative_windows.append(q_logits[batch_index][negative_mask].max())
    if positive_windows and negative_windows:
        positive = torch.stack(positive_windows)
        negative = torch.stack(negative_windows)
        ranking_term = F.relu(ranking_margin - positive[:, None] + negative[None, :]).mean()
    else:
        ranking_term = q_logits.sum() * 0.0
    if ranking_weight:
        total = total + float(ranking_weight) * ranking_term
    details["window_ranking"] = float(ranking_term.detach())
    if negative_windows and hard_negative_weight:
        negative_anchor = F.softplus(torch.stack(negative_windows)).mean()
        total = total + float(hard_negative_weight) * negative_anchor
        details["hard_negative_anchor"] = float(negative_anchor.detach())
    else:
        details["hard_negative_anchor"] = 0.0

    if release_target is not None and "release" in logits and release_weight:
        if release_mask is None:
            raise ValueError("release_mask is required; release_target>=0 is not a valid substitute")
        rmask = release_mask.to(dtype=torch.bool)
        raw = F.binary_cross_entropy_with_logits(
            logits["release"], release_target.clamp(0.0, 1.0), reduction="none"
        )
        n_release = rmask.sum().clamp_min(1)
        release_term = (raw * rmask.float()).sum() / n_release
        total = total + float(release_weight) * release_term
        details["release_bce"] = float(release_term.detach())
    else:
        details["release_bce"] = 0.0
    details["total"] = float(total.detach())
    return total, details


def _write_seal(root: Path) -> tuple[str, str]:
    seal_names = {"SHA256SUMS", "SHA256SUMS.sha256"}
    payloads = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in seal_names),
        key=lambda p: str(p.relative_to(root)),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256_file(path)}  {str(path.relative_to(root)).replace(os.sep, '/')}\n"
            for path in payloads
        ),
        encoding="utf-8",
    )
    sums_sha = sha256_file(sums)
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(sums), sha256_file(root / "SHA256SUMS.sha256")


def save_v4_checkpoint_bundle(
    model: V4StatefulQualityGRU,
    output_dir: Path,
    *,
    view: str,
    aux_release: bool,
    seed: int,
    fold_id: int,
    normalization: V4Normalization,
    losses: list[float],
    protocol_sha256: str,
    feature_protocol_sha256: str,
    teacher_protocol_sha256: str,
    s1_root_sha256: str,
    teacher_root_sha256: str,
    fold_bundle_sha256: str,
    normalization_bundle_sha256: str,
    authorization_sha256: str,
    runner_binding_sha256: str,
    train_identity_sha256: str,
    device: str,
    dtype: str,
    trainer_sha256: str,
    evaluator_sha256: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write a non-overwrite, sealed candidate bundle through a staging dir."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        checkpoint = staging / "checkpoint.pt"
        torch.save(
            {
                "schema": V4_CHECKPOINT_SCHEMA,
                "model_state": model.state_dict(),
                "view": view,
                "aux_release": bool(aux_release),
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "seed": int(seed),
                "fold_id": int(fold_id),
                "normalization": normalization.to_dict(),
                "loss_history": list(losses),
            },
            checkpoint,
        )
        manifest = {
            "schema": V4_CHECKPOINT_SCHEMA,
            "checkpoint_status": V4_CHECKPOINT_STATUS,
            "eligible_for_model_selection": False,
            "formal_attack_authorized": False,
            "view": view,
            "feature_names": list(VIEW_FEATURE_NAMES[view]),
            "feature_count": len(VIEW_FEATURE_NAMES[view]),
            "aux_release": bool(aux_release),
            "hidden_dim": model.hidden_dim,
            "seed": int(seed),
            "fold_id": int(fold_id),
            "train_identity_sha256": train_identity_sha256,
            "normalization_sha256": normalization.sha256,
            "normalization_bundle_sha256": normalization_bundle_sha256,
            "protocol_sha256": protocol_sha256,
            "feature_protocol_sha256": feature_protocol_sha256,
            "teacher_protocol_sha256": teacher_protocol_sha256,
            "s1_root_sha256": s1_root_sha256,
            "teacher_root_sha256": teacher_root_sha256,
            "fold_bundle_sha256": fold_bundle_sha256,
            "authorization_sha256": authorization_sha256,
            "runner_binding_sha256": runner_binding_sha256,
            "trainer_sha256": trainer_sha256,
            "evaluator_sha256": evaluator_sha256,
            "device": device,
            "dtype": dtype,
            "loss_history": list(losses),
            "final_loss": losses[-1] if losses else None,
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        if extra:
            manifest.update(dict(extra))
        (staging / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_seal(staging)
        os.replace(staging, output_dir)
        return str(manifest["checkpoint_sha256"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_v4_authorization(authorization: Mapping[str, Any], **actual_inputs: str) -> dict[str, str]:
    if authorization.get("schema") != "B3_OFFICIAL_V3_DETECTOR_V4_TRAINING_AUTHORIZATION_V1":
        raise ValueError("wrong V4 authorization schema")
    if authorization.get("formal_training_authorized") is not True:
        raise ValueError("formal training is not authorized")
    expected = authorization.get("input_shas", {})
    if not isinstance(expected, Mapping):
        raise ValueError("authorization input_shas missing")
    verified = {}
    for key, value in expected.items():
        if key not in actual_inputs or actual_inputs[key] != value:
            raise ValueError(f"authorization SHA mismatch: {key}")
        verified[key] = str(value)
    return verified
