"""Detector V4 formal model, normalization, loss, and checkpoint contracts.

Based on B3 formal shell patterns but with:
- Variable-dim feature views (25D/33D/39D) instead of fixed 25D
- Single quality head + optional auxiliary release head
- Masked BCE + differentiable cross-episode ranking + release BCE loss
- V4 checkpoint schema with candidate-only status
"""

from __future__ import annotations

import hashlib, json, os, shutil, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── schemas ────────────────────────────────────────────────────────────
V4_CHECKPOINT_SCHEMA = "c2g.v4.official_v3.quality_detector_checkpoint.v1"
V4_CHECKPOINT_STATUS = "FIT_FOLD_TRAINED_CANDIDATE"
V4_MODEL_SELECTION_ELIGIBLE = False
V4_ATTACK_AUTHORIZED = False

V4_HEADS_BASE = ("quality",)
V4_HEADS_RELEASE = ("quality", "release")

FEATURE_NAMES_A = tuple(f"f25d_{i}" for i in range(25))
FEATURE_NAMES_B = FEATURE_NAMES_A + (
    "delta_gripper_qpos", "delta2_gripper_qpos",
    "gripper_command_qpos_deviation", "close_dwell_duration",
    "time_since_close_onset", "recent_close_count",
    "opening_trend", "recent_command_variance",
)
FEATURE_NAMES_C = FEATURE_NAMES_B + (
    "eef_velocity", "eef_acceleration", "eef_vertical_velocity",
    "eef_stability", "eef_displacement_since_close_onset",
    "action_consistency",
)
VIEW_FEATURE_NAMES = {"A": FEATURE_NAMES_A, "B": FEATURE_NAMES_B, "C": FEATURE_NAMES_C}
VIEW_FEATURE_COUNTS = {"A": 25, "B": 33, "C": 39}


# ── helpers ────────────────────────────────────────────────────────────
def json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _is_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdefABCDEF" for c in value)


# ── normalization ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class V4Normalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_count: int
    view: str

    def __post_init__(self) -> None:
        if len(self.mean) != self.feature_count or len(self.std) != self.feature_count:
            raise ValueError(f"normalization expects {self.feature_count} values")
        if not all(v > 0.0 for v in self.std):
            raise ValueError("std must be positive")
        if not all(torch.isfinite(torch.tensor(v, dtype=torch.float64)).item() for v in self.mean + self.std):
            raise ValueError("normalization values must be finite")

    @property
    def sha256(self) -> str:
        return json_sha({"mean": list(self.mean), "std": list(self.std),
                         "feature_count": self.feature_count, "view": self.view})

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "std": list(self.std),
                "feature_count": self.feature_count, "view": self.view}

    @classmethod
    def from_dict(cls, d: Mapping) -> "V4Normalization":
        return cls(tuple(float(v) for v in d["mean"]), tuple(float(v) for v in d["std"]),
                   int(d["feature_count"]), str(d["view"]))

    def normalize(self, features: Tensor) -> Tensor:
        m = torch.tensor(self.mean, dtype=features.dtype, device=features.device)
        s = torch.tensor(self.std, dtype=features.dtype, device=features.device)
        return (features - m) / s


# ── model ──────────────────────────────────────────────────────────────
class V4StatefulQualityGRU(nn.Module):
    """Stateful GRU with per-step valid_mask gating (B3 pattern).

    Invalid steps do NOT update hidden state. Padding steps do NOT update.
    Hidden resets only at episode boundaries.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 aux_release: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.aux_release = aux_release
        if aux_release:
            self.release_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, step_valid_mask: Tensor,
                episode_boundaries: Tensor) -> dict[str, Tensor]:
        """Process batch with step-level state gating.

        Args:
            x: [B, T, F] normalized features
            step_valid_mask: [B, T] bool — True = valid step, False = skip (no hidden update)
            episode_boundaries: [B, T] bool — True at episode start (reset hidden)

        Returns:
            logits dict with 'quality' and optionally 'release' keys, each [B, T]
        """
        B, T, _ = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)

        quality_logits = torch.zeros(B, T, device=device)
        release_logits = torch.zeros(B, T, device=device) if self.aux_release else None

        for t in range(T):
            xt = x[:, t, :]          # [B, F]
            valid = step_valid_mask[:, t]  # [B]
            boundary = episode_boundaries[:, t]  # [B]

            # Reset hidden at episode boundaries
            h = torch.where(boundary.unsqueeze(1), torch.zeros_like(h), h)

            # Compute next hidden for valid steps
            h_new = self.gru(xt, h)  # [B, H]

            # Only update hidden for valid steps
            h = torch.where(valid.unsqueeze(1), h_new, h)

            # Apply heads
            quality_logits[:, t] = self.quality_head(h).squeeze(-1)
            if self.aux_release:
                release_logits[:, t] = self.release_head(h).squeeze(-1)

        result = {"quality": quality_logits}
        if self.aux_release:
            result["release"] = release_logits
        return result


# ── loss ────────────────────────────────────────────────────────────────
def compute_v4_loss(logits: dict[str, Tensor],
                    quality_target: Tensor, quality_mask: Tensor,
                    release_target: Optional[Tensor] = None,
                    release_weight: float = 0.3,
                    ranking_weight: float = 0.5,
                    ranking_margin: float = 0.3,
                    ) -> tuple[Tensor, dict[str, float]]:
    """Compute V4 quality loss with optional ranking and release terms.

    Returns (total_loss, loss_components_dict).
    """
    components = {}
    q_logits = logits["quality"]
    device = q_logits.device

    # Quality BCE (masked)
    q_loss = F.binary_cross_entropy_with_logits(q_logits, quality_target, reduction="none")
    n_q = quality_mask.sum().clamp_min(1)
    quality_term = (q_loss * quality_mask.float()).sum() / n_q
    components["quality_bce"] = float(quality_term.detach())

    total = quality_term

    # Ranking: mean quality logit on positive steps > mean on negative steps
    if ranking_weight > 0:
        q_probs = torch.sigmoid(q_logits)
        pos_scores = []
        neg_scores = []
        for b in range(q_logits.shape[0]):
            m = quality_mask[b]
            if m.sum() == 0:
                continue
            t = quality_target[b][m]
            s = q_probs[b][m]  # stays in graph
            if (t > 0.5).any():
                pos_scores.append(s[t > 0.5].mean())
            if (t < 0.5).any():
                neg_scores.append(s[t < 0.5].mean())

        if len(pos_scores) > 0 and len(neg_scores) > 0:
            pos_t = torch.stack(pos_scores)
            neg_t = torch.stack(neg_scores)
            pos_exp = pos_t.unsqueeze(1).expand(-1, len(neg_t))
            neg_exp = neg_t.unsqueeze(0).expand(len(pos_t), -1)
            rank_loss = F.relu(ranking_margin - (pos_exp - neg_exp)).mean()
            total = total + ranking_weight * rank_loss
            components["ranking"] = float(rank_loss.detach())
        else:
            components["ranking"] = 0.0

    # Release auxiliary BCE
    if release_target is not None and release_weight > 0 and "release" in logits:
        r_logits = logits["release"]
        r_target = release_target.clamp(0, 1)
        r_mask = (release_target >= 0).float()
        n_r = r_mask.sum().clamp_min(1)
        r_loss = F.binary_cross_entropy_with_logits(r_logits, r_target, reduction="none")
        release_term = (r_loss * r_mask).sum() / n_r
        total = total + release_weight * release_term
        components["release_bce"] = float(release_term.detach())

    components["total"] = float(total.detach())
    return total, components


# ── checkpoint ──────────────────────────────────────────────────────────
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
    s1_root_sha256: str,
    v21_root_sha256: str,
    fold_bundle_sha256: str,
    normalization_bundle_sha256: str,
    runner_binding_sha256: str,
    train_episode_count: int,
) -> str:
    """Save sealed checkpoint bundle. Returns checkpoint SHA256."""
    output_dir.mkdir(parents=True, exist_ok=False)

    # Save model
    checkpoint_pt = output_dir / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "view": view, "aux_release": aux_release,
        "input_dim": model.input_dim, "hidden_dim": model.hidden_dim,
        "seed": seed, "fold_id": fold_id,
        "losses": losses,
        "normalization": normalization.to_dict(),
    }, checkpoint_pt)

    # Manifest
    manifest = {
        "schema": V4_CHECKPOINT_SCHEMA,
        "checkpoint_status": V4_CHECKPOINT_STATUS,
        "eligible_for_model_selection": V4_MODEL_SELECTION_ELIGIBLE,
        "formal_attack_authorized": V4_ATTACK_AUTHORIZED,
        "view": view, "aux_release": aux_release,
        "seed": seed, "fold_id": fold_id,
        "train_episode_count": train_episode_count,
        "final_loss": losses[-1] if losses else None,
        "normalization_sha256": normalization.sha256,
        "protocol_sha256": protocol_sha256,
        "s1_root_sha256": s1_root_sha256,
        "v21_root_sha256": v21_root_sha256,
        "fold_bundle_sha256": fold_bundle_sha256,
        "normalization_bundle_sha256": normalization_bundle_sha256,
        "runner_binding_sha256": runner_binding_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_pt),
    }
    with open(output_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # SHA256SUMS
    SEAL_FILES = {"SHA256SUMS", "SHA256SUMS.sha256"}
    file_list = sorted(
        [f for f in output_dir.rglob("*") if f.is_file() and f.name not in SEAL_FILES],
        key=lambda f: str(f.relative_to(output_dir)))
    with open(output_dir / "SHA256SUMS", "w", encoding="utf-8") as fh:
        for fp in file_list:
            rel = str(fp.relative_to(output_dir))
            fh.write(f"{sha256_file(fp)}  {rel}\n")
    sha = sha256_file(output_dir / "SHA256SUMS")
    with open(output_dir / "SHA256SUMS.sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{sha}  SHA256SUMS\n")

    return manifest["checkpoint_sha256"]


def validate_v4_authorization(authorization: dict, **inputs: dict) -> dict:
    """Verify all authorized input SHAs match actual inputs. Returns verified dict."""
    results = {}
    for name, expected in authorization.get("input_shas", {}).items():
        actual = inputs.get(name)
        if actual is None:
            raise ValueError(f"missing authorized input: {name}")
        if actual != expected:
            raise ValueError(f"authorized SHA mismatch for {name}: expected {expected}, got {actual}")
        results[name] = actual
    return results
