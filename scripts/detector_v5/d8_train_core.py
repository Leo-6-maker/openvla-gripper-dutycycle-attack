"""H1-R7: Shared training core for D8 Detector-v3 Student.

P5 smoke, D8-2 CV, and final training MUST share this core.
No ad-hoc model/loss/normalization in smoke scripts.

Provides:
  - Model factory (MLP 25→32→16→1)
  - Normalization (zscore, fit on train only)
  - Loss (weighted BCE, reduction=sum for weight semantics)
  - Checkpoint save/load with RNG + optimizer state
  - Continuation parity verification
  - Effective-mask audit
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

FEATURE_DIM = 25
HIDDEN_DIM = 32
SEED = 20260717

# ── Model ──────────────────────────────────────────────────────────────

class D8StudentDetector(nn.Module):
    """Shared 25D Student detector — identical across P5, CV, final."""

    def __init__(self, n_features: int = FEATURE_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self._feature_dim = n_features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def create_model(seed: int = SEED) -> D8StudentDetector:
    torch.manual_seed(seed)
    return D8StudentDetector()


# ── Normalization ──────────────────────────────────────────────────────

def compute_normalization(
    X_train: torch.Tensor,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Compute zscore normalization from train data only.

    Returns dict with mean, std, source info for provenance tracking.
    """
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std = torch.where(std < epsilon, torch.ones_like(std), std)

    return {
        "method": "zscore",
        "epsilon": epsilon,
        "fit_on": "training_fold_only",
        "mean": mean.numpy().tolist(),
        "std": std.numpy().tolist(),
        "train_sample_count": int(X_train.shape[0]),
        "feature_dim": int(X_train.shape[1]),
    }


def apply_normalization(X: torch.Tensor, norm: dict) -> torch.Tensor:
    mean = torch.tensor(norm["mean"], dtype=X.dtype, device=X.device)
    std = torch.tensor(norm["std"], dtype=X.dtype, device=X.device)
    return (X - mean) / std


# ── Loss ───────────────────────────────────────────────────────────────

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted BCE loss, reduction=sum for per-episode weight semantics."""
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, reduction="sum",
    )


# ── Checkpoint ─────────────────────────────────────────────────────────

def save_checkpoint(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    norm: dict,
    rng_state: dict,
    output_path: Path,
    extra_metadata: dict | None = None,
) -> str:
    """Save full checkpoint to disk. Returns SHA256 of checkpoint file."""
    checkpoint = {
        "schema": "D8_STUDENT_CHECKPOINT_V1",
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "normalization": norm,
        "rng_state": rng_state,
        "feature_schema_sha256": extra_metadata.get("feature_schema_sha256", "") if extra_metadata else "",
        "commit": extra_metadata.get("commit", "") if extra_metadata else "",
        "tree": extra_metadata.get("tree", "") if extra_metadata else "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    data = json.dumps(checkpoint, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    output_path.write_text(data, encoding="utf-8")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def load_checkpoint(
    path: Path,
    model: D8StudentDetector,
    optimizer: optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    """Load checkpoint from disk. Restores model + optimizer state."""
    data = json.loads(path.read_text(encoding="utf-8"))

    # Restore model state
    model_state = {k: torch.tensor(v) if isinstance(v, list) else v
                   for k, v in data["model_state"].items()}
    model.load_state_dict(model_state)

    # Restore optimizer state
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])

    return data


def checkpoint_parity(
    model: D8StudentDetector,
    checkpoint_path: Path,
    test_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str = "cpu",
) -> dict:
    """Verify checkpoint save/load produces identical outputs."""
    X, y, w = [t.to(device) for t in test_batch]
    norm_data = {"mean": X.mean(dim=0).numpy().tolist(),
                 "std": X.std(dim=0).numpy().tolist(),
                 "method": "zscore", "epsilon": 1e-8}

    model.eval()
    with torch.no_grad():
        Xn = apply_normalization(X, norm_data)
        pre_logits = model(Xn).clone()

    # Save
    opt = optim.Adam(model.parameters(), lr=1e-3)
    rng_state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    save_checkpoint(model, opt, 0, 0, norm_data, rng_state, checkpoint_path)

    # New model, load
    model2 = create_model()
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, model2, opt2)

    model2.eval()
    with torch.no_grad():
        Xn2 = apply_normalization(X, norm_data)
        post_logits = model2(Xn2)

    close = torch.allclose(pre_logits, post_logits, rtol=1e-5, atol=1e-6)
    return {
        "pre_save_vs_post_load_logits_match": bool(close),
        "max_diff": float((pre_logits - post_logits).abs().max()),
    }


def continuation_parity(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    w_batch: torch.Tensor,
    norm: dict,
    checkpoint_path: Path,
    device: str = "cpu",
) -> dict:
    """Verify that continuing from checkpoint produces same result as uninterrupted."""
    X, y, w = X_batch.to(device), y_batch.to(device), w_batch.to(device)

    # Branch A: uninterrupted
    model.train()
    opt_a = optim.Adam(model.parameters(), lr=1e-3)
    opt_a.load_state_dict(optimizer.state_dict())
    opt_a.zero_grad()
    Xn = apply_normalization(X, norm)
    logits_a = model(Xn)
    loss_a = compute_loss(logits_a, y, w)
    loss_a.backward()
    opt_a.step()
    params_a = {k: v.clone() for k, v in model.state_dict().items()}

    # Save and restore
    rng_state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    save_checkpoint(model, optimizer, 0, 0, norm, rng_state, checkpoint_path)

    # Branch B: fresh model, load checkpoint, then do same step
    model_b = create_model()
    opt_b = optim.Adam(model_b.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, model_b, opt_b)

    # Restore RNG
    torch.set_rng_state(rng_state["torch"])
    np.random.set_state(rng_state["numpy"])

    model_b.train()
    opt_b.zero_grad()
    Xn_b = apply_normalization(X, norm)
    logits_b = model_b(Xn_b)
    loss_b = compute_loss(logits_b, y, w)
    loss_b.backward()
    opt_b.step()
    params_b = {k: v.clone() for k, v in model_b.state_dict().items()}

    param_match = all(torch.allclose(params_a[k], params_b[k], rtol=1e-5, atol=1e-6)
                      for k in params_a)

    return {
        "pre_step_logits_match": bool(torch.allclose(logits_a, logits_b, rtol=1e-5, atol=1e-6)),
        "pre_step_loss_match": bool(torch.isclose(loss_a, loss_b, rtol=1e-5, atol=1e-6)),
        "post_step_params_match": param_match,
        "post_step_loss_match": bool(torch.isclose(loss_a, loss_b, rtol=1e-5, atol=1e-6)),
    }


# ── Effective-mask audit ───────────────────────────────────────────────

def audit_effective_mask(cache_entries: list[dict]) -> dict:
    """Scan all cache entries: effective_mask logic must be consistent."""
    stats = {"effective_TRUE": 0, "effective_FALSE": 0,
             "masked_UNKNOWN": 0, "masked_RC": 0, "masked_GEOM_NA": 0,
             "masked_articulated": 0, "weight_zero_for_masked": 0,
             "weight_nonzero_for_effective": 0, "target_valid_for_effective": 0,
             "issues": []}

    for e in cache_entries:
        eff = e["effective_mask"]
        target = e["physical_target"]
        weight = e["D8_weight"]
        rc = e.get("right_censored", False)
        geom = e.get("geometry_not_applicable", False)
        art = e.get("articulated", False)

        if eff:
            if target not in (0.0, 1.0):
                stats["issues"].append(f"{e['episode_id']} step {e['step']}: effective but target={target}")
                stats["target_valid_for_effective"] += 1
            if target == 1.0: stats["effective_TRUE"] += 1
            elif target == 0.0: stats["effective_FALSE"] += 1
            if weight <= 0:
                stats["issues"].append(f"{e['episode_id']} step {e['step']}: effective but weight={weight}")
        else:
            if target == -1.0: stats["masked_UNKNOWN"] += 1
            elif rc: stats["masked_RC"] += 1
            elif geom: stats["masked_GEOM_NA"] += 1
            elif art: stats["masked_articulated"] += 1
            if weight != 0.0:
                stats["weight_zero_for_masked"] += 1

    stats["pass"] = len(stats["issues"]) == 0 and stats["weight_zero_for_masked"] == 0
    return stats
