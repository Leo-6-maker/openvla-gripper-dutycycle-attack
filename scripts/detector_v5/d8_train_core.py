"""H1-R7: Shared training core for D8 Detector-v3 Student.

P5 smoke, D8-2 CV, and final training MUST share this core.
"""
from __future__ import annotations

import hashlib
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


class D8StudentDetector(nn.Module):
    def __init__(self, n_features: int = FEATURE_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self._feature_dim = n_features

    @property
    def feature_dim(self) -> int: return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def create_model(seed: int = SEED) -> D8StudentDetector:
    torch.manual_seed(seed)
    return D8StudentDetector()


# ── Normalization ──────────────────────────────────────────────────────

def compute_normalization(X_train: torch.Tensor, epsilon: float = 1e-8) -> dict:
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std = torch.where(std < epsilon, torch.ones_like(std), std)
    return {
        "method": "zscore", "epsilon": epsilon, "fit_on": "training_fold_only",
        "mean": mean.numpy().tolist(), "std": std.numpy().tolist(),
        "train_sample_count": int(X_train.shape[0]), "feature_dim": int(X_train.shape[1]),
    }


def apply_normalization(X: torch.Tensor, norm: dict) -> torch.Tensor:
    mean = torch.tensor(norm["mean"], dtype=X.dtype, device=X.device)
    std = torch.tensor(norm["std"], dtype=X.dtype, device=X.device)
    return (X - mean) / std


# ── Loss ───────────────────────────────────────────────────────────────

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")


# ── Checkpoint (P0-6: torch.save/load) ─────────────────────────────────

def save_checkpoint(
    model: D8StudentDetector, optimizer: optim.Optimizer,
    epoch: int, global_step: int, norm: dict,
    output_path: Path, extra_metadata: dict | None = None,
) -> str:
    checkpoint = {
        "schema": "D8_STUDENT_CHECKPOINT_V1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step,
        "normalization": norm,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "feature_schema_sha256": (extra_metadata or {}).get("feature_schema_sha256", ""),
        "commit": (extra_metadata or {}).get("commit", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, str(output_path))
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def load_checkpoint(
    path: Path, model: D8StudentDetector,
    optimizer: optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict:
    data = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    return data


# ── Checkpoint parity (P0-7: correct branch logic) ─────────────────────

def checkpoint_roundtrip_parity(
    model: D8StudentDetector, optimizer: optim.Optimizer,
    X_batch: torch.Tensor, y_batch: torch.Tensor, w_batch: torch.Tensor,
    norm: dict, checkpoint_path: Path, device: str = "cpu",
) -> dict:
    """Verify save→destroy→load→same output."""
    X, y, w = X_batch.to(device), y_batch.to(device), w_batch.to(device)

    model.eval()
    with torch.no_grad():
        pre_logits = model(apply_normalization(X, norm)).clone()

    # Save current state
    save_checkpoint(model, optimizer, 0, 0, norm, checkpoint_path)

    # Destroy and rebuild
    model2 = create_model()
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, model2, opt2)

    model2.eval()
    with torch.no_grad():
        post_logits = model2(apply_normalization(X, norm))

    return {
        "pre_post_logits_match": bool(torch.allclose(pre_logits, post_logits, rtol=1e-5, atol=1e-6)),
        "max_logit_diff": float((pre_logits - post_logits).abs().max()),
        "params_match": all(
            torch.allclose(model.state_dict()[k], model2.state_dict()[k], rtol=1e-5, atol=1e-6)
            for k in model.state_dict()
        ),
    }


def continuation_parity(
    model: D8StudentDetector, optimizer: optim.Optimizer,
    X_batch: torch.Tensor, y_batch: torch.Tensor, w_batch: torch.Tensor,
    norm: dict, checkpoint_path: Path, device: str = "cpu",
) -> dict:
    """P0-7: Correct continuation parity.

    From common starting point S0:
    - Branch A: one optimizer step
    - Branch B: save S0 → destroy → load → same RNG → same batch → one step
    Compare pre-step and post-step states.
    """
    X, y, w = X_batch.to(device), y_batch.to(device), w_batch.to(device)

    # Freeze common starting point S0
    rng_torch_s0 = torch.get_rng_state()
    rng_numpy_s0 = np.random.get_state()
    if torch.cuda.is_available():
        rng_cuda_s0 = torch.cuda.get_rng_state_all()

    # Save S0 to disk
    save_checkpoint(model, optimizer, 0, 0, norm, checkpoint_path,
                    extra_metadata={"commit": "", "feature_schema_sha256": ""})

    # Pre-step: record from S0
    model.eval()
    with torch.no_grad():
        Xn = apply_normalization(X, norm)
        pre_logits = model(Xn).clone()
    pre_loss = float(compute_loss(pre_logits, y, w))

    # ── Branch A: step from S0 ──
    torch.set_rng_state(rng_torch_s0)
    np.random.set_state(rng_numpy_s0)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_cuda_s0)

    model.train()
    opt_a = optim.Adam(model.parameters(), lr=1e-3)
    opt_a.load_state_dict(optimizer.state_dict())
    opt_a.zero_grad()
    logits_a = model(apply_normalization(X, norm))
    loss_a = compute_loss(logits_a, y, w)
    loss_a.backward()
    opt_a.step()
    params_a = {k: v.clone() for k, v in model.state_dict().items()}
    post_loss_a = float(loss_a)

    # ── Branch B: restore from disk S0, then step ──
    model_b = create_model()
    opt_b = optim.Adam(model_b.parameters(), lr=1e-3)
    ck = load_checkpoint(checkpoint_path, model_b, opt_b)

    torch.set_rng_state(rng_torch_s0)
    np.random.set_state(rng_numpy_s0)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_cuda_s0)

    model_b.train()
    opt_b.zero_grad()
    logits_b = model_b(apply_normalization(X, norm))
    loss_b = compute_loss(logits_b, y, w)
    loss_b.backward()
    opt_b.step()
    params_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    post_loss_b = float(loss_b)

    # Compare
    pre_logits_match = bool(torch.allclose(pre_logits, logits_b, rtol=1e-5, atol=1e-6))
    post_params_match = all(
        torch.allclose(params_a[k], params_b[k], rtol=1e-5, atol=1e-6)
        for k in params_a
    )
    post_loss_match = abs(post_loss_a - post_loss_b) < 1e-5

    return {
        "pre_step_logits_match": pre_logits_match,
        "pre_step_loss_match": abs(pre_loss - float(loss_b.detach())) < 1e-5,
        "post_step_params_match": post_params_match,
        "post_step_loss_match": post_loss_match,
        "pre_loss": pre_loss, "post_loss_a": post_loss_a, "post_loss_b": post_loss_b,
    }


# ── Effective-mask audit ───────────────────────────────────────────────

def audit_effective_mask(cache_entries: list[dict]) -> dict:
    """Mutually exclusive taxonomy scan of all 196,483 entries."""
    taxonomy = {"effective_TRUE": 0, "effective_FALSE": 0,
                "articulated": 0, "RIGHT_CENSORED": 0, "GEOM_NA": 0,
                "UNKNOWN_excluded": 0, "other": 0}
    issues = []

    for e in cache_entries:
        eff = e["effective_mask"]
        target = e["physical_target"]
        weight = e["D8_weight"]
        rc = e.get("right_censored", False)
        geom = e.get("geometry_not_applicable", False)
        art = e.get("articulated", False)

        if eff:
            if target == 1.0: taxonomy["effective_TRUE"] += 1
            elif target == 0.0: taxonomy["effective_FALSE"] += 1
            else: issues.append(f"{e['episode_id']}/{e['step']}: effective target={target}")
            if weight <= 0: issues.append(f"{e['episode_id']}/{e['step']}: effective weight={weight}")
        else:
            if art: taxonomy["articulated"] += 1
            elif rc: taxonomy["RIGHT_CENSORED"] += 1
            elif geom: taxonomy["GEOM_NA"] += 1
            elif target == -1.0: taxonomy["UNKNOWN_excluded"] += 1
            else: taxonomy["other"] += 1
            if weight != 0.0: issues.append(f"{e['episode_id']}/{e['step']}: masked weight={weight}")

    total = sum(taxonomy.values())
    taxonomy["total"] = total
    taxonomy["pass"] = (total == 196483) and len(issues) == 0
    return {"taxonomy": taxonomy, "issues": issues}
