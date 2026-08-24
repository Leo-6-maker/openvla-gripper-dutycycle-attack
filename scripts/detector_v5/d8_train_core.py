"""Shared formal training core for Detector-v3 D8.

P5 smoke, D8-2 CV and final training must use this module for model creation,
normalization, loss, checkpointing and restore/continuation parity.
"""
from __future__ import annotations

import copy
import hashlib
import json
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


def compute_normalization(
    x_train: torch.Tensor,
    epsilon: float = 1e-8,
    *,
    source_identity_digest: str = "",
) -> dict[str, Any]:
    if x_train.ndim != 2 or x_train.shape[1] != FEATURE_DIM:
        raise ValueError(f"normalization expects [N,{FEATURE_DIM}], got {tuple(x_train.shape)}")
    if x_train.shape[0] == 0:
        raise ValueError("normalization cannot be fitted on an empty training set")
    if not torch.isfinite(x_train).all():
        raise ValueError("normalization input contains non-finite values")
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    std = torch.where(std < epsilon, torch.ones_like(std), std)
    return {
        "schema": "D8_NORMALIZATION_V2",
        "method": "zscore",
        "epsilon": epsilon,
        "fit_on": "outer_training_fold_only",
        "mean": mean.detach().cpu().numpy().tolist(),
        "std": std.detach().cpu().numpy().tolist(),
        "train_sample_count": int(x_train.shape[0]),
        "feature_dim": int(x_train.shape[1]),
        "source_identity_digest": source_identity_digest,
    }


def apply_normalization(x: torch.Tensor, norm: dict[str, Any]) -> torch.Tensor:
    if norm.get("schema") != "D8_NORMALIZATION_V2":
        raise ValueError("unsupported normalization schema")
    mean = torch.tensor(norm["mean"], dtype=x.dtype, device=x.device)
    std = torch.tensor(norm["std"], dtype=x.dtype, device=x.device)
    return (x - mean) / std


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError(
            f"loss shape mismatch logits={tuple(logits.shape)} "
            f"targets={tuple(targets.shape)} weights={tuple(weights.shape)}"
        )
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
        raise ValueError("loss inputs contain non-finite values")
    if torch.any(weights <= 0):
        raise ValueError("formal batches may contain only positive weights")
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, reduction="sum"
    )


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not torch.equal(left["torch"], right["torch"]):
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if left_np[0] != right_np[0] or left_np[2:] != right_np[2:]:
        return False
    if not np.array_equal(left_np[1], right_np[1]):
        return False
    if len(left["cuda"]) != len(right["cuda"]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _optimizer_to(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _scalar_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=0.0))
    return left == right


def optimizer_states_equal(
    left: optim.Optimizer,
    right: optim.Optimizer,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> bool:
    if len(left.param_groups) != len(right.param_groups):
        return False
    for group_left, group_right in zip(left.param_groups, right.param_groups):
        keys_left = set(group_left) - {"params"}
        keys_right = set(group_right) - {"params"}
        if keys_left != keys_right:
            return False
        for key in keys_left:
            if not _scalar_equal(group_left[key], group_right[key]):
                return False
        params_left = group_left["params"]
        params_right = group_right["params"]
        if len(params_left) != len(params_right):
            return False
        for param_left, param_right in zip(params_left, params_right):
            state_left = left.state.get(param_left, {})
            state_right = right.state.get(param_right, {})
            if set(state_left) != set(state_right):
                return False
            for key in state_left:
                value_left, value_right = state_left[key], state_right[key]
                if isinstance(value_left, torch.Tensor) or isinstance(value_right, torch.Tensor):
                    if not isinstance(value_left, torch.Tensor) or not isinstance(value_right, torch.Tensor):
                        return False
                    if not torch.allclose(
                        value_left.detach().cpu(),
                        value_right.detach().cpu(),
                        rtol=rtol,
                        atol=atol,
                    ):
                        return False
                elif not _scalar_equal(value_left, value_right):
                    return False
    return True


def model_states_equal(
    left: nn.Module,
    right: nn.Module,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    if set(left_state) != set(right_state):
        return False
    return all(
        torch.allclose(
            left_state[key].detach().cpu(),
            right_state[key].detach().cpu(),
            rtol=rtol,
            atol=atol,
        )
        for key in left_state
    )


def save_checkpoint(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    norm: dict[str, Any],
    output_path: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = extra_metadata or {}
    checkpoint = {
        "schema": "D8_STUDENT_CHECKPOINT_V2",
        "model_state": _cpu_clone(model.state_dict()),
        "optimizer_state": _cpu_clone(optimizer.state_dict()),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "normalization": copy.deepcopy(norm),
        "rng_state": _cpu_clone(capture_rng_state()),
        "feature_schema_sha256": metadata.get("feature_schema_sha256", ""),
        "source_snapshot_sha256": metadata.get("source_snapshot_sha256", ""),
        "executable_source_commit": metadata.get("executable_source_commit", ""),
        "executable_source_tree": metadata.get("executable_source_tree", ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint, str(output_path))
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def load_checkpoint(
    path: Path,
    model: D8StudentDetector,
    optimizer: optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        data = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        data = torch.load(str(path), map_location=map_location)
    if data.get("schema") != "D8_STUDENT_CHECKPOINT_V2":
        raise ValueError(f"unsupported checkpoint schema: {data.get('schema')!r}")
    model.load_state_dict(data["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer_state"])
        _optimizer_to(optimizer, _model_device(model))
    return data


def checkpoint_roundtrip_parity(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    w_batch: torch.Tensor,
    norm: dict[str, Any],
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = torch.device(device)
    x, y, w = x_batch.to(device), y_batch.to(device), w_batch.to(device)
    model.eval()
    with torch.no_grad():
        pre_logits = model(apply_normalization(x, norm)).clone()
        pre_loss = compute_loss(pre_logits, y, w).clone()

    save_checkpoint(
        model, optimizer, 0, 0, norm, checkpoint_path,
        extra_metadata=checkpoint_metadata,
    )

    restored_model = create_model().to(device)
    restored_optimizer = optim.Adam(restored_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(
        checkpoint_path, restored_model, restored_optimizer, map_location=device
    )
    restored_model.eval()
    with torch.no_grad():
        post_logits = restored_model(apply_normalization(x, norm))
        post_loss = compute_loss(post_logits, y, w)

    return {
        "pre_post_logits_match": bool(torch.allclose(pre_logits, post_logits, rtol=1e-6, atol=1e-7)),
        "pre_post_loss_match": bool(torch.allclose(pre_loss, post_loss, rtol=1e-6, atol=1e-7)),
        "max_logit_diff": float((pre_logits - post_logits).abs().max().item()),
        "params_match": model_states_equal(model, restored_model),
        "optimizer_match": optimizer_states_equal(optimizer, restored_optimizer),
        "normalization_match": checkpoint["normalization"] == norm,
    }


def _single_step(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    norm: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(apply_normalization(x, norm))
    loss = compute_loss(logits, y, w)
    loss.backward()
    optimizer.step()
    return logits.detach().clone(), loss.detach().clone()


def continuation_parity(
    model: D8StudentDetector,
    optimizer: optim.Optimizer,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    w_batch: torch.Tensor,
    norm: dict[str, Any],
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
    checkpoint_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare an uninterrupted in-memory branch with a disk-restored branch."""
    device = torch.device(device)
    x, y, w = x_batch.to(device), y_batch.to(device), w_batch.to(device)
    rng_s0 = capture_rng_state()
    save_checkpoint(
        model, optimizer, 0, 0, norm, checkpoint_path,
        extra_metadata=checkpoint_metadata,
    )

    branch_a = copy.deepcopy(model).to(device)
    optimizer_a = optim.Adam(branch_a.parameters(), lr=1e-3)
    optimizer_a.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    _optimizer_to(optimizer_a, device)

    branch_b = create_model().to(device)
    optimizer_b = optim.Adam(branch_b.parameters(), lr=1e-3)
    load_checkpoint(checkpoint_path, branch_b, optimizer_b, map_location=device)

    branch_a.eval()
    branch_b.eval()
    with torch.no_grad():
        pre_logits_a = branch_a(apply_normalization(x, norm)).clone()
        pre_logits_b = branch_b(apply_normalization(x, norm)).clone()
        pre_loss_a = compute_loss(pre_logits_a, y, w).clone()
        pre_loss_b = compute_loss(pre_logits_b, y, w).clone()

    restore_rng_state(rng_s0)
    _single_step(branch_a, optimizer_a, x, y, w, norm)
    rng_after_a = capture_rng_state()

    restore_rng_state(rng_s0)
    _single_step(branch_b, optimizer_b, x, y, w, norm)
    rng_after_b = capture_rng_state()

    branch_a.eval()
    branch_b.eval()
    with torch.no_grad():
        post_logits_a = branch_a(apply_normalization(x, norm)).clone()
        post_logits_b = branch_b(apply_normalization(x, norm)).clone()
        post_loss_a = compute_loss(post_logits_a, y, w).clone()
        post_loss_b = compute_loss(post_logits_b, y, w).clone()

    return {
        "pre_step_logits_match": bool(torch.allclose(pre_logits_a, pre_logits_b, rtol=1e-6, atol=1e-7)),
        "pre_step_loss_match": bool(torch.allclose(pre_loss_a, pre_loss_b, rtol=1e-6, atol=1e-7)),
        "post_step_params_match": model_states_equal(branch_a, branch_b),
        "post_step_optimizer_match": optimizer_states_equal(optimizer_a, optimizer_b),
        "post_step_logits_match": bool(torch.allclose(post_logits_a, post_logits_b, rtol=1e-6, atol=1e-7)),
        "post_step_loss_match": bool(torch.allclose(post_loss_a, post_loss_b, rtol=1e-6, atol=1e-7)),
        "post_step_rng_match": rng_states_equal(rng_after_a, rng_after_b),
        "max_post_logit_diff": float((post_logits_a - post_logits_b).abs().max().item()),
        "pre_loss_a": float(pre_loss_a.item()),
        "pre_loss_b": float(pre_loss_b.item()),
        "post_loss_a": float(post_loss_a.item()),
        "post_loss_b": float(post_loss_b.item()),
    }


def audit_effective_mask(cache_entries: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = {
        "effective_TRUE": 0,
        "effective_FALSE": 0,
        "articulated": 0,
        "RIGHT_CENSORED": 0,
        "GEOM_NA": 0,
        "UNKNOWN_excluded": 0,
        "other": 0,
    }
    issues = []
    for entry in cache_entries:
        effective = entry["effective_mask"]
        target = entry["physical_target"]
        weight = entry["D8_weight"]
        if effective:
            if target == 1.0:
                taxonomy["effective_TRUE"] += 1
            elif target == 0.0:
                taxonomy["effective_FALSE"] += 1
            else:
                issues.append(f"{entry['episode_id']}/{entry['step']}: effective target={target}")
            if not np.isfinite(weight) or weight <= 0:
                issues.append(f"{entry['episode_id']}/{entry['step']}: effective weight={weight}")
        else:
            if entry.get("articulated", False):
                taxonomy["articulated"] += 1
            elif entry.get("right_censored", False):
                taxonomy["RIGHT_CENSORED"] += 1
            elif entry.get("geometry_not_applicable", False):
                taxonomy["GEOM_NA"] += 1
            elif target == -1.0:
                taxonomy["UNKNOWN_excluded"] += 1
            else:
                taxonomy["other"] += 1
            if weight != 0.0:
                issues.append(f"{entry['episode_id']}/{entry['step']}: masked weight={weight}")

    total = sum(taxonomy.values())
    taxonomy["total"] = total
    taxonomy["pass"] = total == 196_483 and not issues
    return {"taxonomy": taxonomy, "issues": issues}


def identity_digest(identities: list[str]) -> str:
    payload = json.dumps(sorted(identities), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
