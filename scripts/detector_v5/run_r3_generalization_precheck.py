"""Synthetic-only G3 pipeline precheck.

It validates the Student training mechanics without loading the 670 records,
features, labels, or any checkpoint.  The output is an engineering receipt,
not a training or evaluation result.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_r3_contact_input import sha256_file, verify_seal
from build_r3_generalization_transition import (
    EXPECTED_PERMISSION_MATRIX,
    _input_root,
    _output_root,
    _validate_permissions,
)
from gripper_attack.seal_utils import rename_noreplace
from run_r3_full670_student_development import ACTIVE_HEADS, HEADS, INACTIVE_HEADS, _load_model, _loss, _restore_optimizer


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _validate_transition(root: Path) -> dict[str, Any]:
    root = _input_root(root, "G2 transition root")
    seal = verify_seal(root)
    transition_path = root / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V1.json"
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    if transition.get("schema") != "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V1" or transition.get("status") != "PASS_G2_DEVELOPMENT_TRANSITION":
        raise ValueError("G2 transition is not passing")
    if type(transition.get("protected_reads")) is not int or transition.get("protected_reads") != 0:
        raise ValueError("G2 model/protected boundary is not closed")
    model_boundary = transition.get("model_boundary", {})
    expected_model_boundary = {
        "random_initialization_required": True,
        "all_670_engineering_checkpoint_allowed": False,
        "checkpoint_consumed": False,
        "privileged_oracle_nondeployable": True,
    }
    if model_boundary != expected_model_boundary or any(type(model_boundary.get(key)) is not bool for key in expected_model_boundary):
        raise ValueError("G2 model initialization boundary is not exact")
    permissions = transition.get("permissions", {})
    if any(type(permissions.get(key)) is not (bool if isinstance(expected, bool) else int) for key, expected in EXPECTED_PERMISSION_MATRIX.items()):
        raise ValueError("G2 permission field types are not exact")
    _validate_permissions(permissions)
    if type(permissions.get("protected_reads")) is not int or permissions["protected_reads"] != 0:
        raise ValueError("G2 permission protected_reads is not integer zero")
    for field in ("formal_training_authorized", "formal_inference_authorized", "shadow_offline_authorized", "shadow_live_authorized", "rollout_authorized", "attack_authorized"):
        if type(transition.get(field)) is not bool or transition.get(field) is not False:
            raise ValueError(f"G2 boundary is not closed: {field}")
    if type(transition.get("teacher_privileged_fields_in_student")) is not bool or type(transition.get("consumable_for_scientific_promotion")) is not bool or transition.get("teacher_privileged_fields_in_student") is not False or transition.get("consumable_for_scientific_promotion") is not False:
        raise ValueError("G2 privileged/scientific promotion boundary is not closed")
    return {"root": str(root.resolve()), "seal_sha256sums_sha256": seal["sha256sums_sha256"], "transition_sha256": sha256_file(transition_path), "transition": transition}


def _batch(device: torch.device) -> tuple[Any, Any, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    x = torch.linspace(-1.0, 1.0, 4 * 12 * 25, dtype=torch.float32, device=device).reshape(4, 12, 25)
    valid = torch.ones((4, 12), dtype=torch.bool, device=device)
    targets = {head: torch.zeros((4, 12), dtype=torch.float32, device=device) for head in HEADS}
    masks = {head: torch.zeros((4, 12), dtype=torch.bool, device=device) for head in HEADS}
    weights = {head: torch.zeros((4, 12), dtype=torch.float32, device=device) for head in HEADS}
    for index, head in enumerate(ACTIVE_HEADS):
        masks[head][:, index:index + 6] = True
        targets[head][:, index:index + 6] = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32, device=device)
        weights[head][masks[head]] = 1.0
    return x, valid, targets, masks, weights


def _max_logit_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((left[key] - right[key]).abs().max().detach().cpu()) for key in left)


def _compute(binding: dict[str, Any], *, seed: int, threads: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    Model = _load_model()
    x, valid, targets, masks, weights = _batch(device)
    active_masks = {head: (masks[head] if head in ACTIVE_HEADS else torch.zeros_like(masks[head])) for head in HEADS}
    model = Model(input_dim=25, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    initial_loss = float(_loss(model(x, timestep_mask=valid), targets, active_masks, weights)[0].detach().cpu())
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss(model(x, timestep_mask=valid), targets, active_masks, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite G3 synthetic loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite G3 synthetic gradient")
        optimizer.step()
    logits = model(x, timestep_mask=valid)
    final_loss = float(_loss(logits, targets, active_masks, weights)[0].detach().cpu())
    unknown_targets = {head: value.clone() for head, value in targets.items()}
    unknown_masks = {head: value.clone() for head, value in active_masks.items()}
    unknown_targets["physical_criticality"][0, 0] = 1.0
    unknown_masks["physical_criticality"][0, 0] = False
    unknown_loss = float(_loss(model(x, timestep_mask=valid), unknown_targets, unknown_masks, weights)[0].detach().cpu())
    if abs(unknown_loss - final_loss) > 1e-7:
        raise AssertionError("UNKNOWN target changed loss")
    model.zero_grad(set_to_none=True)
    loss, _ = _loss(model(x, timestep_mask=valid), targets, active_masks, weights)
    loss.backward()
    safe_module = model.heads[Model.HEAD_NAMES.index("safe_release")]
    disabled_gradient = float(sum(parameter.grad.abs().sum().detach().cpu() for parameter in safe_module.parameters() if parameter.grad is not None))
    if disabled_gradient != 0.0:
        raise AssertionError("disabled safe_release head received gradient")
    checkpoint = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict())}
    resumed = Model(input_dim=25, dropout=0.0).to(device)
    resumed_optimizer = _restore_optimizer(resumed, checkpoint["optimizer"], learning_rate=1e-3, weight_decay=1e-5)
    resumed.load_state_dict(checkpoint["model"], strict=True)
    resume_diff = _max_logit_diff(logits, resumed(x, timestep_mask=valid))
    if resume_diff > 1e-7:
        raise AssertionError(f"resume diff {resume_diff}")
    continued = Model(input_dim=25, dropout=0.0).to(device)
    continued_optimizer = _restore_optimizer(continued, checkpoint["optimizer"], learning_rate=1e-3, weight_decay=1e-5)
    continued.load_state_dict(checkpoint["model"], strict=True)
    torch.set_num_threads(1)
    try:
        for branch, branch_optimizer in ((resumed, resumed_optimizer), (continued, continued_optimizer)):
            branch_optimizer.zero_grad(set_to_none=True)
            branch_loss, _ = _loss(branch(x, timestep_mask=valid), targets, active_masks, weights)
            branch_loss.backward()
            branch_optimizer.step()
        continuation_diff = _max_logit_diff(resumed(x, timestep_mask=valid), continued(x, timestep_mask=valid))
    finally:
        torch.set_num_threads(threads)
    if continuation_diff > 1e-7:
        raise AssertionError(f"continuation diff {continuation_diff}")
    report = {
        "schema": "V5_R3_GENERALIZATION_G3_PRECHECK_V1",
        "status": "PASS_G3_SYNTHETIC_PIPELINE_PRECHECK",
        "seed": seed, "threads": threads, "device": str(device),
        "synthetic_only": True, "production_records_loaded": False,
        "all_670_checkpoint_loaded": False, "initial_loss": initial_loss, "final_loss": final_loss,
        "finite_forward_backward": True, "unknown_loss_contribution": 0.0,
        "disabled_head_gradient": {head: 0.0 for head in INACTIVE_HEADS},
        "checkpoint_resume_max_logit_diff": resume_diff,
        "checkpoint_continuation_max_logit_diff": continuation_diff,
        "binding": binding, "protected_reads": 0,
        "formal_training_authorized": False, "formal_inference_authorized": False,
        "shadow_offline_authorized": False, "rollout_authorized": False, "attack_authorized": False,
    }
    return report


def run(*, transition_root: Path, output_root: Path, seed: int = 20260717, threads: int = 4) -> dict[str, Any]:
    transition_root = _input_root(transition_root, "G2 transition root")
    output_root = _output_root(output_root, transition_root.parent)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        report = _compute(_validate_transition(transition_root), seed=seed, threads=threads)
        (staging / "G3_PRECHECK_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception as exc:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_G3_PRECHECK_FAILURE_V1", "error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        raise
    report["sha256sums_sha256"] = digest
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(transition_root=args.transition_root, output_root=args.output_root, seed=args.seed, threads=args.threads), indent=2, sort_keys=True))
