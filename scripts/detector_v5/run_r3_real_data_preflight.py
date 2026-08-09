"""Small, non-consumable train/validation-only real-data pipeline smoke."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "scripts" / "detector_v5", ROOT / "n5" / "phase3_student"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import sha256_file, verify_seal
from run_r3_full670_student_development import ACTIVE_HEADS, INACTIVE_HEADS, _load_model, _load_records, _loss
from run_r3_heldout_development import _active_masks, _batch, _check_split_closure, _load_g2, _load_splits, _safe_output_root, _write_seal


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: tuple[Any, ...], active: tuple[str, ...]) -> float:
    x, valid, targets, masks, weights = batch
    optimizer.zero_grad(set_to_none=True)
    logits = model(x, timestep_mask=valid)
    loss, _ = _loss(logits, targets, _active_masks(masks, active), weights)
    if not torch.isfinite(loss):
        raise FloatingPointError("preflight loss is not finite")
    loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
        raise FloatingPointError("preflight gradient is not finite")
    optimizer.step()
    return float(loss.detach().cpu())


def _clone_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[dict[str, Any], dict[str, Any]]:
    return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def run(*, g2_root: Path, output_root: Path, per_split: int = 2, seed: int = 20260717) -> dict[str, Any]:
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    transition, binding = _load_g2(g2_root)
    _safe_output_root(output_root, Path(binding["g2_root"]).parent)
    split_ids, split_meta = _load_splits(Path(binding["g1_root"]), "episode", binding["split_manifests"])
    if per_split < 1 or per_split > min(len(split_ids["episode_train"]), len(split_ids["episode_validation"])):
        raise ValueError("invalid deterministic preflight subset size")
    selected = {"train": split_ids["episode_train"][:per_split], "validation": split_ids["episode_validation"][:per_split]}
    selected_ids = set(selected["train"]) | set(selected["validation"])
    if set(selected["train"]) & set(selected["validation"]):
        raise ValueError("preflight train/validation overlap")
    records, record_binding = _load_records(Path(transition["t4"]["root"]), allow_descendant_snapshot=True, identity_allowlist=selected_ids, skip_source_binding=True)
    records_by_id = {row["identity"]: row for row in records}
    _check_split_closure({"episode_train": selected["train"], "episode_validation": selected["validation"]}, records, "episode", loaded_ids=selected_ids)
    mean = np.asarray(split_meta["normalization"]["mean"], dtype=np.float64)
    std = np.asarray(split_meta["normalization"]["std"], dtype=np.float64)
    device = torch.device("cpu")
    train_batch = _batch(records_by_id, selected["train"], mean, std, device)
    val_batch = _batch(records_by_id, selected["validation"], mean, std, device)
    model_cls = _load_model()
    active = tuple(ACTIVE_HEADS)
    model = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_before = _step(model, optimizer, train_batch, active)
    state_model, state_optimizer = _clone_state(model, optimizer)
    resume = model_cls(input_dim=25, hidden=64, short_rf=32, long_rf=128, dropout=0.0).to(device)
    resume_optimizer = torch.optim.AdamW(resume.parameters(), lr=1e-3, weight_decay=1e-5)
    resume.load_state_dict(state_model)
    resume_optimizer.load_state_dict(state_optimizer)
    with torch.no_grad():
        original_logits = model(train_batch[0], timestep_mask=train_batch[1])
        resumed_logits = resume(train_batch[0], timestep_mask=train_batch[1])
    continuation_initial_max_abs = max(float((original_logits[name] - resumed_logits[name]).abs().max()) for name in original_logits)
    loss_after_original = _step(model, optimizer, train_batch, active)
    loss_after_resume = _step(resume, resume_optimizer, train_batch, active)
    continuation_parameter_max_abs = max(float((model.state_dict()[name] - resume.state_dict()[name]).abs().max()) for name in model.state_dict())
    unknown_loss_zero = True
    altered_targets = {head: tensor.clone() for head, tensor in train_batch[2].items()}
    for head in ACTIVE_HEADS:
        altered_targets[head][~train_batch[3][head]] = 1.0 - altered_targets[head][~train_batch[3][head]]
    with torch.no_grad():
        baseline_logits = model(train_batch[0], timestep_mask=train_batch[1])
    baseline_loss, _ = _loss(baseline_logits, train_batch[2], _active_masks(train_batch[3], active), train_batch[4])
    altered_loss, _ = _loss(baseline_logits, altered_targets, _active_masks(train_batch[3], active), train_batch[4])
    unknown_loss_zero = float((baseline_loss - altered_loss).abs()) == 0.0
    safe_grad_max = 0.0
    for name, parameter in model.named_parameters():
        if "safe_release" in name and parameter.grad is not None:
            safe_grad_max = max(safe_grad_max, float(parameter.grad.abs().max()))
    report = {
        "schema": "V5_R3_REAL_DATA_PIPELINE_SMOKE_V1",
        "status": "REAL_DATA_PIPELINE_SMOKE_NONCONSUMABLE",
        "device": "cpu",
        "seed": seed,
        "per_split": per_split,
        "selected_identities": selected,
        "test_payload_read": 0,
        "protected_reads": 0,
        "train_validation_disjoint": True,
        "normalization_source": "sealed_g1_train_only",
        "active_heads": list(active),
        "inactive_heads": list(INACTIVE_HEADS),
        "loss_before": loss_before,
        "loss_after_original": loss_after_original,
        "loss_after_resume": loss_after_resume,
        "finite_forward_backward": True,
        "unknown_loss_contribution": 0.0 if unknown_loss_zero else None,
        "disabled_safe_release_gradient_max_abs": safe_grad_max,
        "checkpoint_continuation": {"initial_logits_max_abs": continuation_initial_max_abs, "parameter_max_abs_after_step": continuation_parameter_max_abs, "pass": continuation_initial_max_abs == 0.0 and continuation_parameter_max_abs <= 1e-7},
        "binding": {"g2_root": binding["g2_root"], "g2_seal_sha256sums_sha256": binding["g2_seal_sha256sums_sha256"], "g1_root": binding["g1_root"], "g1_seal_sha256sums_sha256": binding["g1_seal_sha256sums_sha256"], "t4_root": record_binding["t4_root"], "t4_seal_sha256sums_sha256": record_binding["t4_seal_sha256sums_sha256"], "teacher_root_sha256sums_sha256": record_binding["teacher_root_sha256sums_sha256"], "feature_order_sha256": record_binding["feature_order_sha256"], "normalization_sha256": split_meta["normalization_sha256"], "trainer_sha256": sha256_file(Path(__file__))},
        "permissions": {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": True, "development_inference": True, "test_payload_read": 0, "protected_reads": 0, "safe_release_training": False, "formal_training": False, "full_fit": False, "rollout": False, "attack": False},
    }
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": seed, "active_heads": list(active)}, staging / "preflight_checkpoint.pt")
        (staging / "PREFLIGHT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        if staging.exists() and not (staging / "FAILURE.json").exists():
            (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_REAL_DATA_PIPELINE_SMOKE_FAILURE_V1"}, sort_keys=True) + "\n", encoding="utf-8")
            _write_seal(staging)
        raise
    report["sha256sums_sha256"] = verify_seal(output_root)["sha256sums_sha256"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    print(json.dumps(run(g2_root=args.g2_root, output_root=args.output_root, per_split=args.per_split, seed=args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
