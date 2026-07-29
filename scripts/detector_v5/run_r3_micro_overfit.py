"""Non-consumable R3-3E canary learnability check."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_features import ACTION_GRIPPER_SOURCE, FEATURE_ORDER, load_feature_binding, materialize_fit670_features
from gripper_attack.v5_r3_student import MODEL_HEAD_ALIASES
from gripper_attack.v5_r3_teacher import HEADS
from audit_r3_contact_input import load_consumable_episodes, sha256_file, verify_seal


ACTIVE_HEADS = ("k10_feasibility", "instability", "gripper_closing_state")
MODEL_HEADS = {"k10_feasibility": "k10_feasible"}
INACTIVE_HEADS = tuple(head for head in HEADS if head not in ACTIVE_HEADS)
LABEL_VALUES = {"TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"}


def _is_false_flag(value: Any) -> bool:
    return value is False or (type(value) is int and value == 0)


def _known_label(label: Mapping[str, Any], head: str) -> bool:
    if not isinstance(label, Mapping) or label.get("value") not in LABEL_VALUES:
        raise ValueError(f"invalid {head} label value")
    for field in ("valid_mask", "mask", "right_censored"):
        if type(label.get(field)) is not bool:
            raise ValueError(f"invalid {head} label field: {field}")
    return head in ACTIVE_HEADS and label["valid_mask"] and label["mask"] and not label["right_censored"] and label["value"] in {"TRUE", "FALSE"}


def _load_model():
    model_root = ROOT / "n5" / "phase3_student"
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    from n5_student_model import N5MultiHeadStudent
    return N5MultiHeadStudent


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _event_weights(candidates: list[bool], known: list[bool]) -> np.ndarray:
    """Give each contiguous causal event equal total weight."""
    weights = np.zeros(len(candidates), dtype=np.float32)
    start = 0
    while start < len(candidates):
        state = bool(candidates[start])
        end = start + 1
        while end < len(candidates) and bool(candidates[end]) == state:
            end += 1
        known_count = sum(bool(value) for value in known[start:end])
        if known_count:
            weights[start:end] = np.asarray(known[start:end], dtype=np.float32) / float(known_count)
        start = end
    return weights


def _load_data(input_root: Path, teacher_root: Path, transition: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, episodes, input_seal = load_consumable_episodes(input_root, expected_count=8, transition_manifest_path=transition)
    teacher_seal = verify_seal(teacher_root)
    teacher_manifest = json.loads((teacher_root / "teacher_manifest.json").read_text(encoding="utf-8"))
    transition_data = json.loads(transition.read_text(encoding="utf-8"))
    if teacher_manifest.get("status") != "DEVELOPMENT_NONCONSUMABLE":
        raise ValueError("teacher manifest is not the non-consumable canary contract")
    for field in ("protected_reads", "formal_inference_authorized", "formal_training_authorized", "attack_authorized"):
        if not _is_false_flag(teacher_manifest.get(field)):
            raise ValueError(f"teacher manifest {field} is not false")
    if transition_data.get("protected_payload_read") is not False or transition_data.get("attack_authorized") is not False:
        raise ValueError("transition is not FIT-only")
    if transition_data.get("identity_set_digest") != manifest.get("identity_set_digest"):
        raise ValueError("transition/input identity-set mismatch")
    if transition_data.get("collection_source_commit") != manifest.get("collection_source_commit") or transition_data.get("collection_source_tree") != manifest.get("collection_source_tree"):
        raise ValueError("transition/input source lineage mismatch")
    if teacher_manifest.get("source_root") != str(input_root.resolve()):
        raise ValueError("teacher/input source-root mismatch")
    if teacher_manifest.get("transition_manifest_sha256") != sha256_file(transition):
        raise ValueError("teacher/transition manifest mismatch")
    if teacher_manifest.get("identity_allowlist_sha256") != transition_data.get("identity_allowlist_file_sha256"):
        raise ValueError("teacher/transition allowlist mismatch")
    if teacher_manifest.get("input_sha256sums_sha256") != input_seal["sha256sums_sha256"]:
        raise ValueError("teacher/input seal mismatch")
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for line in (teacher_root / "teacher_records.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["episode_id"]), int(row["step"]))
        if key in labels:
            raise ValueError(f"duplicate teacher row: {key}")
        labels[key] = row
    records: list[dict[str, Any]] = []
    feature_binding_path = ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json"
    feature_binding = load_feature_binding(feature_binding_path, ROOT)
    feature_source_sha = feature_binding["adapter_source_sha256"]
    feature_binding_sha = sha256_file(feature_binding_path)
    feature_order_sha = hashlib.sha256(json.dumps(list(FEATURE_ORDER), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    for item in episodes:
        identity = str(item["manifest"]["episode_id"])
        raw_path = input_root / item["manifest"]["relative_path"]
        raw_episode = json.loads(raw_path.read_text(encoding="utf-8"))
        features = materialize_fit670_features(raw_episode)
        if len(features) != len(item["rows"]):
            raise ValueError(f"feature/teacher length mismatch: {identity}")
        targets = {head: [] for head in HEADS}
        masks = {head: [] for head in HEADS}
        candidates = []
        for feature_row in features:
            key = (identity, int(feature_row["step"]))
            teacher_row = labels.get(key)
            if teacher_row is None:
                raise ValueError(f"missing teacher row: {key}")
            candidates.append(bool(feature_row["candidate_close"]))
            if feature_row.get("action_gripper_source") != ACTION_GRIPPER_SOURCE or feature_row.get("feature_order") != list(FEATURE_ORDER):
                raise ValueError(f"feature source/order binding mismatch: {key}")
            for head in HEADS:
                label = teacher_row["labels"][head]
                known = _known_label(label, head)
                targets[head].append(float(label.get("value") == "TRUE"))
                masks[head].append(known)
        weights = {
            head: _event_weights(candidates, masks[head])
            for head in HEADS
        }
        records.append({
            "identity": identity,
            "features": np.asarray([row["features_25d"] for row in features], dtype=np.float32),
            "targets": {head: np.asarray(values, dtype=np.float32) for head, values in targets.items()},
            "masks": {head: np.asarray(values, dtype=bool) for head, values in masks.items()},
            "weights": weights,
        })
    expected = sum(len(item["features"]) for item in records)
    if len(labels) != expected:
        raise ValueError(f"teacher coverage mismatch: {len(labels)} != {expected}")
    binding = {
        "input_schema": manifest["schema"],
        "input_seal": input_seal["sha256sums_sha256"],
        "teacher_seal": teacher_seal["sha256sums_sha256"],
        "teacher_manifest_sha256": sha256_file(teacher_root / "teacher_manifest.json"),
        "transition_manifest_sha256": sha256_file(transition),
        "identities": [item["identity"] for item in records],
        "steps": expected,
        "feature_order": list(FEATURE_ORDER),
        "feature_order_sha256": feature_order_sha,
        "feature_source_sha256": feature_source_sha,
        "feature_source_hash_algorithm": feature_binding["adapter_source_hash_algorithm"],
        "feature_source_hash_normalization": feature_binding["adapter_source_hash_normalization"],
        "feature_binding_path": str(feature_binding_path),
        "feature_binding_sha256": feature_binding_sha,
        "action_gripper_source": ACTION_GRIPPER_SOURCE,
        "feature_binding": feature_binding,
        "protected_reads": int(
            bool(manifest.get("protected_reads"))
            or transition_data.get("protected_payload_read") is not False
            or transition_data.get("protected_overlap_verified") != 0
        ),
    }
    if binding["protected_reads"] != 0:
        raise ValueError("protected reads detected")
    return records, binding


def _batch(records: list[dict[str, Any]], device: torch.device):
    max_steps = max(len(item["features"]) for item in records)
    x = torch.zeros((len(records), max_steps, 25), dtype=torch.float32, device=device)
    valid = torch.zeros((len(records), max_steps), dtype=torch.bool, device=device)
    targets = {head: torch.zeros((len(records), max_steps), dtype=torch.float32, device=device) for head in HEADS}
    masks = {head: torch.zeros((len(records), max_steps), dtype=torch.bool, device=device) for head in HEADS}
    weights = {head: torch.zeros((len(records), max_steps), dtype=torch.float32, device=device) for head in HEADS}
    flat = np.concatenate([item["features"] for item in records], axis=0)
    mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32, device=device)
    std = torch.tensor(np.maximum(flat.std(axis=0), 1e-6), dtype=torch.float32, device=device)
    for index, item in enumerate(records):
        length = len(item["features"])
        x[index, :length] = (torch.tensor(item["features"], device=device) - mean) / std
        valid[index, :length] = True
        for head in HEADS:
            targets[head][index, :length] = torch.tensor(item["targets"][head], device=device)
            masks[head][index, :length] = torch.tensor(item["masks"][head], device=device)
            weights[head][index, :length] = torch.tensor(item["weights"][head], device=device)
    return x, valid, targets, masks, weights, mean, std


def _loss(logits: dict[str, torch.Tensor], targets, masks, weights) -> tuple[torch.Tensor, dict[str, float | None]]:
    terms = []
    diagnostics: dict[str, float | None] = {}
    for head in HEADS:
        model_head = MODEL_HEADS.get(head, head)
        known = masks[head]
        if not bool(known.any()):
            diagnostics[head] = None
            continue
        sample_weights = weights[head][known]
        sample_weights = sample_weights / sample_weights.sum().clamp_min(1e-12)
        value = F.binary_cross_entropy_with_logits(logits[model_head][known], targets[head][known], weight=sample_weights, reduction="sum")
        diagnostics[head] = float(value.detach().cpu())
        terms.append(value)
    if not terms:
        raise ValueError("all heads are disabled or UNKNOWN")
    return torch.stack(terms).mean(), diagnostics


def _accuracy(logits: dict[str, torch.Tensor], targets, masks) -> dict[str, float | None]:
    result = {}
    for head in HEADS:
        model_head = MODEL_HEADS.get(head, head)
        known = masks[head]
        if not bool(known.any()):
            result[head] = None
            continue
        pred = (logits[model_head][known] >= 0).to(targets[head].dtype)
        result[head] = float((pred == targets[head][known]).float().mean().detach().cpu())
    return result


def _train(model, x, valid, targets, masks, weights, optimizer, epochs: int):
    history = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, timestep_mask=valid)
        loss, components = _loss(logits, targets, masks, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite micro-overfit loss")
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
            raise FloatingPointError("non-finite micro-overfit gradient")
        optimizer.step()
        history.append({"loss": float(loss.detach().cpu()), "components": components})
    return history


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    records, binding = _load_data(args.input_root.resolve(), args.teacher_root.resolve(), args.transition.resolve())
    x, valid, targets, masks, weights, mean, std = _batch(records, device)
    N5MultiHeadStudent = _load_model()
    single_results = {}
    shared_results = None
    for requested_head in ACTIVE_HEADS:
        model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
        single_masks = {head: torch.zeros_like(masks[head]) for head in HEADS}
        single_masks[requested_head] = masks[requested_head]
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        with torch.no_grad():
            initial = float(_loss(model(x, timestep_mask=valid), targets, single_masks, weights)[0].cpu())
        history = _train(model, x, valid, targets, single_masks, weights, optimizer, args.epochs)
        with torch.no_grad():
            final_logits = model(x, timestep_mask=valid)
            final, components = _loss(final_logits, targets, single_masks, weights)
        single_results[requested_head] = {
            "initial_loss": initial,
            "final_loss": float(final.cpu()),
            "loss_reduction": 1.0 - float(final.cpu()) / max(initial, 1e-12),
            "accuracy": _accuracy(final_logits, targets, single_masks)[requested_head],
            "components": components,
            "epochs": args.epochs,
        }
    model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    with torch.no_grad():
        initial = float(_loss(model(x, timestep_mask=valid), targets, masks, weights)[0].cpu())
    history = _train(model, x, valid, targets, masks, weights, optimizer, args.epochs)
    with torch.no_grad():
        final_logits = model(x, timestep_mask=valid)
        final, components = _loss(final_logits, targets, masks, weights)
    inactive_gradient = {head: 0.0 for head in INACTIVE_HEADS}
    for parameter in model.parameters():
        parameter.grad = None
    optimizer.zero_grad(set_to_none=True)
    loss, _ = _loss(model(x, timestep_mask=valid), targets, masks, weights)
    loss.backward()
    for head in INACTIVE_HEADS:
        module = model.heads[N5MultiHeadStudent.HEAD_NAMES.index(MODEL_HEAD_ALIASES.get(head, head))]
        inactive_gradient[head] = float(sum(parameter.grad.abs().sum().cpu() for parameter in module.parameters() if parameter.grad is not None))
        if not np.isfinite(inactive_gradient[head]) or inactive_gradient[head] != 0.0:
            raise AssertionError(f"inactive head received gradient: {head}")
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "epoch": args.epochs,
    }
    resumed = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=args.learning_rate)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    with torch.no_grad():
        resume_diff = max(float((final_logits[key] - resumed(x, timestep_mask=valid)[key]).abs().max().cpu()) for key in final_logits)
    if not np.isfinite(resume_diff) or resume_diff > 1e-7:
        raise AssertionError(f"checkpoint resume mismatch: {resume_diff}")
    shuffled_targets = {head: value.clone() for head, value in targets.items()}
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    for head in ACTIVE_HEADS:
        known = masks[head]
        values = shuffled_targets[head][known].detach().cpu()
        if values.numel() > 1:
            shuffled_targets[head][known] = values[torch.randperm(values.numel(), generator=generator)].to(device)
    shuffle_model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    shuffle_optimizer = torch.optim.AdamW(shuffle_model.parameters(), lr=args.learning_rate)
    _train(shuffle_model, x, valid, shuffled_targets, masks, weights, shuffle_optimizer, args.epochs)
    with torch.no_grad():
        shuffle_accuracy = _accuracy(shuffle_model(x, timestep_mask=valid), targets, masks)
    result = {
        "schema": "V5_R3_3E_MICRO_OVERFIT_V1",
        "status": "ENGINEERING_MICRO_OVERFIT_NONCONSUMABLE",
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "active_heads": list(ACTIVE_HEADS),
        "inactive_heads": list(INACTIVE_HEADS),
        "steps": binding["steps"],
        "identity_count": len(records),
        "binding": binding,
        "single_head": single_results,
        "shared_three_head": {
            "initial_loss": initial,
            "final_loss": float(final.cpu()),
            "loss_reduction": 1.0 - float(final.cpu()) / max(initial, 1e-12),
            "accuracy": _accuracy(final_logits, targets, masks),
            "components": components,
        },
        "disabled_head_gradient_sum": inactive_gradient,
        "checkpoint_resume_max_logit_diff": resume_diff,
        "label_shuffle_accuracy_against_original": shuffle_accuracy,
        "shadow_action_mutation_count": 0,
        "protected_reads": binding["protected_reads"],
        "formal_inference_authorized": False,
        "formal_training_authorized": False,
        "attack_authorized": False,
    }
    staging = args.output_root.with_name(f".{args.output_root.name}.staging.{os.getpid()}")
    if staging.exists() or args.output_root.exists():
        raise FileExistsError(args.output_root)
    staging.mkdir(parents=True)
    try:
        (staging / "micro_overfit_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "feature_normalization.json").write_text(json.dumps({"mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        torch.save(checkpoint, staging / "checkpoint.pt")
        digest = _write_seal(staging)
        rename_noreplace(staging, args.output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result["sha256sums_sha256"] = digest
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
