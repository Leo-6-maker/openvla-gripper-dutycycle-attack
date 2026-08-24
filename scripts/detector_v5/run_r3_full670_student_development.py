"""Full FIT670 development-only Student learnability run.

Consumes only the sealed T4 transition and its already sealed source/Teacher
roots.  It trains the four eligible heads jointly and leaves safe_release
masked out.  This is an engineering learnability result, not held-out
evaluation or a formal model-selection artifact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
N5_ROOT = ROOT / "n5" / "phase3_student"
for path in (SRC, N5_ROOT, ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_r3_contact_input import sha256_file, verify_seal
from build_r3_teacher_student_transition import (
    _assert_role_roots_disjoint,
    _feature_binding,
    _git_snapshot,
    _require_clean_git,
    _safe_file,
    _safe_output,
    _safe_root,
    _sealed_json,
    _source_bindings,
    _validate_nested_input_binding,
    _validate_t0b_permissions,
    _validate_t3,
    _validate_teacher_manifest,
)
from gripper_attack.seal_utils import rename_noreplace
from gripper_attack.v5_r3_features import ACTION_GRIPPER_SOURCE, FEATURE_ORDER, load_feature_binding, materialize_fit670_features
from gripper_attack.v5_r3_teacher import HEADS
from run_r3_micro_overfit import _accuracy, _batch, _event_weights, _load_model, _loss, _physical_event_weights, _train, _write_seal


ACTIVE_HEADS = ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
INACTIVE_HEADS = ("safe_release",)
TRUTH_VALUES = {"TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"}
_TEACHER_IDENTITY_RE = re.compile(r'"episode_id"\s*:\s*("(?:\\.|[^"\\])*")')
_TEACHER_STEP_RE = re.compile(r'"step"\s*:\s*(-?\d+)')


def _validate_t4_permissions(permissions: Any) -> None:
    expected = {
        "teacher_label_read": True,
        "student_dataset_generation": True,
        "student_training": True,
        "student_training_scope": "DEVELOPMENT_ONLY",
        "development_student_training_authorized": True,
        "development_inference": True,
        "development_inference_authorized": True,
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
        "shadow_authorized": False,
        "rollout_authorized": False,
        "protected_reads": 0,
        "CAL_READ": False,
        "CHECK_READ": False,
        "G10_READ": False,
        "T2R_D_READ": False,
        "attack_authorized": False,
    }
    if permissions != expected:
        raise ValueError("T4 nested permission matrix is not exact development-only scope")


def _known_label(label: Any, *, active: bool) -> bool:
    if not isinstance(label, Mapping) or label.get("value") not in TRUTH_VALUES:
        raise ValueError("invalid tri-state Teacher label")
    for field in ("valid_mask", "mask", "right_censored"):
        if type(label.get(field)) is not bool:
            raise ValueError(f"invalid Teacher label field: {field}")
    return bool(active and label["valid_mask"] and label["mask"] and not label["right_censored"] and label["value"] in {"TRUE", "FALSE"})


def _safe_episode(formal_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("episodes",) or relative.name != "episode.json":
        raise ValueError(f"unsafe episode path: {relative_path}")
    path = (formal_root / relative).resolve()
    if formal_root.resolve() not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(f"episode path escapes source root: {relative_path}")
    current = path
    while current != formal_root.resolve():
        if current.is_symlink():
            raise ValueError(f"symlinked episode component: {current}")
        current = current.parent
    return path


def _snapshot_matches(snapshot: Mapping[str, Any], *, allow_descendant_snapshot: bool) -> bool:
    actual_commit, actual_tree = _git_snapshot(ROOT)
    if (actual_commit, actual_tree) == (snapshot.get("commit"), snapshot.get("tree")):
        return True
    if not allow_descendant_snapshot:
        return False
    source_commit = snapshot.get("commit")
    source_tree = snapshot.get("tree")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or not isinstance(source_tree, str) or len(source_tree) != 40:
        return False
    try:
        resolved_tree = subprocess.check_output(("git", "rev-parse", f"{source_commit}^{{tree}}"), cwd=ROOT, text=True).strip()
        is_ancestor = subprocess.run(("git", "merge-base", "--is-ancestor", source_commit, actual_commit), cwd=ROOT, check=False).returncode == 0
    except (OSError, subprocess.CalledProcessError):
        return False
    return resolved_tree == source_tree and is_ancestor


def _load_t4(t4_root: Path, *, allow_descendant_snapshot: bool = False, skip_source_binding: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    t4_root = _safe_root(t4_root)
    t4_seal = verify_seal(t4_root)
    transition_path = _safe_file(t4_root / "TEACHER_STUDENT_TRANSITION.json", label="T4 transition")
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    if transition.get("schema") != "V5_R3_TEACHER_STUDENT_TRANSITION_V1" or transition.get("status") != "PASS_DEVELOPMENT_ELIGIBLE_HEADS":
        raise ValueError("T4 transition is not development-consumable")
    if transition.get("eligible_heads") != list(ACTIVE_HEADS) or transition.get("held_heads", {}).keys() != {"safe_release"}:
        raise ValueError("T4 head eligibility is not the frozen four-head scope")
    _validate_t4_permissions(transition.get("permissions"))
    if transition.get("safe_release_training_authorized") is not False or transition.get("formal_training_authorized") is not False or transition.get("attack_authorized") is not False or transition.get("protected_reads") != 0 or transition.get("labels_copied") is not False:
        raise ValueError("T4 permission boundary is not closed")
    permissions = transition.get("permissions")
    if not isinstance(permissions, Mapping) or permissions.get("student_training_scope") != "DEVELOPMENT_ONLY" or permissions.get("development_student_training_authorized") is not True or permissions.get("formal_training_authorized") is not False or permissions.get("formal_inference_authorized") is not False or permissions.get("attack_authorized") is not False:
        raise ValueError("T4 development permission scope is invalid")

    snapshot = transition.get("code_snapshot")
    if not isinstance(snapshot, Mapping) or not _snapshot_matches(snapshot, allow_descendant_snapshot=allow_descendant_snapshot):
        raise ValueError("T4 code snapshot does not match the consuming checkout")
    protocol_path = _safe_file(ROOT / "configs" / "R3_DEV_PROTOCOL.json", label="R3 protocol")
    if sha256_file(protocol_path) != transition.get("protocol", {}).get("sha256"):
        raise ValueError("T4 protocol binding mismatch")
    feature_binding = _feature_binding(ROOT)
    if not skip_source_binding and (transition.get("feature_binding") != feature_binding or transition.get("source_bindings") != _source_bindings(ROOT)):
        raise ValueError("T4 source/feature binding mismatch")

    teacher_root = _safe_root(Path(str(transition["teacher_root"])))
    coverage_root = _safe_root(Path(str(transition["coverage_root"])))
    t0a_root = _safe_root(Path(str(transition["t0_a"]["root"])))
    t0b_path = _safe_file(Path(str(transition["t0_b"]["manifest"])), label="T0-B transition")
    t0b_root = _safe_root(t0b_path.parent)
    teacher_manifest_path = _safe_file(teacher_root / "teacher_manifest.json", label="Teacher manifest")
    teacher_seal = verify_seal(teacher_root)
    coverage_seal = verify_seal(coverage_root)
    t0a_manifest, t0a_seal = _sealed_json(t0a_root, "FORMAL_INPUT_MANIFEST.json")
    t0b_manifest, t0b_seal = _sealed_json(t0b_root, t0b_path.name)
    formal_root = _safe_root(Path(str(t0a_manifest["formal_root"])))
    _assert_role_roots_disjoint({"t4": t4_root, "teacher": teacher_root, "coverage": coverage_root, "t0a": t0a_root, "t0b": t0b_root, "formal": formal_root})
    if t0a_seal["sha256sums_sha256"] != transition["t0_a"].get("seal_sha256sums_sha256") or sha256_file(t0a_root / "FORMAL_INPUT_MANIFEST.json") != transition["t0_a"].get("manifest_sha256"):
        raise ValueError("T4 T0-A binding mismatch")
    if t0b_seal["sha256sums_sha256"] != transition["t0_b"].get("seal_sha256sums_sha256") or sha256_file(t0b_path) != transition["t0_b"].get("manifest_sha256"):
        raise ValueError("T4 T0-B binding mismatch")
    _validate_teacher_manifest(teacher_manifest := json.loads(teacher_manifest_path.read_text(encoding="utf-8")), root_seal=teacher_seal["sha256sums_sha256"])
    _validate_t3(json.loads((coverage_root / "coverage_report.json").read_text(encoding="utf-8")), teacher_root=teacher_root, teacher_seal=teacher_seal["sha256sums_sha256"])
    if t0a_manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or t0a_manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or t0a_manifest.get("episode_count") != 670 or t0a_manifest.get("protected_reads") != 0 or t0a_manifest.get("teacher_labels_generated") is not False:
        raise ValueError("T0-A boundary/cardinality mismatch")
    if t0b_manifest.get("schema") != "FIT_TO_TEACHER_TRANSITION_V1" or t0b_manifest.get("status") != "PASS_FIT_TO_TEACHER_AUTHORIZATION" or t0b_manifest.get("identity_count") != 670 or t0b_manifest.get("protected_reads") != 0 or t0b_manifest.get("student_training_authorized") is not False or t0b_manifest.get("attack_authorized") is not False:
        raise ValueError("T0-B boundary/cardinality mismatch")
    _validate_t0b_permissions(t0b_manifest.get("permissions"))
    _validate_nested_input_binding(
        teacher_manifest,
        t0a_manifest=t0a_manifest,
        t0a_manifest_sha=sha256_file(t0a_root / "FORMAL_INPUT_MANIFEST.json"),
        t0a_seal=t0a_seal,
        t0a_root=t0a_root,
        t0b_manifest=t0b_manifest,
        t0b_manifest_sha=sha256_file(t0b_path),
        t0b_seal=t0b_seal,
        t0b_path=t0b_path,
    )
    if t0a_manifest.get("formal_root") != t0b_manifest.get("formal_root") or t0a_manifest.get("formal_root") != teacher_manifest.get("input_binding", {}).get("formal_root"):
        raise ValueError("formal source root binding mismatch")
    if teacher_seal["sha256sums_sha256"] != transition.get("teacher_root_sha256sums_sha256") or coverage_seal["sha256sums_sha256"] != transition.get("coverage_root_sha256sums_sha256"):
        raise ValueError("T4 input root seal mismatch")
    records_path = teacher_root / "teacher_records.jsonl"
    if sha256_file(teacher_manifest_path) != transition.get("teacher_manifest_sha256") or sha256_file(records_path) != transition.get("teacher_records_sha256"):
        raise ValueError("T4 Teacher file binding mismatch")
    coverage = json.loads((coverage_root / "coverage_report.json").read_text(encoding="utf-8"))
    if teacher_manifest.get("identity_count") != 670 or teacher_manifest.get("step_count") != 196483 or teacher_manifest.get("protected_reads") != 0 or teacher_manifest.get("unknown_to_negative") is not False:
        raise ValueError("Teacher root cardinality/permission mismatch")
    if coverage.get("status") != "HOLD_COVERAGE" or coverage.get("protected_reads") != 0 or coverage.get("unknown_as_negative") is not False:
        raise ValueError("T3 coverage boundary mismatch")
    return transition, teacher_manifest, coverage, {"teacher_root": teacher_root, "coverage_root": coverage_root, "formal_root": formal_root, "t0a_root": t0a_root, "t0b_path": t0b_path, "t0a_manifest": t0a_manifest, "t0b_manifest": t0b_manifest, "t4_seal": t4_seal}


def _load_records(t4_root: Path, *, allow_descendant_snapshot: bool = False, identity_allowlist: set[str] | None = None, skip_source_binding: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transition, teacher_manifest, coverage, roots = _load_t4(t4_root, allow_descendant_snapshot=allow_descendant_snapshot, skip_source_binding=skip_source_binding)
    audit_root = Path(str(transition["t0_a"]["root"])).resolve()
    audit_seal = verify_seal(audit_root)
    audit_path = audit_root / "FORMAL_INPUT_MANIFEST.json"
    if audit_seal["sha256sums_sha256"] != transition["t0_a"]["seal_sha256sums_sha256"] or sha256_file(audit_path) != transition["t0_a"]["manifest_sha256"]:
        raise ValueError("T0-A binding mismatch")
    audited = json.loads(audit_path.read_text(encoding="utf-8"))
    if audited.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or audited.get("episode_count") != 670 or audited.get("protected_reads") != 0 or audited.get("teacher_labels_generated") is not False:
        raise ValueError("T0-A source is not consumable")
    formal_root = _safe_root(Path(str(audited["formal_root"])))
    bindings = audited.get("episode_bindings")
    if not isinstance(bindings, Mapping) or len(bindings) != 670:
        raise ValueError("T0-A identity closure is incomplete")
    all_identities = {str(identity) for identity in bindings}
    selected_identities = sorted(all_identities if identity_allowlist is None else {str(identity) for identity in identity_allowlist})
    if not selected_identities or not set(selected_identities).issubset(all_identities):
        raise ValueError("identity allowlist is empty or outside T0-A")

    labels: dict[tuple[str, int], dict[str, Any]] = {}
    seen_teacher_keys: set[tuple[str, int]] = set()
    total_teacher_rows = 0
    with (roots["teacher_root"] / "teacher_records.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_teacher_rows += 1
            identity_matches = _TEACHER_IDENTITY_RE.findall(line)
            step_matches = _TEACHER_STEP_RE.findall(line)
            if len(identity_matches) != 1 or len(step_matches) != 1:
                raise ValueError(f"malformed Teacher identity index at line {line_number}")
            identity, step = str(json.loads(identity_matches[0])), int(step_matches[0])
            if not isinstance(step, int) or isinstance(step, bool) or step < 0 or (identity, step) in seen_teacher_keys:
                raise ValueError(f"duplicate/malformed Teacher record at line {line_number}")
            seen_teacher_keys.add((identity, step))
            if identity not in all_identities:
                raise ValueError(f"Teacher record outside T0-A identity closure: {identity}")
            if identity in selected_identities:
                row = json.loads(line)
                row_labels = row.get("labels")
                if str(row.get("episode_id")) != identity or row.get("step") != step or not isinstance(row_labels, Mapping) or set(row_labels) != set(HEADS):
                    raise ValueError(f"Teacher head closure failed at line {line_number}")
                labels[(identity, step)] = row
    if total_teacher_rows != 196483:
        raise ValueError(f"Teacher record count mismatch: {total_teacher_rows}")

    binding_rows: list[dict[str, Any]] = []
    for identity in selected_identities:
        binding = bindings[identity]
        if not isinstance(binding, Mapping) or binding.get("episode_id") != identity:
            raise ValueError(f"identity binding mismatch: {identity}")
        episode_path = _safe_episode(formal_root, str(binding.get("relative_path")))
        if sha256_file(episode_path) != binding.get("episode_sha256"):
            raise ValueError(f"episode file SHA mismatch: {identity}")
        episode_seal = verify_seal(episode_path.parent)
        if episode_seal["sha256sums_sha256"] != binding.get("episode_sha256sums_sha256"):
            raise ValueError(f"episode seal mismatch: {identity}")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "episode_id": identity,
            "suite": binding.get("suite"),
            "task_id": binding.get("task_id"),
            "state_id": binding.get("state_id"),
            "initial_state_sha256": binding.get("initial_state_sha256"),
        }
        if any(episode.get(field) != value for field, value in expected_metadata.items()):
            raise ValueError(f"episode metadata binding mismatch: {identity}")
        if episode.get("collection_seed") != binding.get("seed") or episode.get("step_count") != binding.get("worker_result_steps") or episode.get("n_steps") != binding.get("worker_result_steps"):
            raise ValueError(f"episode step/seed binding mismatch: {identity}")
        provenance = episode.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("collection_seed") != binding.get("seed") or provenance.get("collector_commit") != binding.get("collection_source_commit") or provenance.get("collector_tree") != binding.get("collection_source_tree") or provenance.get("collector_script_sha256") != binding.get("collector_script_sha256"):
            raise ValueError(f"episode provenance binding mismatch: {identity}")
        episode_binding = episode.get("episode_bindings")
        expected_episode_binding = {
            "identity_set_digest": audited.get("identity_set_digest"),
            "collection_source_commit": binding.get("collection_source_commit"),
            "collection_source_tree": binding.get("collection_source_tree"),
            "schema_version": "FIT670_FEATURE_SCHEMA_V1",
            "transition_schema": "FIT670_INFERENCE_TRANSITION_V2",
            "transition_receipt_sha256": roots["t0b_manifest"].get("parent_transition_sha256sums_sha256"),
        }
        if not isinstance(episode_binding, Mapping) or any(episode_binding.get(field) != value for field, value in expected_episode_binding.items()):
            raise ValueError(f"episode nested binding mismatch: {identity}")
        features = materialize_fit670_features(episode)
        expected_steps = binding.get("worker_result_steps")
        if not isinstance(expected_steps, int) or isinstance(expected_steps, bool) or len(features) != expected_steps:
            raise ValueError(f"episode step mismatch: {identity}")
        targets = {head: [] for head in HEADS}
        masks = {head: [] for head in HEADS}
        candidates: list[bool] = []
        right_censored = {head: [] for head in HEADS}
        for feature_row in features:
            step = int(feature_row["step"])
            row = labels.get((identity, step))
            if row is None:
                raise ValueError(f"missing Teacher row: {identity}/{step}")
            if feature_row.get("action_gripper_source") != ACTION_GRIPPER_SOURCE or feature_row.get("feature_order") != list(FEATURE_ORDER):
                raise ValueError(f"feature binding mismatch: {identity}/{step}")
            candidates.append(bool(feature_row["candidate_close"]))
            for head in HEADS:
                label = row["labels"][head]
                known = _known_label(label, active=head in ACTIVE_HEADS)
                targets[head].append(float(label["value"] == "TRUE"))
                masks[head].append(known)
                right_censored[head].append(bool(label["right_censored"]))
        binding_rows.append({
            "identity": identity,
            "features": np.asarray([row["features_25d"] for row in features], dtype=np.float32),
            "candidate_close": np.asarray(candidates, dtype=bool),
            "targets": {head: np.asarray(values, dtype=np.float32) for head, values in targets.items()},
            "masks": {head: np.asarray(values, dtype=bool) for head, values in masks.items()},
            "right_censored": {head: np.asarray(values, dtype=bool) for head, values in right_censored.items()},
            "weights": {
                "physical_criticality": _physical_event_weights(targets["physical_criticality"], masks["physical_criticality"]),
                **{head: _event_weights(candidates, masks[head]) for head in HEADS if head != "physical_criticality"},
            },
        })
    expected_step_count = 196483 if identity_allowlist is None else sum(int(bindings[identity]["worker_result_steps"]) for identity in selected_identities)
    if sum(len(row["features"]) for row in binding_rows) != expected_step_count:
        raise ValueError("feature/Teacher total step mismatch")
    feature_binding_path = ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json"
    feature_binding = load_feature_binding(feature_binding_path, ROOT)
    return binding_rows, {
        "t4_root": str(t4_root.resolve()),
        "t4_seal_sha256sums_sha256": roots["t4_seal"]["sha256sums_sha256"],
        "teacher_root": str(roots["teacher_root"]),
        "teacher_root_sha256sums_sha256": transition["teacher_root_sha256sums_sha256"],
        "coverage_root": str(roots["coverage_root"]),
        "coverage_root_sha256sums_sha256": transition["coverage_root_sha256sums_sha256"],
        "teacher_manifest_sha256": transition["teacher_manifest_sha256"],
        "teacher_records_sha256": transition["teacher_records_sha256"],
        "t0a_manifest": audited,
        "t0b_manifest": roots["t0b_manifest"],
        "identity_count": len(binding_rows),
        "step_count": sum(len(row["features"]) for row in binding_rows),
        "eligible_heads": list(ACTIVE_HEADS),
        "held_heads": list(INACTIVE_HEADS),
        "feature_binding_sha256": sha256_file(feature_binding_path),
        "feature_order_sha256": hashlib.sha256(json.dumps(list(FEATURE_ORDER), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest(),
        "feature_source_sha256": feature_binding["adapter_source_sha256"],
        "execution_source_sha256": sha256_file(ROOT / "scripts/detector_v5/run_r3_full670_student_development.py"),
        "protected_reads": 0,
    }


def _shuffle_known_targets(targets: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Shuffle each active head only within its known-label population."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffled = {head: values.clone() for head, values in targets.items()}
    for head in ACTIVE_HEADS:
        flat_mask = masks[head].reshape(-1)
        indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
        if indices.numel() < 2:
            continue
        flat = shuffled[head].reshape(-1)
        values = flat[indices].clone()
        permutation = torch.randperm(indices.numel(), generator=generator)
        if torch.equal(permutation, torch.arange(indices.numel())):
            permutation = torch.roll(permutation, 1)
        flat[indices] = values[permutation]
    return shuffled


def _restore_optimizer(model: torch.nn.Module, checkpoint_state: Mapping[str, Any], *, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Checkpoint state contains mutable AdamW tensors; each restored branch must own a copy.
    optimizer.load_state_dict(copy.deepcopy(checkpoint_state))
    return optimizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output_root.is_absolute():
        raise ValueError("output root must be absolute")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    _require_clean_git(ROOT)
    records, binding = _load_records(args.transition_root.resolve())
    output_root = _safe_output(args.output_root, parent=Path(binding["teacher_root"]).parent)
    x, valid, targets, masks, weights, mean, std = _batch(records, device)
    N5MultiHeadStudent = _load_model()
    single_results: dict[str, Any] = {}
    for requested_head in ACTIVE_HEADS:
        model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
        single_masks = {head: torch.zeros_like(masks[head]) for head in HEADS}
        single_masks[requested_head] = masks[requested_head]
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        with torch.no_grad():
            initial = float(_loss(model(x, timestep_mask=valid), targets, single_masks, weights)[0])
        history = _train(model, x, valid, targets, single_masks, weights, optimizer, args.epochs)
        with torch.no_grad():
            final_logits = model(x, timestep_mask=valid)
            final, components = _loss(final_logits, targets, single_masks, weights)
        single_results[requested_head] = {
            "initial_loss": initial,
            "final_loss": float(final),
            "loss_reduction": 1.0 - float(final) / max(initial, 1e-12),
            "accuracy": _accuracy(final_logits, targets, single_masks)[requested_head],
            "history": history,
            "components": components,
        }

    init_rng_state = torch.get_rng_state()
    init_np_state = np.random.get_state()
    model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    frozen_initial_state = copy.deepcopy(model.state_dict())
    init_state_digest = hashlib.sha256(b"".join(param.detach().cpu().numpy().tobytes() for param in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    active_masks = {head: (masks[head] if head in ACTIVE_HEADS else torch.zeros_like(masks[head])) for head in HEADS}
    with torch.no_grad():
        initial = float(_loss(model(x, timestep_mask=valid), targets, active_masks, weights)[0])
    history = _train(model, x, valid, targets, active_masks, weights, optimizer, args.epochs)
    with torch.no_grad():
        final_logits = model(x, timestep_mask=valid)
        final, components = _loss(final_logits, targets, active_masks, weights)
    if not torch.isfinite(final):
        raise FloatingPointError("nonfinite shared Student loss")

    model.zero_grad(set_to_none=True)
    loss, _ = _loss(model(x, timestep_mask=valid), targets, active_masks, weights)
    loss.backward()
    inactive_gradient = {}
    for head in INACTIVE_HEADS:
        module = model.heads[N5MultiHeadStudent.HEAD_NAMES.index(head)]
        inactive_gradient[head] = float(sum(parameter.grad.abs().sum() for parameter in module.parameters() if parameter.grad is not None))
        if inactive_gradient[head] != 0.0:
            raise AssertionError(f"disabled head received gradient: {head}")

    checkpoint = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "epoch": args.epochs, "active_heads": list(ACTIVE_HEADS), "torch_rng_state": torch.get_rng_state().clone()}
    resumed = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    resumed_optimizer = _restore_optimizer(resumed, checkpoint["optimizer"], learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    resumed.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        resume_diff = max(float((final_logits[key] - resumed(x, timestep_mask=valid)[key]).abs().max()) for key in final_logits)
    if not np.isfinite(resume_diff) or resume_diff > 1e-7:
        raise AssertionError(f"checkpoint resume mismatch: {resume_diff}")

    continued = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    continued_optimizer = _restore_optimizer(continued, checkpoint["optimizer"], learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    continued.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.zero_grad(set_to_none=True)
    training_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.set_rng_state(checkpoint["torch_rng_state"])
        _train(continued, x, valid, targets, active_masks, weights, continued_optimizer, 1)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        _train(resumed, x, valid, targets, active_masks, weights, resumed_optimizer, 1)
        with torch.no_grad():
            continuation_diff = max(float((continued(x, timestep_mask=valid)[key] - resumed(x, timestep_mask=valid)[key]).abs().max()) for key in final_logits)
    finally:
        torch.set_num_threads(training_threads)
    if not np.isfinite(continuation_diff) or continuation_diff > 1e-7:
        raise AssertionError(f"checkpoint continuation mismatch: {continuation_diff}")

    shuffled_targets = _shuffle_known_targets(targets, active_masks, args.seed + 1)
    shuffle_model = N5MultiHeadStudent(input_dim=25, dropout=0.0).to(device)
    shuffle_model.load_state_dict(frozen_initial_state, strict=True)
    shuffle_init_digest = hashlib.sha256(b"".join(param.detach().cpu().numpy().tobytes() for param in shuffle_model.parameters())).hexdigest()
    if shuffle_init_digest != init_state_digest:
        raise AssertionError("label-shuffle model does not share real model initialization")
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)
    shuffle_optimizer = torch.optim.AdamW(shuffle_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    with torch.no_grad():
        shuffle_initial = float(_loss(shuffle_model(x, timestep_mask=valid), shuffled_targets, active_masks, weights)[0])
    shuffle_history = _train(shuffle_model, x, valid, shuffled_targets, active_masks, weights, shuffle_optimizer, args.epochs)
    with torch.no_grad():
        shuffle_logits = shuffle_model(x, timestep_mask=valid)
        shuffle_final, shuffle_components = _loss(shuffle_logits, shuffled_targets, active_masks, weights)
    if not torch.isfinite(shuffle_final):
        raise FloatingPointError("nonfinite label-shuffle Student loss")

    base_rate = {}
    for head in ACTIVE_HEADS:
        known = active_masks[head]
        rate = float(targets[head][known].mean())
        prediction = 1.0 if rate >= 0.5 else 0.0
        base_rate[head] = {"positive_rate": rate, "majority_accuracy": float((targets[head][known] == prediction).float().mean())}

    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        torch.save(checkpoint, staging / "checkpoint.pt")
        checkpoint_sha = sha256_file(staging / "checkpoint.pt")
        report = {
            "schema": "V5_R3_FULL670_STUDENT_LEARNABILITY_V1",
            "status": "ENGINEERING_FULL670_LEARNABILITY_NONCONSUMABLE",
            "device": str(device),
            "threads": args.threads,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "identity_count": binding["identity_count"],
            "step_count": binding["step_count"],
            "active_heads": list(ACTIVE_HEADS),
            "inactive_heads": list(INACTIVE_HEADS),
            "binding": binding,
            "single_head": single_results,
            "shared_four_head": {"initial_loss": initial, "final_loss": float(final), "loss_reduction": 1.0 - float(final) / max(initial, 1e-12), "accuracy": _accuracy(final_logits, targets, active_masks), "components": components, "history": history},
            "initial_state_sha256": init_state_digest,
            "label_shuffle_same_initialization": True,
            "label_shuffle_control": {"seed": args.seed + 1, "initial_state_sha256": shuffle_init_digest, "initial_state_matches_real": shuffle_init_digest == init_state_digest, "initial_loss": shuffle_initial, "final_loss": float(shuffle_final), "loss_reduction": 1.0 - float(shuffle_final) / max(shuffle_initial, 1e-12), "accuracy": _accuracy(shuffle_logits, shuffled_targets, active_masks), "components": shuffle_components, "history": shuffle_history},
            "base_rate": base_rate,
            "disabled_head_gradient_sum": inactive_gradient,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_resume_max_logit_diff": resume_diff,
            "checkpoint_continuation_max_logit_diff": continuation_diff,
            "checkpoint_continuation_threads": 1,
            "checkpoint_rng_state_saved": True,
            "feature_normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "heldout_evaluation": False,
            "threshold_selected": False,
            "model_selection": False,
            "protected_reads": 0,
            "formal_training_authorized": False,
            "formal_inference_authorized": False,
            "shadow_authorized": False,
            "attack_authorized": False,
        }
        (staging / "learnability_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "feature_normalization.json").write_text(json.dumps(report["feature_normalization"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        digest = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
