"""Preparation-only contracts for corrected V2.1C Student training.

This module intentionally does not import torch, allocate a model, train, or write
checkpoints.  It validates immutable V2.1C Teacher inputs, builds the exact
four-fold OOF plan, and provides explicit target adapters for a later trainer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .b3_training_protocol import load_fit_fold_bundle, sha256_file, verify_sealed_directory

V21C_PROTOCOL_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL"
V21C_MANIFEST_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_V21C_MANIFEST"
V21C_AUDIT_SCHEMA = "DETECTOR_V5_PHYSICS_TEACHER_V21C_AUDIT_V1"
GEOMETRY_AUDIT_SCHEMA = "DETECTOR_V5_V21C_GEOMETRY_AUDIT_V1"
PREP_PROTOCOL_SCHEMA = "DETECTOR_V5_V21C_OOF_TRAINING_PREP_PROTOCOL_V1"
PREP_PLAN_SCHEMA = "DETECTOR_V5_V21C_OOF_TRAINING_PLAN_V1"
ALLOWED_TEACHER_AUDIT_STATUSES = {"PASS", "PASS_WITH_EXPLICIT_NON_GRASP_TASKS"}
ALLOWED_GEOMETRY_STATUSES = {"PASS_CONTROL_TRAINING_ONLY"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_sha(value: Any, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class V21CTeacherBinding:
    root: str
    root_sha256s_sha256: str
    manifest_sha256: str
    protocol_sha256: str
    audit_sha256: str
    action_contract_sha256: str
    v5_physics_sha256: str
    source_git_commit: str
    identity_count: int
    step_count: int
    task_count: int
    known_step_count: int
    label_file_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryGateBinding:
    root: str
    root_sha256s_sha256: str
    audit_sha256: str
    status: str
    teacher_root_sha256s_sha256: str
    control_training_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_v21c_teacher_root(
    root: Path,
    *,
    expected_source_commit: str | None = None,
) -> V21CTeacherBinding:
    """Validate one sealed 800-identity V2.1C Teacher root fail-closed."""

    root = root.resolve()
    verify_sealed_directory(root)
    manifest_path = root / "physics_teacher_v21c_manifest.json"
    protocol_path = root / "protocol.json"
    audit_path = root / "audit_report.json"
    action_contract_path = root / "action_contract.json"
    for path in (manifest_path, protocol_path, audit_path, action_contract_path):
        _require(path.is_file(), f"missing V2.1C binding file: {path}")

    manifest = _read_json(manifest_path)
    protocol = _read_json(protocol_path)
    audit = _read_json(audit_path)
    action_contract = _read_json(action_contract_path)

    _require(manifest.get("schema") == V21C_MANIFEST_SCHEMA, "wrong V2.1C manifest schema")
    _require(manifest.get("teacher_version") == "V2.1C", "wrong V2.1C teacher version")
    _require(protocol.get("schema") == V21C_PROTOCOL_SCHEMA, "wrong V2.1C protocol schema")
    _require(manifest.get("protocol_schema") == V21C_PROTOCOL_SCHEMA, "manifest/protocol schema mismatch")
    _require(audit.get("schema") == V21C_AUDIT_SCHEMA, "wrong V2.1C audit schema")
    _require(audit.get("status") in ALLOWED_TEACHER_AUDIT_STATUSES, "V2.1C audit is not passing")
    _require(manifest.get("formal_training_authorized") is False, "Teacher root must not authorize training")
    _require(manifest.get("formal_attack_authorized") is False, "Teacher root must not authorize attack")
    _require(audit.get("formal_training_authorized") is False, "Teacher audit must not authorize training")
    _require(audit.get("formal_attack_authorized") is False, "Teacher audit must not authorize attack")

    _require(int(manifest.get("identity_count", -1)) == 800, "V2.1C identity count must be 800")
    _require(int(manifest.get("step_count", -1)) == 176336, "V2.1C step count must be 176336")
    _require(int(manifest.get("task_count", -1)) == 40, "V2.1C task count must be 40")
    known_steps = int(manifest.get("known_step_count", -1))
    _require(0 <= known_steps <= 176336, "invalid V2.1C known-step count")

    source_commit = manifest.get("source_git_commit")
    _require(_is_sha(source_commit, 40), "V2.1C source commit must be exact 40-char lowercase SHA")
    if expected_source_commit is not None:
        _require(_is_sha(expected_source_commit, 40), "expected source commit must be 40-char lowercase SHA")
        _require(source_commit == expected_source_commit, "V2.1C source commit mismatch")

    for name in ("action_contract_sha256", "v5_physics_sha256", "protocol_sha256"):
        _require(_is_sha(manifest.get(name), 64), f"invalid manifest binding: {name}")
    _require(action_contract.get("schema") == manifest.get("action_contract_schema"), "action contract schema mismatch")
    _require(action_contract.get("file_sha256") == manifest.get("action_contract_sha256"), "action contract SHA mismatch")
    _require(sha256_file(protocol_path) == manifest.get("protocol_sha256"), "protocol file SHA mismatch")

    embedded = audit.get("manifest")
    _require(isinstance(embedded, dict), "V2.1C audit lacks embedded manifest")
    for key in ("schema", "source_git_commit", "identity_count", "step_count", "known_step_count"):
        _require(embedded.get(key) == manifest.get(key), f"audit/manifest mismatch: {key}")

    labels = list(root.glob("labels/*/task_*/state_*/physics_teacher_v21c.jsonl"))
    _require(len(labels) == 800, f"expected 800 V2.1C label files, got {len(labels)}")

    return V21CTeacherBinding(
        root=str(root),
        root_sha256s_sha256=sha256_file(root / "SHA256SUMS"),
        manifest_sha256=sha256_file(manifest_path),
        protocol_sha256=sha256_file(protocol_path),
        audit_sha256=sha256_file(audit_path),
        action_contract_sha256=str(manifest["action_contract_sha256"]),
        v5_physics_sha256=str(manifest["v5_physics_sha256"]),
        source_git_commit=str(source_commit),
        identity_count=800,
        step_count=176336,
        task_count=40,
        known_step_count=known_steps,
        label_file_count=len(labels),
    )


def validate_geometry_gate(root: Path, *, teacher: V21CTeacherBinding) -> GeometryGateBinding:
    """Validate the future D2.1.5 geometry gate required before any training."""

    root = root.resolve()
    verify_sealed_directory(root)
    audit_path = root / "geometry_audit.json"
    _require(audit_path.is_file(), "geometry gate root lacks geometry_audit.json")
    audit = _read_json(audit_path)
    _require(audit.get("schema") == GEOMETRY_AUDIT_SCHEMA, "wrong geometry audit schema")
    _require(audit.get("status") in ALLOWED_GEOMETRY_STATUSES, "geometry audit does not authorize control training")
    _require(audit.get("teacher_root_sha256s_sha256") == teacher.root_sha256s_sha256, "geometry audit is not bound to the V2.1C root")
    _require(audit.get("control_training_authorized") is True, "geometry audit lacks control-training authorization")
    _require(audit.get("formal_attack_authorized") is False, "geometry audit must not authorize attack")
    return GeometryGateBinding(
        root=str(root),
        root_sha256s_sha256=sha256_file(root / "SHA256SUMS"),
        audit_sha256=sha256_file(audit_path),
        status=str(audit["status"]),
        teacher_root_sha256s_sha256=teacher.root_sha256s_sha256,
        control_training_authorized=True,
    )


def extract_v21c_control_targets(
    row: Mapping[str, Any],
    *,
    release_threshold: float,
    regrasp_threshold: float,
) -> dict[str, Any]:
    """Map one V2.1C row to explicit control targets without silent coercion."""

    if release_threshold < 0 or regrasp_threshold < 0:
        raise ValueError("risk thresholds must be non-negative")
    known = row.get("known_mask")
    action_known = row.get("action_known")
    candidate_close = row.get("candidate_close")
    intent = row.get("action_intent")
    tier = row.get("utility_tier")
    confidence = row.get("teacher_confidence")
    if not isinstance(known, bool) or not isinstance(action_known, bool) or not isinstance(candidate_close, bool):
        raise TypeError("V2.1C boolean contract fields must be bool")
    if intent not in {"CLOSE", "OPEN", "BOUNDARY", "UNKNOWN"}:
        raise ValueError("invalid V2.1C action intent")
    if known != (tier is not None):
        raise ValueError("known-mask/tier mismatch")
    if not action_known:
        if candidate_close or known or tier is not None or float(confidence) != 0.0:
            raise ValueError("unknown action must fully abstain")
        if intent not in {"BOUNDARY", "UNKNOWN"}:
            raise ValueError("unknown action has a known intent")
    else:
        raw = row.get("raw_gripper")
        if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError("known action requires finite raw_gripper")
        expected_close = intent == "CLOSE"
        if intent not in {"CLOSE", "OPEN"} or candidate_close != expected_close:
            raise ValueError("known action intent/candidate-close mismatch")
        if expected_close != (float(raw) < 0.5):
            raise ValueError("raw/action semantic mismatch")

    student_valid = bool(row.get("student_valid", True))
    phase_name = str(row.get("phase_name", "UNKNOWN"))
    window_id = str(row.get("window_id", "none:"))

    # Replicate mature V5 loader rankable gate exactly:
    #   student_valid AND candidate_close AND known_mask
    #   AND phase_name != "UNKNOWN" AND NOT window_id.startswith("none:")
    control_rankable = (
        student_valid
        and candidate_close
        and known
        and phase_name != "UNKNOWN"
        and not window_id.startswith("none:")
    )

    primary_known = control_rankable
    primary_target = int(tier >= 2) if control_rankable else None

    # Release / regrasp targets use the same veto gate as mature V5
    veto_known = control_rankable
    release_risk = float(row.get("release_risk", 0.0))
    regrasp_risk = float(row.get("regrasp_or_instability_risk", 0.0))
    if not math.isfinite(release_risk) or not math.isfinite(regrasp_risk):
        raise ValueError("non-finite V2.1C risk target")

    # Diagnostic: steps that have tier>=2 but are excluded by the rankable gate.
    # These 4,276 OPEN+UNKNOWN+Tier2/3 steps are NOT training positives but are
    # preserved for forensic monitoring.
    diagnostic_tier23_outside_control = (
        bool(known) and tier is not None and int(tier) >= 2 and not control_rankable
    )

    return {
        "target_mode": "V21C_CONTROL",
        "primary_known": primary_known,
        "primary_target": primary_target,
        "release_known": veto_known,
        "release_target": int(release_risk >= release_threshold) if veto_known else None,
        "regrasp_known": veto_known,
        "regrasp_target": int(regrasp_risk >= regrasp_threshold) if veto_known else None,
        "candidate_close_context": candidate_close,
        "diagnostic_tier23_outside_control": diagnostic_tier23_outside_control,
    }


def validate_factorized_target_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future three-head label interface without defining formulas."""

    heads = ("grasp_established", "manipulation_active", "release_or_instability")
    result: dict[str, Any] = {"target_mode": "FACTORIZED_THREE_HEAD"}
    for head in heads:
        mask_name = f"{head}_known_mask"
        known = row.get(mask_name)
        value = row.get(head)
        if not isinstance(known, bool):
            raise TypeError(f"{mask_name} must be bool")
        if known and not isinstance(value, bool):
            raise TypeError(f"known {head} must be bool")
        if not known and value is not None:
            raise ValueError(f"unknown {head} must be None")
        result[mask_name] = known
        result[head] = value
    if result["manipulation_active_known_mask"] and result["manipulation_active"]:
        if not result["grasp_established_known_mask"] or not result["grasp_established"]:
            raise ValueError("manipulation_active implies grasp_established")
    return result


def build_oof_preparation_plan(
    *,
    fold_root: Path,
    teacher: V21CTeacherBinding,
    prep_protocol_sha256: str,
    seeds: Sequence[int],
    geometry: GeometryGateBinding | None = None,
) -> dict[str, Any]:
    """Build an exact four-fold preparation plan; never authorize execution."""

    _require(_is_sha(prep_protocol_sha256, 64), "invalid preparation protocol SHA")
    unique_seeds = sorted({int(seed) for seed in seeds})
    _require(bool(unique_seeds) and all(seed >= 0 for seed in unique_seeds), "at least one non-negative seed is required")
    folds = load_fit_fold_bundle(fold_root.resolve())
    jobs: list[dict[str, Any]] = []
    for fold in folds["folds"]:
        for seed in unique_seeds:
            jobs.append({
                "fold_id": int(fold["fold_id"]),
                "seed": seed,
                "train_identity_count": int(fold["train_identity_count"]),
                "validation_identity_count": int(fold["validation_identity_count"]),
                "train_identity_sha256": str(fold["train_identity_sha256"]),
                "validation_identity_sha256": str(fold["validation_identity_sha256"]),
                "target_mode": "V21C_CONTROL",
                "model_variant": "V5_A_PROPRIO",
                "status": "PREPARED_NOT_AUTHORIZED",
            })
    return {
        "schema": PREP_PLAN_SCHEMA,
        "status": "PASS_PREPARATION_ONLY" if geometry is not None else "HOLD_GEOMETRY_GATE",
        "teacher_binding": teacher.to_dict(),
        "geometry_binding": None if geometry is None else geometry.to_dict(),
        "fold_root": str(fold_root.resolve()),
        "fold_root_sha256s_sha256": sha256_file(fold_root.resolve() / "SHA256SUMS"),
        "prep_protocol_sha256": prep_protocol_sha256,
        "fit_identity_count": 800,
        "fold_count": 4,
        "seed_count": len(unique_seeds),
        "job_count": len(jobs),
        "jobs": jobs,
        "training_executed": False,
        "checkpoint_write_authorized": False,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "full_fit_authorized": False,
    }


__all__ = [
    "V21CTeacherBinding",
    "GeometryGateBinding",
    "validate_v21c_teacher_root",
    "validate_geometry_gate",
    "extract_v21c_control_targets",
    "validate_factorized_target_row",
    "build_oof_preparation_plan",
]
