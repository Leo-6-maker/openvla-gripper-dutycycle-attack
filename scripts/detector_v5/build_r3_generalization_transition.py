"""Seal the FIT-only G2 Teacher -> held-out Student transition.

This is metadata-only.  It validates sealed roots and split closure, but never
loads episode payloads, Teacher rows, features, checkpoints, or a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts" / "detector_v5"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gripper_attack.seal_utils import rename_noreplace
from audit_r3_contact_input import sha256_file, verify_seal
from build_r3_generalization_splits import (
    FORBIDDEN_PATH_PARTS,
    _reject_forbidden,
    _validate_protocol_contract,
)


ACTIVE_HEADS = ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
EXPECTED_SPLITS = (
    "episode_train", "episode_validation", "episode_test",
    "task_train", "task_validation", "task_test",
)
EXPECTED_PERMISSION_MATRIX = {
    "teacher_labels_read": True,
    "fit_development_features_read": True,
    "student_training": True,
    "development_inference": True,
    "privileged_oracle_diagnostic": True,
    "shadow_offline": False,
    "shadow_live": False,
    "formal_training": False,
    "full_fit": False,
    "rollout": False,
    "attack": False,
    "protected_reads": 0,
}


def _git_snapshot() -> tuple[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()

    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _require_clean_git() -> None:
    dirty = subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("consuming checkout is dirty")


def _reject_symlink_components(raw: Path, label: str) -> None:
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be absolute and contain no parent component")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked {label} component: {current}")


def _input_root(raw: Path, label: str) -> Path:
    _reject_symlink_components(raw, label)
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in raw.parts):
        raise ValueError(f"{label} is under a forbidden path")
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError(f"{label} is not a regular directory")
    verify_seal(raw)
    return raw.resolve(strict=True)


def _output_root(raw: Path, parent: Path) -> Path:
    _reject_symlink_components(raw, "output root")
    if raw.exists() or raw.is_symlink():
        raise ValueError(f"output root already exists: {raw}")
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in raw.parts):
        raise ValueError("output root is under a forbidden path")
    if raw.parent.resolve(strict=False) != parent.resolve(strict=True):
        raise ValueError("output root must be a new sibling of the sealed phase roots")
    return raw


def _sha_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }


def _json(root: Path, name: str) -> tuple[dict[str, Any], str]:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value, sha256_file(path)


def _validate_manifest_rows(g1_root: Path, split: str, expected_count: int, expected_bindings: Mapping[str, Any], expected_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    manifest, digest = _json(g1_root, f"{split.upper()}_MANIFEST.json")
    if not isinstance(manifest, list) or len(manifest) != expected_count:
        raise ValueError(f"{split} manifest count mismatch")
    identities: list[str] = []
    for index, row in enumerate(manifest):
        if not isinstance(row, Mapping) or not isinstance(row.get("episode_id"), str):
            raise ValueError(f"malformed {split} manifest row {index}")
        if not isinstance(row.get("suite"), str) or type(row.get("task_id")) is not int:
            raise ValueError(f"malformed task identity in {split}[{index}]")
        authoritative = expected_metadata.get(row["episode_id"])
        if not isinstance(authoritative, Mapping):
            raise ValueError(f"unknown identity in {split}[{index}]")
        for key in ("suite", "task_id", "state_id", "seed"):
            if row.get(key) != authoritative.get(key):
                raise ValueError(f"identity metadata mismatch in {split}[{index}]: {key}")
        identities.append(row["episode_id"])
        _reject_forbidden(row, f"{split}[{index}]")
        if "labels" in row or "teacher_labels" in row or "Teacher" in row:
            raise ValueError(f"forbidden Teacher field in {split}[{index}]")
        for key, expected in expected_bindings.items():
            if row.get(key) != expected:
                raise ValueError(f"binding mismatch in {split}[{index}]: {key}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate identity in {split}")
    return {
        "path": f"{split.upper()}_MANIFEST.json", "sha256": digest,
        "identity_count": len(identities), "identity_ids": identities,
        "task_keys": sorted({f"{row['suite']}:{row['task_id']}" for row in manifest}),
    }


def _validate_split_identity_sets(split_bindings: Mapping[str, Mapping[str, Any]], family: str, expected_identity_ids: set[str]) -> None:
    sets = [set(split_bindings[f"{family}_{name}"]["identity_ids"]) for name in ("train", "validation", "test")]
    if set().union(*sets) != expected_identity_ids:
        raise ValueError(f"{family} split union does not equal the sealed identity set")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError(f"{family} split manifests overlap")


def _validate_task_split_keys(split_bindings: Mapping[str, Mapping[str, Any]], expected_task_keys: set[str]) -> None:
    sets = [set(split_bindings[f"task_{name}"]["task_keys"]) for name in ("train", "validation", "test")]
    if set().union(*sets) != expected_task_keys:
        raise ValueError("task split keys do not equal the sealed task set")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("task split keys overlap")


def _validate_g1(g1_root: Path, *, t4_seal: str, g0_seal: str, feature_order_sha: str, expected_manifest_bindings: Mapping[str, Any], expected_identity_ids: set[str], expected_task_keys: set[str], expected_metadata: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    audit, audit_sha = _json(g1_root, "G1_SPLIT_AUDIT.json")
    if audit.get("status") != "PASS_SPLIT_CLOSURE_WITH_HEAD_COVERAGE_FLAGS":
        raise ValueError("G1 is not the expected split-closure result")
    if audit.get("consumable") is not False or audit.get("development_training_consumable") is not True:
        raise ValueError("G1 consumability boundary is invalid")
    checks = audit.get("checks", {})
    required_checks = {
        "identity_closure": True,
        "episode_intersections": 0,
        "task_intersections": 0,
        "event_intersections": 0,
        "normalization_train_only": True,
        "deterministic": True,
        "teacher_fields_in_manifests": False,
        "protected_reads": 0,
    }
    if any(checks.get(key) != value for key, value in required_checks.items()):
        raise ValueError("G1 closure checks are not passing")
    if audit.get("builder_source", {}).get("commit") != _git_snapshot()[0] or audit.get("builder_source", {}).get("tree") != _git_snapshot()[1]:
        raise ValueError("G1 builder was not run from the consuming code snapshot")
    closure, closure_sha = _json(g1_root, "IDENTITY_CLOSURE.json")
    if closure.get("duplicate_missing_extra") != {"duplicate": 0, "missing": 0, "extra": 0}:
        raise ValueError("G1 identity closure has duplicate/missing/extra identities")
    if closure.get("total_identities") != 670:
        raise ValueError("G1 identity total mismatch")
    split_bindings = {
        "episode_train": _validate_manifest_rows(g1_root, "EPISODE_TRAIN", 445, expected_manifest_bindings, expected_metadata),
        "episode_validation": _validate_manifest_rows(g1_root, "EPISODE_VAL", 81, expected_manifest_bindings, expected_metadata),
        "episode_test": _validate_manifest_rows(g1_root, "EPISODE_TEST", 144, expected_manifest_bindings, expected_metadata),
        "task_train": _validate_manifest_rows(g1_root, "TASK_TRAIN", 498, expected_manifest_bindings, expected_metadata),
        "task_validation": _validate_manifest_rows(g1_root, "TASK_VAL", 83, expected_manifest_bindings, expected_metadata),
        "task_test": _validate_manifest_rows(g1_root, "TASK_TEST", 89, expected_manifest_bindings, expected_metadata),
    }
    _validate_split_identity_sets(split_bindings, "episode", expected_identity_ids)
    _validate_split_identity_sets(split_bindings, "task", expected_identity_ids)
    _validate_task_split_keys(split_bindings, expected_task_keys)
    normalization, normalization_sha = _json(g1_root, "NORMALIZATION.json")
    if set(normalization) != {"episode_heldout", "task_heldout"}:
        raise ValueError("G1 normalization layout is not canonical")
    for family in normalization.values():
        train = family.get("train")
        if not isinstance(train, Mapping) or train.get("source_split") != "train" or train.get("identity_count", 0) <= 0:
            raise ValueError("G1 normalization is not train-only")
        if len(train.get("mean", [])) != 25 or len(train.get("std", [])) != 25:
            raise ValueError("G1 normalization shape mismatch")
    split_protocol, split_protocol_sha = _json(g1_root, "SPLIT_PROTOCOL.json")
    if split_protocol.get("source_commit") != _git_snapshot()[0] or split_protocol.get("source_tree") != _git_snapshot()[1] or split_protocol.get("protected_reads") != 0:
        raise ValueError("G1 split protocol source/boundary mismatch")
    if split_protocol.get("normalization_source") != "train_only" or split_protocol.get("test_read_once") is not True:
        raise ValueError("G1 split protocol is not closed")
    audit_input = audit.get("input_binding", {})
    if audit_input.get("t4_seal_sha256sums_sha256") != t4_seal or audit_input.get("g0_seal_sha256sums_sha256") != g0_seal or audit_input.get("feature_order_sha256") != feature_order_sha:
        raise ValueError("G1 nested source binding mismatch")
    return {
        "audit": {"path": "G1_SPLIT_AUDIT.json", "sha256": audit_sha},
        "identity_closure": {"path": "IDENTITY_CLOSURE.json", "sha256": closure_sha},
        "normalization": {"path": "NORMALIZATION.json", "sha256": normalization_sha},
        "split_protocol": {"path": "SPLIT_PROTOCOL.json", "sha256": split_protocol_sha},
        "split_manifests": split_bindings,
        "file_sha256": _sha_map(g1_root),
        "head_coverage": audit.get("heads", {}),
    }


def _validate_permissions(value: Mapping[str, Any]) -> None:
    if dict(value) != EXPECTED_PERMISSION_MATRIX:
        raise ValueError("G2 permission matrix is not exact development-only scope")


def _validate_t4_permissions(value: Mapping[str, Any]) -> None:
    expected = {
        "teacher_label_read": True, "student_dataset_generation": True,
        "student_training": True, "student_training_scope": "DEVELOPMENT_ONLY",
        "development_student_training_authorized": True,
        "development_inference": True, "development_inference_authorized": True,
        "formal_training_authorized": False, "formal_inference_authorized": False,
        "shadow_authorized": False, "rollout_authorized": False,
        "protected_reads": 0, "CAL_READ": False, "CHECK_READ": False,
        "G10_READ": False, "T2R_D_READ": False, "attack_authorized": False,
    }
    if dict(value) != expected:
        raise ValueError("T4 permission matrix is not exact development-only scope")


def _validate_g0_permissions(value: Mapping[str, Any]) -> None:
    expected = {
        "teacher_label_read": True, "student_training": False,
        "formal_training_authorized": False, "heldout_evaluation": False,
        "protected_reads": 0, "CAL_READ": False, "CHECK_READ": False,
        "G10_READ": False, "T2R_D_READ": False, "shadow": False,
        "rollout": False, "attack": False,
    }
    if dict(value) != expected:
        raise ValueError("G0 permission matrix is not exact diagnostic-only scope")


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def build(*, t4_root: Path, g0_root: Path, g1_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    _require_clean_git()
    commit, tree = _git_snapshot()
    _reject_symlink_components(protocol_path, "protocol")
    if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in protocol_path.parts):
        raise ValueError("protocol is under a forbidden path")
    protocol_path = protocol_path.resolve(strict=True)
    if ROOT.resolve(strict=True) not in protocol_path.parents or protocol_path.is_symlink() or not protocol_path.is_file():
        raise ValueError("protocol must be a regular repository file")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _validate_protocol_contract(protocol)
    protocol_sha = sha256_file(protocol_path)
    t4_root = _input_root(t4_root, "T4 root")
    g0_root = _input_root(g0_root, "G0 root")
    g1_root = _input_root(g1_root, "G1 root")
    if g1_root.parent != t4_root.parent:
        raise ValueError("G1 root must be a sibling of T4")
    t4_seal = verify_seal(t4_root)["sha256sums_sha256"]
    g0_seal = verify_seal(g0_root)["sha256sums_sha256"]
    t4_transition, t4_transition_sha = _json(t4_root, "TEACHER_STUDENT_TRANSITION.json")
    if t4_transition.get("schema") != "V5_R3_TEACHER_STUDENT_TRANSITION_V1" or t4_transition.get("status") != "PASS_DEVELOPMENT_ELIGIBLE_HEADS" or t4_transition.get("protected_reads") != 0 or t4_transition.get("attack_authorized") is not False:
        raise ValueError("T4 is not FIT-only development input")
    _validate_t4_permissions(t4_transition.get("permissions", {}))
    t0a_root = _input_root(Path(str(t4_transition.get("t0_a", {}).get("root"))), "T0-A root")
    t0a_manifest, t0a_manifest_sha = _json(t0a_root, "FORMAL_INPUT_MANIFEST.json")
    if t0a_manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or t0a_manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or t0a_manifest.get("episode_count") != 670 or t0a_manifest.get("protected_reads") != 0:
        raise ValueError("T0-A is not a sealed FIT-only identity source")
    bindings = t0a_manifest.get("episode_bindings")
    if not isinstance(bindings, Mapping) or any(not isinstance(key, str) for key in bindings):
        raise ValueError("T0-A identity bindings are malformed")
    expected_identity_ids = set(str(key) for key in bindings)
    if len(expected_identity_ids) != 670:
        raise ValueError("T0-A identity closure is not exactly 670")
    expected_task_keys: set[str] = set()
    expected_metadata: dict[str, dict[str, Any]] = {}
    for identity, binding in bindings.items():
        if not isinstance(binding, Mapping) or not isinstance(binding.get("suite"), str) or type(binding.get("task_id")) is not int:
            raise ValueError(f"malformed T0-A task identity: {identity}")
        expected_task_keys.add(f"{binding['suite']}:{binding['task_id']}")
        expected_metadata[str(identity)] = {key: binding.get(key) for key in ("suite", "task_id", "state_id", "seed")}
    if len(expected_task_keys) != 40:
        raise ValueError("T0-A task closure is not the canonical 40-task set")
    if t4_transition.get("t0_a", {}).get("manifest_sha256") != t0a_manifest_sha or t4_transition.get("t0_a", {}).get("seal_sha256sums_sha256") != verify_seal(t0a_root)["sha256sums_sha256"]:
        raise ValueError("T4/T0-A seal binding mismatch")
    g0_report, g0_report_sha = _json(g0_root, "G0_LABEL_BASELINE_AUDIT.json")
    if g0_report.get("status") != "PASS_LABEL_AND_BASELINE_AUDIT" or g0_report.get("protected_reads") != 0 or g0_report.get("consumable") is not False:
        raise ValueError("G0 is not a passing non-consumable diagnostic")
    _validate_g0_permissions(g0_report.get("permissions", {}))
    feature_path = ROOT / "configs" / "R3_SC5_FEATURE_BINDING_V1.json"
    feature_binding = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_sha = sha256_file(feature_path)
    feature_order_sha = feature_binding.get("feature_order_sha256")
    if not isinstance(feature_order_sha, str) or len(feature_order_sha) != 64:
        raise ValueError("feature-order binding is missing")
    expected_manifest_bindings = {
        "protocol_sha256": protocol_sha,
        "t4_seal_sha256sums_sha256": t4_seal,
        "g0_report_sha256": g0_report_sha,
        "g0_root_sha256sums_sha256": g0_seal,
        "feature_order_sha256": feature_order_sha,
        "teacher_root_sha256sums_sha256": t4_transition.get("teacher_root_sha256sums_sha256"),
        "t0a_manifest_sha256": t4_transition.get("t0_a", {}).get("manifest_sha256"),
        "t0a_root_sha256sums_sha256": t4_transition.get("t0_a", {}).get("seal_sha256sums_sha256"),
        "t0a_identity_set_digest": t0a_manifest.get("identity_set_digest"),
    }
    if any(not isinstance(value, str) or not value for value in expected_manifest_bindings.values()):
        raise ValueError("incomplete expected G1 manifest binding")
    g1_binding = _validate_g1(g1_root=g1_root, t4_seal=t4_seal, g0_seal=g0_seal, feature_order_sha=feature_order_sha, expected_manifest_bindings=expected_manifest_bindings, expected_identity_ids=expected_identity_ids, expected_task_keys=expected_task_keys, expected_metadata=expected_metadata)
    output_root = _output_root(output_root, t4_root.parent)
    source_files = {
        "split_builder": {"path": "scripts/detector_v5/build_r3_generalization_splits.py", "sha256": sha256_file(ROOT / "scripts/detector_v5/build_r3_generalization_splits.py")},
        "transition_builder": {"path": "scripts/detector_v5/build_r3_generalization_transition.py", "sha256": sha256_file(Path(__file__))},
        "student_trainer_reference": {"path": "scripts/detector_v5/run_r3_full670_student_development.py", "sha256": sha256_file(ROOT / "scripts/detector_v5/run_r3_full670_student_development.py")},
        "feature_binding": {"path": "configs/R3_SC5_FEATURE_BINDING_V1.json", "sha256": feature_sha},
        "generalization_protocol": {"path": "configs/R3_GENERALIZATION_PROTOCOL_V1.json", "sha256": protocol_sha},
    }
    payload: dict[str, Any] = {
        "schema": "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V1",
        "status": "PASS_G2_DEVELOPMENT_TRANSITION",
        "code_snapshot": {"commit": commit, "tree": tree},
        "protocol": {"path": "configs/R3_GENERALIZATION_PROTOCOL_V1.json", "sha256": protocol_sha},
        "t4": {"root": str(t4_root), "seal_sha256sums_sha256": t4_seal, "manifest": {"path": "TEACHER_STUDENT_TRANSITION.json", "sha256": t4_transition_sha}},
        "g0": {"root": str(g0_root), "seal_sha256sums_sha256": g0_seal, "report": {"path": "G0_LABEL_BASELINE_AUDIT.json", "sha256": g0_report_sha}},
        "g1": {"root": str(g1_root), "seal_sha256sums_sha256": verify_seal(g1_root)["sha256sums_sha256"], **g1_binding},
        "feature_binding": {"path": "configs/R3_SC5_FEATURE_BINDING_V1.json", "sha256": feature_sha, "feature_order_sha256": feature_order_sha, "schema": feature_binding.get("schema")},
        "source_files": source_files,
        "expected_split_keys": list(EXPECTED_SPLITS),
        "heads": {"active": list(ACTIVE_HEADS), "safe_release": "HOLD_COVERAGE"},
        "model_boundary": {"random_initialization_required": True, "all_670_engineering_checkpoint_allowed": False, "checkpoint_consumed": False, "privileged_oracle_nondeployable": True},
        "permissions": dict(EXPECTED_PERMISSION_MATRIX),
        "negative_contracts": [
            "split_identity_overlap", "event_overlap", "normalization_from_validation_or_test",
            "all_670_checkpoint_load", "safe_release_enablement", "unknown_in_loss",
            "wrong_teacher_or_feature_seal", "protected_or_attack_permission", "empty_split",
            "empty_minority_class", "task_heldout_leakage",
        ],
        "protected_reads": 0,
        "teacher_privileged_fields_in_student": False,
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
        "shadow_offline_authorized": False,
        "shadow_live_authorized": False,
        "rollout_authorized": False,
        "attack_authorized": False,
        "consumable_for_scientific_promotion": False,
    }
    _reject_forbidden(payload)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V1.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "PERMISSION_MATRIX.json").write_text(json.dumps(EXPECTED_PERMISSION_MATRIX, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "SPLIT_BINDINGS.json").write_text(json.dumps({"expected_split_keys": list(EXPECTED_SPLITS), "bindings": g1_binding["split_manifests"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal = _write_seal(staging)
        rename_noreplace(staging, output_root)
    except Exception as exc:
        (staging / "FAILURE.json").write_text(json.dumps({"schema": "V5_R3_G2_TRANSITION_FAILURE_V1", "error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_seal(staging)
        raise
    payload["sha256sums_sha256"] = seal
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t4-root", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(t4_root=args.t4_root, g0_root=args.g0_root, g1_root=args.g1_root, protocol_path=args.protocol, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
