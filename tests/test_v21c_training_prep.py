from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gripper_attack.b3_training_protocol import (
    build_fit_fold_manifest,
    sha256_file,
    write_fit_fold_bundle,
)
from gripper_attack.v21c_training_prep import (
    V21CTeacherBinding,
    build_oof_preparation_plan,
    extract_v21c_control_targets,
    validate_factorized_target_row,
    validate_v21c_teacher_root,
)


def _base_row() -> dict:
    return {
        "raw_gripper": 0.0,
        "action_intent": "CLOSE",
        "action_known": True,
        "candidate_close": True,
        "known_mask": True,
        "utility_tier": 2,
        "teacher_confidence": 1.0,
        "student_valid": True,
        "phase_name": "VALID_RETENTION",
        "window_id": "candidate:0",
        "release_risk": 0.1,
        "regrasp_or_instability_risk": 0.2,
    }


def _teacher_binding() -> V21CTeacherBinding:
    return V21CTeacherBinding(
        root="/synthetic/v21c",
        root_sha256s_sha256="1" * 64,
        manifest_sha256="2" * 64,
        protocol_sha256="3" * 64,
        audit_sha256="4" * 64,
        action_contract_sha256="5" * 64,
        v5_physics_sha256="6" * 64,
        source_git_commit="7" * 40,
        identity_count=800,
        step_count=176336,
        task_count=40,
        known_step_count=170107,
        label_file_count=800,
    )


def _recursive_seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name not in excluded),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    sums = root / "SHA256SUMS"
    sums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    (root / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(sums)}  SHA256SUMS\n",
        encoding="utf-8",
    )


def test_v21c_control_known_close_targets():
    targets = extract_v21c_control_targets(
        _base_row(), release_threshold=0.6, regrasp_threshold=0.6
    )
    assert targets["primary_known"] is True
    assert targets["primary_target"] == 1
    assert targets["release_known"] is True
    assert targets["release_target"] == 0
    assert targets["candidate_close_context"] is True


def test_v21c_control_unknown_fully_abstains():
    row = _base_row()
    row.update(
        {
            "raw_gripper": None,
            "action_intent": "UNKNOWN",
            "action_known": False,
            "candidate_close": False,
            "known_mask": False,
            "utility_tier": None,
            "teacher_confidence": 0.0,
        }
    )
    targets = extract_v21c_control_targets(
        row, release_threshold=0.6, regrasp_threshold=0.6
    )
    assert targets["primary_known"] is False
    assert targets["primary_target"] is None
    assert targets["release_known"] is False
    assert targets["release_target"] is None


def test_v21c_control_open_tier2_excluded_by_rankable_gate():
    """OPEN + tier>=2 fails rankable (no candidate_close) — NOT a training positive."""
    row = _base_row()
    row.update(
        {
            "raw_gripper": 1.0,
            "action_intent": "OPEN",
            "candidate_close": False,
            "utility_tier": 2,
            "phase_name": "UNKNOWN",
            "window_id": "none:5",
        }
    )
    targets = extract_v21c_control_targets(
        row, release_threshold=0.6, regrasp_threshold=0.6
    )
    # Rankable gate: candidate_close=False → fails
    assert targets["primary_known"] is False
    assert targets["primary_target"] is None
    assert targets["candidate_close_context"] is False
    assert targets["release_known"] is False
    # Diagnostic: preserved for forensic monitoring
    assert targets["diagnostic_tier23_outside_control"] is True


def test_v21c_control_rejects_raw_intent_mismatch():
    row = _base_row()
    row["raw_gripper"] = 1.0
    with pytest.raises(ValueError, match="raw/action semantic mismatch"):
        extract_v21c_control_targets(
            row, release_threshold=0.6, regrasp_threshold=0.6
        )


def test_v21c_control_rejects_unknown_with_confidence():
    row = _base_row()
    row.update(
        {
            "raw_gripper": None,
            "action_intent": "UNKNOWN",
            "action_known": False,
            "candidate_close": False,
            "known_mask": False,
            "utility_tier": None,
            "teacher_confidence": 0.8,
        }
    )
    with pytest.raises(ValueError, match="unknown action must fully abstain"):
        extract_v21c_control_targets(
            row, release_threshold=0.6, regrasp_threshold=0.6
        )


def test_factorized_target_accepts_three_independent_masks():
    row = {
        "grasp_established": True,
        "grasp_established_known_mask": True,
        "manipulation_active": True,
        "manipulation_active_known_mask": True,
        "release_or_instability": False,
        "release_or_instability_known_mask": True,
    }
    result = validate_factorized_target_row(row)
    assert result["target_mode"] == "FACTORIZED_THREE_HEAD"
    assert result["manipulation_active"] is True


def test_factorized_target_enforces_manipulation_implies_grasp():
    row = {
        "grasp_established": False,
        "grasp_established_known_mask": True,
        "manipulation_active": True,
        "manipulation_active_known_mask": True,
        "release_or_instability": False,
        "release_or_instability_known_mask": True,
    }
    with pytest.raises(ValueError, match="implies grasp_established"):
        validate_factorized_target_row(row)


def test_oof_plan_has_exact_four_fold_jobs(tmp_path: Path):
    rows = []
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        for task in range(10):
            for state in range(20):
                rows.append(
                    {
                        "canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}",
                        "suite": suite,
                        "task_idx": task,
                        "state_id": state,
                        "split": "FIT_TRAIN",
                    }
                )
    manifest = build_fit_fold_manifest(rows, registry_sha256="0" * 64)
    fold_root = tmp_path / "folds"
    write_fit_fold_bundle(fold_root, manifest)
    plan = build_oof_preparation_plan(
        fold_root=fold_root,
        teacher=_teacher_binding(),
        prep_protocol_sha256="8" * 64,
        seeds=[20260720],
    )
    assert plan["status"] == "HOLD_GEOMETRY_GATE"
    assert plan["fold_count"] == 4
    assert plan["job_count"] == 4
    assert all(job["train_identity_count"] == 600 for job in plan["jobs"])
    assert all(job["validation_identity_count"] == 200 for job in plan["jobs"])
    assert plan["training_executed"] is False
    assert plan["checkpoint_write_authorized"] is False


def test_validate_synthetic_v21c_root(tmp_path: Path):
    root = tmp_path / "teacher"
    root.mkdir()
    protocol = {
        "schema": "DETECTOR_V5_PHYSICS_TEACHER_PROTOCOL_V21C_ACTION_CANONICAL"
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    manifest = {
        "schema": "DETECTOR_V5_PHYSICS_TEACHER_V21C_MANIFEST",
        "teacher_version": "V2.1C",
        "protocol_schema": protocol["schema"],
        "protocol_sha256": sha256_file(protocol_path),
        "action_contract_schema": "CANONICAL_ACTION_CONTRACT_V1",
        "action_contract_sha256": "a" * 64,
        "v5_physics_sha256": "b" * 64,
        "source_git_commit": "c" * 40,
        "identity_count": 800,
        "step_count": 176336,
        "task_count": 40,
        "known_step_count": 170107,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    (root / "physics_teacher_v21c_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "action_contract.json").write_text(
        json.dumps(
            {
                "schema": "CANONICAL_ACTION_CONTRACT_V1",
                "file_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (root / "audit_report.json").write_text(
        json.dumps(
            {
                "schema": "DETECTOR_V5_PHYSICS_TEACHER_V21C_AUDIT_V1",
                "status": "PASS_WITH_EXPLICIT_NON_GRASP_TASKS",
                "manifest": manifest,
                "formal_training_authorized": False,
                "formal_attack_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        for task in range(10):
            for state in range(20):
                path = root / "labels" / suite / f"task_{task:02d}" / f"state_{state:02d}" / "physics_teacher_v21c.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
    _recursive_seal(root)
    binding = validate_v21c_teacher_root(
        root, expected_source_commit="c" * 40
    )
    assert binding.identity_count == 800
    assert binding.step_count == 176336
    assert binding.label_file_count == 800
