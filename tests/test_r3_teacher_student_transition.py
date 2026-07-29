import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_r3_teacher_student_transition",
    ROOT / "scripts" / "detector_v5" / "build_r3_teacher_student_transition.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _coverage(**overrides):
    data = {head: {"pass": head != "safe_release"} for head in MODULE.HEADS}
    data.update(overrides)
    return data


def test_eligible_heads_are_exactly_four_and_safe_release_is_held():
    eligible, held = MODULE._validate_t3(
        {
            "schema": "V5_R3_TEACHER_COVERAGE_AUDIT_V1",
            "status": "HOLD_COVERAGE",
            "identity_count": 670,
            "step_count": 196483,
            "protected_reads": 0,
            "unknown_as_negative": False,
            "right_censored_as_negative": False,
            "protected_read_audit": {"status": "PASS", "forbidden_root_parts": []},
            "input_root": "/tmp/teacher",
            "input_sha256sums_sha256": "a" * 64,
            "coverage": _coverage(),
        },
        teacher_root=Path("/tmp/teacher"),
        teacher_seal="a" * 64,
    )
    assert eligible == ["physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"]
    assert list(held) == ["safe_release"]


def test_t3_rejects_full_five_or_unexpected_head_set():
    data = {
        "schema": "V5_R3_TEACHER_COVERAGE_AUDIT_V1",
        "status": "HOLD_COVERAGE",
        "identity_count": 670,
        "step_count": 196483,
        "protected_reads": 0,
        "unknown_as_negative": False,
        "right_censored_as_negative": False,
        "protected_read_audit": {"status": "PASS", "forbidden_root_parts": []},
        "input_root": "/tmp/teacher",
        "input_sha256sums_sha256": "a" * 64,
        "coverage": _coverage(safe_release={"pass": True}),
    }
    with pytest.raises(ValueError, match="eligible head set"):
        MODULE._validate_t3(data, teacher_root=Path("/tmp/teacher"), teacher_seal="a" * 64)


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_safe_release_pass_must_be_explicit_false(value):
    data = {
        "schema": "V5_R3_TEACHER_COVERAGE_AUDIT_V1",
        "status": "HOLD_COVERAGE",
        "identity_count": 670,
        "step_count": 196483,
        "protected_reads": 0,
        "unknown_as_negative": False,
        "right_censored_as_negative": False,
        "protected_read_audit": {"status": "PASS", "forbidden_root_parts": []},
        "input_root": "/tmp/teacher",
        "input_sha256sums_sha256": "a" * 64,
        "coverage": _coverage(safe_release={"pass": value}),
    }
    with pytest.raises(ValueError):
        MODULE._validate_t3(data, teacher_root=Path("/tmp/teacher"), teacher_seal="a" * 64)


def test_teacher_manifest_rejects_authoritative_or_unknown_to_negative():
    manifest = {
        "schema": "V5_R3_V23_TEACHER_FORMAL_V1",
        "status": "DEVELOPMENT_NONCONSUMABLE",
        "input_status": "PASS_CONSUMABLE_FINAL",
        "identity_count": 670,
        "step_count": 196483,
        "protected_reads": 0,
        "teacher_labels_generated": True,
        "unknown_to_negative": True,
        "formal_inference_authorized": False,
        "formal_training_authorized": False,
        "attack_authorized": False,
        "future_fields_used": False,
        "outcome_fields_used": False,
        "heads": list(MODULE.HEADS),
    }
    with pytest.raises(ValueError, match="teacher manifest mismatch: unknown_to_negative"):
        MODULE._validate_teacher_manifest(manifest, root_seal="a" * 64)


def test_permissions_keep_formal_and_attack_closed():
    assert MODULE.ELIGIBLE_HEADS == (
        "physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"
    )
    assert "safe_release" not in MODULE.ELIGIBLE_HEADS


def test_role_roots_must_be_physical_siblings(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    root.mkdir()
    nested.mkdir()
    with pytest.raises(ValueError, match="overlap or nest"):
        MODULE._assert_role_roots_disjoint({"teacher": root, "coverage": nested})


def test_t0b_permission_matrix_is_exact():
    good = {
        "fit_episode_read": True,
        "teacher_label_generation": True,
        "student_dataset_generation": False,
        "student_training": False,
        "detector_load": False,
        "rollout": False,
        "shadow": False,
        "attack": False,
        "protected_payload_read": False,
        "CAL_READ": False,
        "CHECK_READ": False,
        "G10_READ": False,
        "T2R_D_READ": False,
    }
    MODULE._validate_t0b_permissions(good)
    bad = dict(good)
    bad["student_training"] = True
    with pytest.raises(ValueError, match="permission matrix"):
        MODULE._validate_t0b_permissions(bad)


def test_build_seals_metadata_only_transition(tmp_path, monkeypatch):
    """Exercise the full binding path without consuming production payloads."""
    def publish(source, target):
        if target.exists():
            raise FileExistsError(target)
        source.rename(target)

    monkeypatch.setattr(MODULE, "rename_noreplace", publish)
    # The fixture intentionally lives beside the uncommitted test checkout;
    # production execution uses the real clean-worktree guard.
    monkeypatch.setattr(MODULE, "_require_clean_git", lambda repo_root: None)

    def make_root(name):
        path = tmp_path / name
        path.mkdir()
        return path

    t0a_root = make_root("t0a")
    t0a = {
        "schema": "V5_R3_FORMAL_INPUT_AUDIT_V1",
        "status": "PASS_FORMAL_INPUT_CONSUMABLE",
        "formal_root": str(tmp_path / "formal"),
        "episode_count": 670,
        "identity_set_digest": "e" * 64,
        "episode_binding_digest": "f" * 64,
        "protected_reads": 0,
        "teacher_labels_generated": False,
    }
    (t0a_root / "FORMAL_INPUT_MANIFEST.json").write_text(json.dumps(t0a, sort_keys=True), encoding="utf-8")
    t0a_seal = MODULE._write_seal(t0a_root)
    t0a_sha = MODULE.sha256_file(t0a_root / "FORMAL_INPUT_MANIFEST.json")

    t0b_root = make_root("t0b")
    permissions = {
        "fit_episode_read": True, "teacher_label_generation": True,
        "student_dataset_generation": False, "student_training": False,
        "detector_load": False, "rollout": False, "shadow": False, "attack": False,
        "protected_payload_read": False, "CAL_READ": False, "CHECK_READ": False,
        "G10_READ": False, "T2R_D_READ": False,
    }
    t0b = {
        "schema": "FIT_TO_TEACHER_TRANSITION_V1",
        "status": "PASS_FIT_TO_TEACHER_AUTHORIZATION",
        "protocol_sha256": MODULE.sha256_file(ROOT / "configs/R3_DEV_PROTOCOL.json"),
        "teacher_contract_sha256": MODULE.sha256_file(ROOT / "src/gripper_attack/v5_r3_teacher.py"),
        "teacher_runner_sha256": MODULE.sha256_file(ROOT / "scripts/detector_v5/run_r3_v23_formal_teacher.py"),
        "formal_root": t0a["formal_root"],
        "identity_count": 670,
        "identity_set_digest": t0a["identity_set_digest"],
        "episode_binding_digest": t0a["episode_binding_digest"],
        "episode_seal_digest": "d" * 64,
        "input_audit_manifest_sha256": t0a_sha,
        "input_audit_seal_sha256sums_sha256": t0a_seal,
        "parent_transition_manifest_sha256": "b" * 64,
        "parent_transition_sha256sums_sha256": "c" * 64,
        "protected_reads": 0,
        "student_training_authorized": False,
        "attack_authorized": False,
        "permissions": permissions,
    }
    t0b_path = t0b_root / "FIT_TO_TEACHER_TRANSITION.json"
    t0b_path.write_text(json.dumps(t0b, sort_keys=True), encoding="utf-8")
    t0b_seal = MODULE._write_seal(t0b_root)
    t0b_sha = MODULE.sha256_file(t0b_path)

    teacher_root = make_root("teacher")
    teacher_binding = {
        "schema": "FIT670_V2_FORMAL_CONSUMABLE_INPUT_V1",
        "status": "PASS_CONSUMABLE_FINAL",
        "formal_root": t0a["formal_root"],
        "identity_count": 670, "step_count": 196483, "protected_reads": 0,
        "formal_inference_authorized": False, "formal_training_authorized": False, "attack_authorized": False,
        "input_audit": {"manifest": t0a, "manifest_sha256": t0a_sha, "root": str(t0a_root), "seal_sha256sums_sha256": t0a_seal, "seal": {"sha256sums_sha256": t0a_seal}},
        "fit_to_teacher_transition": {"manifest": t0b, "manifest_sha256": t0b_sha, "manifest_path": str(t0b_path), "seal_sha256sums_sha256": t0b_seal, "seal": {"sha256sums_sha256": t0b_seal}},
        "selection": {"schema": "V5_R3_FULL_FORMAL_SELECTION_FROM_T0_A_V1", "status": "PASS_FULL_FORMAL_T2_SELECTION", "identity_count": 670, "manifest_sha256": t0a_sha, "seal_sha256sums_sha256": t0a_seal},
        "finalization": {"identity_set_digest": t0a["identity_set_digest"], "episode_seal_digest": t0b["episode_seal_digest"]},
        "transition": {"manifest_sha256": t0b["parent_transition_manifest_sha256"], "seal_sha256sums_sha256": t0b["parent_transition_sha256sums_sha256"]},
    }
    teacher_manifest = {
        "schema": "V5_R3_V23_TEACHER_FORMAL_V1", "status": "DEVELOPMENT_NONCONSUMABLE", "selection_mode": "FULL_FORMAL_T2",
        "input_status": "PASS_CONSUMABLE_FINAL", "identity_count": 670, "step_count": 196483,
        "protocol_sha256": MODULE.sha256_file(ROOT / "configs/R3_DEV_PROTOCOL.json"),
        "protected_reads": 0, "teacher_labels_generated": True, "unknown_to_negative": False,
        "formal_inference_authorized": False, "formal_training_authorized": False, "attack_authorized": False,
        "future_fields_used": False, "outcome_fields_used": False, "heads": list(MODULE.HEADS), "input_binding": teacher_binding,
    }
    (teacher_root / "teacher_manifest.json").write_text(json.dumps(teacher_manifest, sort_keys=True), encoding="utf-8")
    (teacher_root / "teacher_records.jsonl").write_text('{"episode_id":"x","step":0}\n', encoding="utf-8")
    teacher_seal = MODULE._write_seal(teacher_root)

    coverage_root = make_root("coverage")
    coverage = {
        "schema": "V5_R3_TEACHER_COVERAGE_AUDIT_V1", "status": "HOLD_COVERAGE", "identity_count": 670,
        "step_count": 196483, "protected_reads": 0, "unknown_as_negative": False, "right_censored_as_negative": False,
        "protocol_sha256": MODULE.sha256_file(ROOT / "configs/R3_DEV_PROTOCOL.json"),
        "protected_read_audit": {"status": "PASS", "forbidden_root_parts": []},
        "input_root": str(teacher_root), "input_sha256sums_sha256": teacher_seal,
        "coverage": {head: {"pass": head != "safe_release"} for head in MODULE.HEADS},
    }
    (coverage_root / "coverage_report.json").write_text(json.dumps(coverage, sort_keys=True), encoding="utf-8")
    MODULE._write_seal(coverage_root)

    code_commit, code_tree = MODULE._git_snapshot(ROOT)
    report = MODULE.build(
        teacher_root=teacher_root, coverage_root=coverage_root, input_audit_root=t0a_root,
        fit_transition=t0b_path, protocol=ROOT / "configs/R3_DEV_PROTOCOL.json",
        feature_binding=ROOT / "configs/R3_SC5_FEATURE_BINDING_V1.json", output_root=tmp_path / "t4",
        repo_root=ROOT, code_commit=code_commit, code_tree=code_tree, environment="synthetic",
    )
    assert report["eligible_heads"] == list(MODULE.ELIGIBLE_HEADS)
    assert report["safe_release_training_authorized"] is False
    assert MODULE.verify_seal(tmp_path / "t4")["file_count"] == 3
