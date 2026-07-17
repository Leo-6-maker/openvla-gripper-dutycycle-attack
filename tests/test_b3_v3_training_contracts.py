import json

import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_formal import B3ModelConfig, B3Normalization, build_b3_model, json_sha, load_b3_checkpoint_bundle, save_b3_checkpoint_bundle
from gripper_attack.b3_training_protocol import (
    build_fit_fold_manifest,
    build_training_authorization,
    load_fit_fold_bundle,
    load_training_authorization_bundle,
    write_fit_fold_bundle,
    write_normalization_bundle,
)
from gripper_attack.b3_v3_attack_protocol import audit_attack_manifest, build_attack_manifest
from gripper_attack.b3_v3_runtime import B3RuntimeThresholds, B3V3StreamingRuntime


def _rows():
    return [
        {
            "canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}",
            "suite": suite, "task_idx": task, "state_id": state, "split": "FIT_TRAIN",
        }
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10")
        for task in range(10) for state in range(20)
    ]


def _runner():
    value = {
        "status": "PASS", "runner_head": "d" * 40, "runner_worktree_clean": True,
        "runner_script_git_blob_sha1": "e" * 40, "config_git_blob_sha1": "f" * 40,
    }
    value["runner_binding_sha256"] = json_sha(value)
    return value


def _snapshots():
    names = (
        "formal_fit_registry_sha256", "formal_registry_summary_sha256", "formal_registry_root_sha256",
        "s1_corpus_sha256", "s1_root_audit_sha256", "teacher_aggregate_sha256",
        "training_protocol_sha256", "source_contract_sha256", "protocol_sha256", "feature_rebuilder_sha256",
        "normalization_bundle_sha256", "normalization_sha256", "fold_manifest_sha256",
    )
    return {name: f"{index + 1:064x}" for index, name in enumerate(names)}


def test_four_fold_manifest_is_exact_and_sealed(tmp_path):
    manifest = build_fit_fold_manifest(_rows(), registry_sha256="a" * 64)
    assert [(item["train_identity_count"], item["validation_identity_count"]) for item in manifest["folds"]] == [(600, 200)] * 4
    root = tmp_path / "folds"
    write_fit_fold_bundle(root, manifest)
    loaded = load_fit_fold_bundle(root)
    assert loaded["fold_count"] == 4
    with pytest.raises(FileExistsError):
        write_fit_fold_bundle(root, manifest)


def test_machine_auth_and_checkpoint_bundle_do_not_select_model_early(tmp_path):
    auth_root = tmp_path / "auth"
    auth = {
        "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_V1",
        "authorization_status": "PASS", "formal_fit_ready": True,
        "s1_materialization_status": "PASS", "teacher_aggregate_status": "PASS",
        "formal_training_authorized": True, "formal_attack_authorized": False,
        "variant": "B3_25D", "fit_scope": "FIT_FOLD", "fold_id": 0, "seed": 20260717,
        "runner_head": _runner()["runner_head"], "runner_binding": _runner(),
        "input_snapshots": _snapshots(),
        "authorization_generation": {
            "schema": "B3_OFFICIAL_V3_TRAINING_AUTHORIZATION_GENERATOR_V1",
            "generator_script_sha256": "a" * 64, "generator_script_git_blob_sha1": "b" * 40,
            "generator_head": "c" * 40, "generator_worktree_clean": True,
            "generator_script_tracked": True, "generator_entrypoint": "build_b3_v3_training_authorization.py",
            "semantic_inputs_verified": True,
        },
        "verification": {"status": "PASS", "semantic_inputs_verified": True, "normalization_recomputed": True, "runner_binding_measured": True},
    }
    auth.update(_snapshots())
    auth["authorization_payload_sha256"] = json_sha(auth)
    from gripper_attack.b3_training_protocol import _write_json, seal_directory
    auth_root.mkdir()
    _write_json(auth_root / "authorization.json", auth)
    _write_json(auth_root / "input_snapshots.json", {"schema": "B3_OFFICIAL_V3_AUTHORIZATION_INPUT_SNAPSHOTS_V1", **_snapshots()})
    seal_directory(auth_root)
    assert load_training_authorization_bundle(auth_root)["formal_training_authorized"] is True
    forged = dict(auth)
    forged.pop("authorization_generation")
    with pytest.raises(ValueError, match="authorization"):
        from gripper_attack.b3_formal import validate_training_authorization
        validate_training_authorization(forged)
    bundle = tmp_path / "checkpoint"
    save_b3_checkpoint_bundle(bundle, build_b3_model(B3ModelConfig()), B3Normalization.identity(), authorization=auth, checkpoint_status="FIT_FOLD_TRAINED_CANDIDATE")
    _, _, _, payload, manifest = load_b3_checkpoint_bundle(bundle, require_formal=True)
    assert payload["eligible_for_model_selection"] is False
    assert manifest["checkpoint_status"] == "FIT_FOLD_TRAINED_CANDIDATE"


def test_normalization_bundle_binds_fold_and_runtime_is_teacher_free(tmp_path):
    root = tmp_path / "normalization"
    write_normalization_bundle(
        root, B3Normalization.identity(), fold_id=0, variant="B3_25D",
        train_identity_sha256="a" * 64, registry_sha256="b" * 64, s1_corpus_sha256="c" * 64, runner_binding=_runner(),
    )
    model = build_b3_model(B3ModelConfig())
    runtime = B3V3StreamingRuntime(model, B3Normalization.identity(), B3RuntimeThresholds(.5, .5, .9))
    result = runtime.step(torch.zeros(25))
    assert result["teacher_inputs_consumed"] is False
    assert result["attack_enabled"] is False


def test_attack_manifest_is_only_preparation_and_has_exact_condition_pairs():
    rows = [
        {"canonical_parent_key": f"{suite}/task_{task:02d}/state_{30 + offset:02d}", "suite": suite, "task_idx": task, "state_id": 30 + offset, "task_success": True}
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10")
        for task in range(10) for offset in range(5)
    ]
    manifest = build_attack_manifest(
        rows, protocol_sha256="a" * 64, check_status="CHECK_PASS", cs200_manifest_sha256="b" * 64,
        check_report_sha256="c" * 64, checkpoint_sha256="d" * 64, calibration_sha256="e" * 64,
    )
    report = audit_attack_manifest(manifest)
    assert report["status"] == "PASS_PREPARATION_ONLY"
    assert manifest["attack_execution_authorized"] is False
