import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_r3_generalization_transition",
    ROOT / "scripts" / "detector_v5" / "build_r3_generalization_transition.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_permission_matrix_is_exact():
    MODULE._validate_permissions(MODULE.EXPECTED_PERMISSION_MATRIX)
    with pytest.raises(ValueError):
        MODULE._validate_permissions({**MODULE.EXPECTED_PERMISSION_MATRIX, "attack": True})


def test_expected_split_keys_are_exact():
    assert MODULE.EXPECTED_SPLITS == (
        "episode_train", "episode_validation", "episode_test",
        "task_train", "task_validation", "task_test",
    )


def test_episode_and_task_split_union_must_match_authoritative_identities():
    good = {
        "episode_train": {"identity_ids": ["a"]},
        "episode_validation": {"identity_ids": ["b"]},
        "episode_test": {"identity_ids": ["c"]},
        "task_train": {"identity_ids": ["a"]},
        "task_validation": {"identity_ids": ["b"]},
        "task_test": {"identity_ids": ["c"]},
    }
    MODULE._validate_split_identity_sets(good, "episode", {"a", "b", "c"})
    MODULE._validate_split_identity_sets(good, "task", {"a", "b", "c"})
    bad = dict(good, task_test={"identity_ids": ["a"]})
    with pytest.raises(ValueError, match="task"):
        MODULE._validate_split_identity_sets(bad, "task", {"a", "b", "c"})


def test_task_keys_must_be_disjoint_across_task_splits():
    good = {
        "task_train": {"task_keys": ["s:0"]},
        "task_validation": {"task_keys": ["s:1"]},
        "task_test": {"task_keys": ["s:2"]},
    }
    MODULE._validate_task_split_keys(good, {"s:0", "s:1", "s:2"})
    bad = dict(good, task_test={"task_keys": ["s:0"]})
    with pytest.raises(ValueError, match="task split keys"):
        MODULE._validate_task_split_keys(bad, {"s:0", "s:1", "s:2"})


def test_protocol_is_frozen_and_random_init_required():
    protocol = json.loads((ROOT / "configs" / "R3_GENERALIZATION_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    MODULE._validate_protocol_contract(protocol)
    assert protocol["model_configs"]["random_initialization_required"] is True
    assert protocol["model_configs"]["all_670_engineering_checkpoint_allowed"] is False


def test_input_root_rejects_forbidden_components(tmp_path):
    with pytest.raises(ValueError):
        MODULE._input_root(tmp_path / "protected" / "root", "root")


def test_output_root_must_be_new_sibling(tmp_path):
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError):
        MODULE._output_root(tmp_path / "existing", tmp_path)
    assert MODULE._output_root(tmp_path / "new", tmp_path) == tmp_path / "new"


def test_transition_has_no_training_or_attack_authority():
    permissions = MODULE.EXPECTED_PERMISSION_MATRIX
    assert permissions["student_training"] is True
    assert permissions["formal_training"] is False
    assert permissions["shadow_offline"] is False
    assert permissions["attack"] is False


def test_manifest_rows_reject_teacher_fields(tmp_path):
    row = {
        "episode_id": "e", "suite": "s", "task_id": 0, "state_id": 1, "seed": 2,
        "labels": {}, "protocol_sha256": "a" * 64,
        "t4_seal_sha256sums_sha256": "b" * 64, "g0_report_sha256": "c" * 64,
        "g0_root_sha256sums_sha256": "d" * 64, "feature_order_sha256": "e" * 64,
        "teacher_root_sha256sums_sha256": "f" * 64, "t0a_manifest_sha256": "1" * 64,
        "t0a_root_sha256sums_sha256": "2" * 64,
    }
    (tmp_path / "EPISODE_TRAIN_MANIFEST.json").write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        MODULE._validate_manifest_rows(tmp_path, "EPISODE_TRAIN", 1, {key: value for key, value in row.items() if key != "labels"}, {"e": {"suite": "s", "task_id": 0, "state_id": 1, "seed": 2}})


def test_manifest_metadata_is_bound_to_identity(tmp_path):
    row = {
        "episode_id": "e", "suite": "wrong", "task_id": 0, "state_id": 1, "seed": 2,
        "protocol_sha256": "a" * 64, "t4_seal_sha256sums_sha256": "b" * 64,
        "g0_report_sha256": "c" * 64, "g0_root_sha256sums_sha256": "d" * 64,
        "feature_order_sha256": "e" * 64, "teacher_root_sha256sums_sha256": "f" * 64,
        "t0a_manifest_sha256": "1" * 64, "t0a_root_sha256sums_sha256": "2" * 64,
        "t0a_identity_set_digest": "3" * 64,
    }
    (tmp_path / "EPISODE_TRAIN_MANIFEST.json").write_text(json.dumps([row]), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        MODULE._validate_manifest_rows(tmp_path, "EPISODE_TRAIN", 1, {key: row[key] for key in row if key != "t0a_identity_set_digest"}, {"e": {"suite": "s", "task_id": 0, "state_id": 1, "seed": 2}})
