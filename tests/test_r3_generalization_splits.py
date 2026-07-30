import numpy as np
import pytest
import copy
import json
from pathlib import Path

from scripts.detector_v5.build_r3_generalization_splits import (
    _event_stats,
    _episode_split,
    _manifest_row,
    _summarize_split,
    _task_split,
    _train_normalization,
    _validate_g0_permissions,
    _validate_protocol_contract,
    _validate_input_root,
    _validate_output_path,
)


def _identities(n=40):
    return [{"episode_id": f"ep/{i:03d}", "suite": f"suite_{i // 10}", "task_id": i % 10} for i in range(n)]


def test_episode_split_is_deterministic_and_disjoint():
    items = _identities(120)
    first = _episode_split(items, 20260717)
    second = _episode_split(items, 20260717)
    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*(set(v) for v in first.values())) == {item["episode_id"] for item in items}


def test_task_split_is_exact_30_5_5():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    tasks = [(suites[i // 10], i % 10) for i in range(40)]
    split = _task_split(tasks, 20260717)
    assert [len(split[name]) for name in ("train", "validation", "test")] == [30, 5, 5]
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])


def test_task_split_rejects_noncanonical_grid():
    with pytest.raises(ValueError, match="canonical"):
        _task_split([("suite_0", i) for i in range(40)], 20260717)


def test_normalization_is_train_only():
    records = {
        "a": {"features": np.zeros((2, 25), dtype=np.float32)},
        "b": {"features": np.ones((2, 25), dtype=np.float32)},
        "heldout": {"features": np.full((2, 25), 100.0, dtype=np.float32)},
    }
    stats = _train_normalization(["a", "b"], records)
    assert stats["source_split"] == "train"
    assert stats["mean"][0] == pytest.approx(0.5)
    assert stats["std"][0] == pytest.approx(0.5)
    assert stats["mean"][0] != 100.0


def test_episode_group_too_small_fails_closed():
    with pytest.raises(ValueError, match="too small"):
        _episode_split([{"episode_id": "only", "suite": "s", "task_id": 0}], 20260717)


def test_unknown_event_requires_a_reason():
    rows = [{
        "episode_id": "ep/0",
        "step": 0,
        "candidate_close": True,
        "labels": {"physical_criticality": {"value": "UNKNOWN", "valid_mask": False, "mask": False, "right_censored": False}},
    }]
    with pytest.raises(ValueError, match="no reason"):
        _event_stats(rows, "physical_criticality", {"suite": "s", "task_id": 0})


def test_manifest_has_no_teacher_fields():
    row = _manifest_row(
        "ep/0",
        {
            "suite": "s", "task_id": 0, "state_id": 0, "seed": 1,
            "relative_path": "episodes/ep/0/episode.json",
            "episode_sha256": "a" * 64,
            "episode_sha256sums_sha256": "b" * 64,
            "initial_state_sha256": "c" * 64,
            "collection_source_commit": "d" * 40,
            "collection_source_tree": "e" * 40,
        },
        {"teacher_root_sha256sums_sha256": "f" * 64, "t4_seal_sha256sums_sha256": "0" * 64, "feature_order_sha256": "1" * 64, "t0a_manifest_sha256": "2" * 64, "t0a_root_sha256sums_sha256": "3" * 64, "t0a_identity_set_digest": "4" * 64},
        "5" * 64,
        "6" * 64,
        "7" * 64,
    )
    assert "labels" not in row
    assert "event_id" not in row


def test_event_task_and_suite_denominators_are_unique():
    rows = {
        "a": [{"episode_id": "a", "step": 0, "candidate_close": True, "labels": {"physical_criticality": {"value": "TRUE", "valid_mask": True, "mask": True, "right_censored": False}}}],
        "b": [{"episode_id": "b", "step": 0, "candidate_close": True, "labels": {"physical_criticality": {"value": "TRUE", "valid_mask": True, "mask": True, "right_censored": False}}}],
    }
    metadata = {"a": {"suite": "libero_spatial", "task_id": 0}, "b": {"suite": "libero_spatial", "task_id": 0}}
    summary = _summarize_split(["a", "b"], rows, metadata)
    assert summary["heads"]["physical_criticality"]["positive_episodes"] == 2
    assert summary["heads"]["physical_criticality"]["positive_tasks"] == 1
    assert summary["heads"]["physical_criticality"]["positive_suites"] == 1


def test_path_contract_rejects_parent_components(tmp_path):
    with pytest.raises(ValueError):
        _validate_input_root(tmp_path / ".." / tmp_path.name, "root")
    with pytest.raises(ValueError):
        _validate_output_path(tmp_path / ".." / "out", tmp_path)


def test_protocol_semantics_are_fail_closed():
    protocol = json.loads((Path(__file__).parents[1] / "configs" / "R3_GENERALIZATION_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    _validate_protocol_contract(protocol)
    mutated = copy.deepcopy(protocol)
    mutated["permissions"]["attack"] = True
    with pytest.raises(ValueError, match="permissions"):
        _validate_protocol_contract(mutated)
    mutated = copy.deepcopy(protocol)
    mutated["input_scope"]["teacher_root_status"] = "UNSEALED"
    with pytest.raises(ValueError, match="input scope"):
        _validate_protocol_contract(mutated)


def test_g0_permission_matrix_is_exact():
    good = {"teacher_label_read": True, "student_training": False, "formal_training_authorized": False, "heldout_evaluation": False, "protected_reads": 0, "CAL_READ": False, "CHECK_READ": False, "G10_READ": False, "T2R_D_READ": False, "shadow": False, "rollout": False, "attack": False}
    _validate_g0_permissions(good)
    bad = dict(good, rollout=True)
    with pytest.raises(ValueError, match="permission"):
        _validate_g0_permissions(bad)
