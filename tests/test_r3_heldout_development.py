import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.detector_v5.run_r3_heldout_development import (
    ACTIVE_HEADS,
    _active_masks,
    _check_split_closure,
    _event_label,
    _event_metrics,
    _load_split_ids,
    _safe_auc,
    _validate_g2_permissions,
    _write_predictions,
)


def _item(identity="ep/0"):
    labels = {head: np.zeros(3, dtype=np.float32) for head in ACTIVE_HEADS}
    masks = {head: np.asarray([True, False, True], dtype=bool) for head in ACTIVE_HEADS}
    return {
        "identity": identity,
        "features": np.zeros((3, 25), dtype=np.float32),
        "candidate_close": np.asarray([True, True, False], dtype=bool),
        "targets": labels,
        "masks": masks,
        "weights": {head: np.ones(3, dtype=np.float32) for head in ACTIVE_HEADS},
    }


def test_tri_valued_event_true_dominates_unknown():
    item = _item()
    item["targets"]["physical_criticality"][:] = [1.0, 0.0, 0.0]
    assert _event_label(item, "physical_criticality", 0, 1) == "TRUE"


def test_tri_valued_event_false_unknown_is_unknown():
    item = _item()
    assert _event_label(item, "physical_criticality", 0, 1) == "UNKNOWN"


def test_event_metric_keeps_known_true_with_unknown_step():
    item = _item()
    item["targets"]["physical_criticality"][:] = [1.0, 0.0, 0.0]
    probabilities = {item["identity"]: np.asarray([[0.9, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])}
    result = _event_metrics([item], [item["identity"]], "physical_criticality", probabilities, 0.5)
    assert result["known_events"] == 1
    assert result["positive_events"] == 1
    assert result["event_recall"] == 1.0


def test_auc_is_none_without_both_classes():
    assert _safe_auc(np.asarray([1, 1]), np.asarray([0.1, 0.2])) is None
    assert _safe_auc(np.asarray([0, 1]), np.asarray([0.1, 0.9])) == 1.0


def test_disabled_heads_are_zero_masks():
    masks = {head: torch.ones((1, 2), dtype=torch.bool) for head in ACTIVE_HEADS + ("safe_release",)}
    active = _active_masks(masks, ("physical_criticality",))
    assert bool(active["physical_criticality"].all())
    assert not bool(active["k10_feasibility"].any())
    assert not bool(active["safe_release"].any())


def test_g2_permission_types_are_strict():
    good = {"teacher_labels_read": True, "fit_development_features_read": True, "student_training": True, "development_inference": True, "privileged_oracle_diagnostic": True, "shadow_offline": False, "shadow_live": False, "formal_training": False, "full_fit": False, "rollout": False, "attack": False, "protected_reads": 0}
    _validate_g2_permissions(good)
    with pytest.raises(ValueError, match="type/value"):
        _validate_g2_permissions({**good, "protected_reads": False})
    with pytest.raises(ValueError, match="type/value"):
        _validate_g2_permissions({**good, "attack": 0})


def test_split_manifest_is_bound_to_g2_expected_sha(tmp_path: Path):
    path = tmp_path / "EPISODE_TRAIN_MANIFEST.json"
    path.write_text(json.dumps([{"episode_id": "ep/0"}]), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _load_split_ids(tmp_path, "episode_train", {"file_sha256": digest, "identity_ids": ["ep/0"], "identity_count": 1}) == ["ep/0"]
    with pytest.raises(ValueError, match="file binding"):
        _load_split_ids(tmp_path, "episode_train", {"file_sha256": "0" * 64, "identity_ids": ["ep/0"], "identity_count": 1})


def test_split_closure_rejects_missing_loaded_identity():
    item = _item()
    with pytest.raises(ValueError, match="split does not close"):
        _check_split_closure({"episode_train": [item["identity"]], "episode_validation": ["ep/1"]}, [item], "episode", loaded_ids={item["identity"], "ep/1"})


def test_predictions_serialize_only_active_heads(tmp_path: Path):
    item = _item()
    probabilities = {item["identity"]: np.full((3, len(ACTIVE_HEADS)), 0.25, dtype=np.float64)}
    _write_predictions(tmp_path, [item], {"episode_train": [item["identity"]]}, probabilities, {"physical_criticality": 0.5}, ("physical_criticality",))
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(set(row) == {"candidate_close", "episode_id", "physical_criticality", "split", "step"} for row in rows)
