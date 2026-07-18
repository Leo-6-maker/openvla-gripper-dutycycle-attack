from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from gripper_attack.v4_contract import FEATURE_INDEX, FEATURE_ORDER_SHA256, SC5_FEATURES, verify_checksum_manifest
from gripper_attack.v4_dataset import V4Episode, derive_dynamic_features, load_v4_episode, select_fold_episodes
from gripper_attack.v4_formal import V4StatefulQualityGRU, compute_v4_loss
from scripts.detector_v4.evaluate_v4_corrected import _metrics, select_working_point
from scripts.detector_v4.build_v4_training_authorization import _verify_teacher_derivative
from scripts.detector_v4.audit_v4_window_semantics import audit_window_semantics
from scripts.detector_v4.build_teacher_v213 import _transform_identity


def _write_seal(root: Path) -> None:
    payloads = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n" for p in payloads))
    value = hashlib.sha256(sums.read_bytes()).hexdigest()
    (root / "SHA256SUMS.sha256").write_text(f"{value}  SHA256SUMS\n")


def _episode(state: int, suite: str = "libero_10", task: int = 0) -> V4Episode:
    return V4Episode(
        canonical_parent_key=f"{suite}/task_{task:02d}/state_{state:02d}", suite=suite,
        task_idx=task, state_id=state, split="FIT_TRAIN", features=torch.zeros(3, 25),
        student_valid_mask=torch.ones(3, dtype=torch.bool), candidate_close=torch.ones(3, dtype=torch.bool),
        quality_target=torch.tensor([1.0, 0.0, -1.0]),
        quality_supervision_mask=torch.tensor([True, True, False]),
        release_target=torch.tensor([0.0, -1.0, -1.0]),
        release_supervision_mask=torch.tensor([True, False, False]),
        event_id=torch.tensor([0, 1, -1]), phase_id=torch.zeros(3, dtype=torch.long),
        window_id=torch.tensor([0, 1, -1]), source_artifact_sha256="a" * 64,
    )


def test_official_feature_order_and_sha_are_frozen() -> None:
    assert SC5_FEATURES[0] == "gripper_command"
    assert SC5_FEATURES[1] == "gripper_qpos"
    assert SC5_FEATURES[3:6] == ("eef_x", "eef_y", "eef_z")
    assert FEATURE_INDEX["time_since_close"] == 17
    assert FEATURE_ORDER_SHA256 == "3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366"


def test_dynamic_features_are_name_bound_and_use_last_valid_history() -> None:
    base = torch.zeros(3, 25)
    base[:, FEATURE_INDEX["gripper_command"]] = torch.tensor([1.0, 1.0, 1.0])
    base[:, FEATURE_INDEX["gripper_qpos"]] = torch.tensor([0.1, 9.0, 0.4])
    base[:, FEATURE_INDEX["eef_x"]] = torch.tensor([0.0, 99.0, 0.3])
    base[:, FEATURE_INDEX["eef_y"]] = torch.tensor([0.0, 99.0, 0.0])
    base[:, FEATURE_INDEX["eef_z"]] = torch.tensor([0.0, 99.0, 0.0])
    base[:, FEATURE_INDEX["time_since_close"]] = torch.tensor([0.0, 1.0, 2.0])
    out = derive_dynamic_features(base, "B", torch.tensor([True, False, True]))
    assert out.shape == (3, 33)
    assert float(out[2, 25]) == pytest.approx(0.3)  # 0.4 - last valid 0.1, not invalid 9.0
    repo_root = Path(__file__).resolve().parents[2]
    assert "IDX_" not in (repo_root / "src/gripper_attack/v4_dataset.py").read_text()


def test_quality_supervision_requires_xor_and_masks_unknown(tmp_path: Path) -> None:
    root = tmp_path / "v4_loader"
    s1 = root / "s1" / "libero_10" / "task_00" / "state_00"
    teacher = root / "teacher" / "libero_10" / "task_00" / "state_00"
    s1.mkdir(parents=True)
    teacher.mkdir(parents=True)
    students = [
        {"canonical_parent_key": "libero_10/task_00/state_00", "step": i, "features_25d": [0.0] * 25, "valid": True}
        for i in range(3)
    ]
    labels = [
        {"step": 0, "candidate_close": True, "quality_valid": True, "veto_invalid": True, "known_mask": True, "event_id": 0, "phase": "VALID_RETENTION"},
        {"step": 1, "candidate_close": True, "quality_valid": False, "veto_invalid": True, "known_mask": True, "event_id": 1, "phase": "RELEASE_IMMINENT_TAIL"},
        {"step": 2, "candidate_close": True, "quality_valid": False, "veto_invalid": False, "known_mask": True, "event_id": 2, "phase": "PRE_SUPPORT"},
    ]
    (s1 / "student_input_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in students))
    (teacher / "teacher_v21_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    ep = load_v4_episode(root / "s1", root / "teacher", "libero_10", 0, 0, "A")
    assert ep is not None
    assert ep.quality_supervision_mask.tolist() == [False, True, False]
    assert ep.quality_target.tolist() == [-1.0, 0.0, -1.0]


def test_window_ranking_has_gradient_and_includes_hard_negative() -> None:
    logits = {"quality": torch.tensor([[0.1, -0.1, 0.2]], requires_grad=True)}
    loss, details = compute_v4_loss(
        logits, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[True, True, True]]),
        window_id=torch.tensor([[0, 1, 1]]), ranking_weight=1.0, hard_negative_weight=1.0,
    )
    loss.backward()
    assert details["window_ranking"] > 0
    assert logits["quality"].grad is not None
    assert float(logits["quality"].grad.abs().sum()) > 0


def test_fold_selection_is_600_train_200_validation_and_disjoint() -> None:
    episodes = [_episode(state, suite, task) for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial") for task in range(10) for state in range(20)]
    for fold in range(4):
        train = select_fold_episodes(episodes, fold, "train")
        valid = select_fold_episodes(episodes, fold, "validation")
        assert len(train) == 600
        assert len(valid) == 200
        assert {e.canonical_parent_key for e in train}.isdisjoint({e.canonical_parent_key for e in valid})


def test_checksum_manifest_detects_tamper(tmp_path: Path) -> None:
    root = tmp_path / "sealed"
    root.mkdir()
    (root / "payload.txt").write_text("original")
    _write_seal(root)
    assert verify_checksum_manifest(root)["status"] == "PASS"
    (root / "payload.txt").write_text("tampered")
    with pytest.raises(ValueError):
        verify_checksum_manifest(root)


def test_model_resets_only_at_episode_boundary_and_accepts_valid_mask() -> None:
    model = V4StatefulQualityGRU(25, hidden_dim=4)
    x = torch.zeros(1, 3, 25)
    valid = torch.tensor([[True, False, True]])
    boundary = torch.tensor([[True, False, False]])
    result = model(x, valid, boundary)
    assert result["quality"].shape == (1, 3)


def test_evaluator_applies_candidate_close_and_student_valid_gate() -> None:
    ep = V4Episode(
        canonical_parent_key="libero_10/task_00/state_00", suite="libero_10", task_idx=0, state_id=0,
        split="FIT_TRAIN", features=torch.zeros(2, 25),
        student_valid_mask=torch.tensor([True, True]), candidate_close=torch.tensor([False, True]),
        quality_target=torch.tensor([1.0, 0.0]), quality_supervision_mask=torch.tensor([True, True]),
        release_target=torch.tensor([1.0, -1.0]), release_supervision_mask=torch.tensor([True, False]),
        event_id=torch.tensor([0, 0]), phase_id=torch.tensor([0, 1]), window_id=torch.tensor([0, 0]),
        source_artifact_sha256="a" * 64,
    )
    report = _metrics([ep], {ep.canonical_parent_key: torch.tensor([0.99, 0.99])}, 0.5)
    assert report["valid_event_hit_n"] == 0
    assert report["release_overlap_n"] == 0


def test_working_point_selector_uses_maximum_threshold_rule() -> None:
    config = {
        "working_point_rule": "maximum_threshold_with_valid_event_hit_gte_0.95",
        "valid_event_hit_minimum": 0.95,
    }
    metrics = [
        {"threshold": 0.05, "valid_event_hit": 1.0},
        {"threshold": 0.50, "valid_event_hit": 0.96},
        {"threshold": 0.65, "valid_event_hit": 0.95},
        {"threshold": 0.70, "valid_event_hit": 0.94},
    ]
    selected = select_working_point(metrics, config)
    assert selected["status"] == "PASS"
    assert selected["threshold"] == pytest.approx(0.65)
    held = select_working_point([{ "threshold": 0.5, "valid_event_hit": 0.94}], config)
    assert held["status"] == "HOLD"
    assert held["threshold"] is None


def test_teacher_derivative_audit_is_bound_to_the_supplied_root(tmp_path: Path) -> None:
    root = tmp_path / "teacher"
    root.mkdir()
    manifest = {
        "schema": "DETECTOR_V4_TEACHER_V212_V1_MANIFEST",
        "source_root_sha256s_sha256": "b" * 64,
        "identity_count": 800,
        "xor_failures": 0,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    (root / "teacher_v212_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_seal(root)
    root_sha = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    audit = tmp_path / "teacher_audit.json"
    audit.write_text(json.dumps({
        "schema": "DETECTOR_V4_WINDOW_SEMANTICS_AUDIT_V1",
        "status": "PASS",
        "teacher_root_sha256s_sha256": root_sha,
        "identity_count": 800,
        "xor_failures": 0,
    }), encoding="utf-8")
    result = _verify_teacher_derivative(root, audit)
    assert result["root_sha256s_sha256"] == root_sha
    audit.write_text(json.dumps({"status": "PASS", "teacher_root_sha256s_sha256": "c" * 64, "identity_count": 800, "xor_failures": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        _verify_teacher_derivative(root, audit)


def test_window_semantics_census_detects_event_split_across_phases(tmp_path: Path) -> None:
    root = tmp_path / "teacher"
    episode = root / "libero_10" / "task_00" / "state_00"
    episode.mkdir(parents=True)
    labels = [
        {"step": 0, "event_id": 0, "phase_name": "PRE_SUPPORT", "window_start": 0, "window_end": 1, "quality_valid": False, "veto_invalid": False},
        {"step": 1, "event_id": 0, "phase_name": "VALID_RETENTION", "window_start": 0, "window_end": 1, "quality_valid": True, "veto_invalid": False},
    ]
    (episode / "teacher_v212_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    (episode / "close_phases.json").write_text(json.dumps({"phases": [{"event_id": 0, "start_step": 0, "end_step": 1}]}), encoding="utf-8")
    _write_seal(root)
    summary = audit_window_semantics(root, tmp_path / "audit")
    assert summary["status"] == "HOLD"
    assert summary["multi_phase_event_identity_count"] == 1
    assert summary["new_teacher_derivative_required"] is True


def test_teacher_v213_assigns_distinct_window_ids_to_phase_segments(tmp_path: Path) -> None:
    source = tmp_path / "source" / "libero_10" / "task_00" / "state_00"
    target = tmp_path / "target" / "libero_10" / "task_00" / "state_00"
    source.mkdir(parents=True)
    labels = [
        {"step": 0, "event_id": 0, "phase": "PRE_SUPPORT", "candidate_close": True, "quality_valid": False, "veto_invalid": False, "known_mask": True},
        {"step": 1, "event_id": 0, "phase": "VALID_RETENTION", "candidate_close": True, "quality_valid": True, "veto_invalid": False, "known_mask": True},
        {"step": 2, "event_id": -1, "phase": "NO_CLOSE", "candidate_close": False, "quality_valid": False, "veto_invalid": False, "known_mask": False},
    ]
    (source / "teacher_v212_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    (source / "close_phases.json").write_text(json.dumps({"phases": [{"event_id": 0, "start_step": 0, "end_step": 1}]}), encoding="utf-8")
    _transform_identity(source, target, "libero_10/task_00/state_00")
    rows = [json.loads(line) for line in (target / "teacher_v213_labels.jsonl").read_text().splitlines()]
    assert [row["window_id"] for row in rows] == [0, 1, -1]
    assert [row["phase_segment_index"] for row in rows] == [0, 1, -1]
    assert [row["window_start"] for row in rows] == [0, 1, -1]
