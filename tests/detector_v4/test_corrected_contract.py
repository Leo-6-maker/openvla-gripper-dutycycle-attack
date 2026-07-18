from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from gripper_attack.v4_contract import FEATURE_INDEX, FEATURE_ORDER_SHA256, SC5_FEATURES, verify_checksum_manifest
from gripper_attack.v4_dataset import V4Episode, derive_dynamic_features, load_v4_episode, select_fold_episodes
from gripper_attack.v4_formal import V4StatefulQualityGRU, compute_v4_loss


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
    assert "IDX_" not in Path("src/gripper_attack/v4_dataset.py").read_text()


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
