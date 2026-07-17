import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_formal import B3Normalization, B3ModelConfig, build_b3_model
from gripper_attack.b3_v3_dataset import B3Episode


def _load(name):
    path = Path(__file__).parents[1] / "scripts" / "detector" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(index: int, variant: str = "B3_25D") -> B3Episode:
    return B3Episode(
        canonical_parent_key=f"libero_object/task_{index % 10:02d}/state_{index // 10:02d}",
        suite="libero_object", task_idx=index % 10, state_id=index // 10, split="FIT_TRAIN",
        task_success=True, features_25d=torch.zeros(2, 25),
        features_9d=torch.zeros(2, 9) if variant == "B3_25D9D" else None,
        targets={head: torch.tensor([1.0, 0.0]) for head in ("grasp_support", "retention_active", "retention_continuation_t10", "release_imminent")},
        known_masks={head: torch.ones(2, dtype=torch.bool) for head in ("grasp_support", "retention_active", "retention_continuation_t10", "release_imminent")},
        valid_mask=torch.ones(2, dtype=torch.bool), event_ids=(0, 0),
    )


def test_fold_prediction_bundle_is_sealed_and_student_only(tmp_path):
    module = _load("build_b3_v3_prediction_bundle.py")
    model = build_b3_model(B3ModelConfig()).eval()
    episodes = [_episode(index) for index in range(200)]
    records = module.build_prediction_records(
        model, episodes, B3Normalization.identity(), checkpoint_sha256="a" * 64,
        fold_id=0, seed=20260717, variant="B3_25D",
    )
    root = tmp_path / "prediction"
    manifest = module.write_prediction_bundle(root, records, fold_id=0, seed=20260717, variant="B3_25D", checkpoint_sha256="a" * 64, validation_identity_sha256="b" * 64)
    loaded, rows = module.load_prediction_bundle(root)
    assert manifest["validation_identity_count"] == loaded["validation_identity_count"] == 200
    assert len(rows) == 400
    assert all(row["attack_enabled"] is False and row["teacher_inputs_consumed"] is False for row in rows)


def test_viability_aggregate_requires_exact_24_coordinates(tmp_path):
    module = _load("build_b3_v3_prediction_bundle.py")
    aggregate = _load("aggregate_b3_v3_fit_viability.py")
    roots = []
    identities = [f"libero_object/task_{index % 10:02d}/state_{index // 10:02d}" for index in range(200)]
    rows = [{
        "schema": "B3_OFFICIAL_V3_FIT_PREDICTION_RECORD_V1", "canonical_parent_key": key,
        "suite": "libero_object", "task_idx": index % 10, "state_id": index // 10, "split": "FIT_TRAIN",
        "step": 0, "event_id": 0, "event_ordinal": 0, "target_t10_known": True, "target_t10": True,
        "pred_emit": True, "release_imminent": False, "recent_close_streak": 3, "time_since_close": 5,
        "attack_enabled": False, "teacher_inputs_consumed": False,
    } for index, key in enumerate(identities)]
    for fold in range(4):
        for variant in ("B3_25D", "B3_25D9D"):
            for seed in (20260717, 20260718, 20260719):
                root = tmp_path / f"fold{fold}_{variant}_{seed}"
                module.write_prediction_bundle(root, [dict(row, fold_id=fold, variant=variant, seed=seed, checkpoint_sha256="a" * 64) for row in rows], fold_id=fold, seed=seed, variant=variant, checkpoint_sha256="a" * 64, validation_identity_sha256="b" * 64)
                roots.append(root)
    output = tmp_path / "aggregate"
    report = aggregate.aggregate_viability(roots, output)
    assert report["run_count"] == 24
    assert aggregate._prediction_module().load_prediction_bundle(roots[0])[0]["attack_enabled"] is False
