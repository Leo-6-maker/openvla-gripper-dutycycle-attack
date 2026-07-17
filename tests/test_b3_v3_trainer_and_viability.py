import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gripper_attack.b3_formal import B3Normalization  # noqa: E402
from gripper_attack.b3_v3_dataset import B3Episode  # noqa: E402


def _load(path_name):
    path = Path(__file__).parents[1] / "scripts" / "detector" / path_name
    spec = importlib.util.spec_from_file_location(path_name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_overfit_is_not_real_effectiveness_claim():
    runner = _load("run_b3_v3_overfit_smoke.py")
    result = runner.run_overfit_smoke(variant="B3_25D", steps=8)
    assert result["status"] == "PASS_SYNTHETIC_ONLY"
    assert result["final_loss"] < result["initial_loss"]


def test_viability_is_fit_only_and_event_level():
    evaluator = _load("evaluate_b3_v3_viability.py")
    rows = [
        {"canonical_parent_key": "libero_10/task_00/state_00", "state_id": 0, "split": "FIT_TRAIN", "event_id": 0, "event_ordinal": 0, "step": 0, "target_t10_known": True, "target_t10": True, "pred_emit": True, "release_imminent": False},
        {"canonical_parent_key": "libero_10/task_00/state_00", "state_id": 0, "split": "FIT_TRAIN", "event_id": 0, "event_ordinal": 0, "step": 1, "target_t10_known": True, "target_t10": False, "pred_emit": False, "release_imminent": False},
    ]
    metrics = evaluator.event_level_metrics(rows)
    assert metrics["full_t10_event_hit_count"] == 1
    assert metrics["effectiveness_metrics_are_not_attack_results"] is True
    with pytest.raises(ValueError, match="FIT_TRAIN"):
        evaluator.validate_fit_only([dict(rows[0], state_id=24, split="CAL")])


def test_formal_trainer_rejects_non_fit_episode():
    trainer = _load("train_b3_v3_detector.py")
    episode = B3Episode(
        canonical_parent_key="libero_object/task_00/state_20", suite="libero_object", task_idx=0, state_id=20,
        split="FIT_DEV", task_success=True, features_25d=torch.zeros(2, 25),
        targets={head: torch.zeros(2) for head in ("grasp_support", "retention_active", "retention_continuation_t10", "release_imminent")},
        known_masks={head: torch.ones(2, dtype=torch.bool) for head in ("grasp_support", "retention_active", "retention_continuation_t10", "release_imminent")},
        valid_mask=torch.ones(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="FIT_TRAIN"):
        trainer.train_model([episode], variant="B3_25D", normalization=B3Normalization.identity(), epochs=1)
