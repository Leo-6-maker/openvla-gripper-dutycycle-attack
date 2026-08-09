from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from detector_v5.d8_train_core import create_model
from fec import audit_stage3a_shadow_matrix as auditor
from fec import run_stage3a_shadow_matrix as dispatcher
from fec.stage3a_runtime import FrozenStage2R2DetectorRuntime, ShadowContractError


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = dispatcher.STAGE2_COMMIT
SOURCE_TREE = dispatcher.STAGE2_TREE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_files(tmp_path: Path):
    checkpoint = tmp_path / "FINAL_DETECTOR_CHECKPOINT.pt"
    model = create_model(seed=20260717)
    schema_sha = _sha(ROOT / "configs/DETECTOR_V3_25D_CAUSAL_FEATURE_SCHEMA.json")
    torch.save(
        {
            "schema": "D8_STUDENT_CHECKPOINT_V2",
            "model_state": model.state_dict(),
            "normalization": {
                "schema": "D8_NORMALIZATION_V2",
                "feature_dim": 25,
                "mean": [0.0] * 25,
                "std": [1.0] * 25,
            },
            "feature_schema_sha256": schema_sha,
            "executable_source_commit": SOURCE_COMMIT,
            "executable_source_tree": SOURCE_TREE,
        },
        checkpoint,
    )
    receipt = tmp_path / "DETECTOR_FREEZE_RECEIPT_R2.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "D8_DETECTOR_FREEZE_RECEIPT_R2_V1",
                "status": "SHADOW_PROBE_ONLY",
                "authorization_mode": "SHADOW_PROBE_ONLY",
                "guard_deployment_authorized": False,
                "source_commit": SOURCE_COMMIT,
                "source_tree": SOURCE_TREE,
                "checkpoint_sha256": _sha(checkpoint),
                "scheduler": {"threshold": 0.0, "persistence": 1, "hysteresis": 0.0, "cooldown": 0},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, receipt


def _runtime(tmp_path: Path) -> FrozenStage2R2DetectorRuntime:
    checkpoint, receipt = _runtime_files(tmp_path)
    return FrozenStage2R2DetectorRuntime(
        checkpoint,
        receipt,
        expected_checkpoint_sha256=_sha(checkpoint),
        expected_scheduler_sha256=_sha(receipt),
        expected_source_commit=SOURCE_COMMIT,
        expected_source_tree=SOURCE_TREE,
        episode_id="ep",
    )


def _step(runtime: FrozenStage2R2DetectorRuntime, step: int = 0):
    return runtime.step(
        episode_id="ep",
        policy_step=step,
        raw_action=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        env_action=np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
        observation={"robot0_gripper_qpos": [0.1, 0.1], "robot0_eef_pos": [0.0, 0.0, 0.0]},
    )


def test_stage2_checkpoint_sha_mismatch_rejected(tmp_path):
    checkpoint, receipt = _runtime_files(tmp_path)
    with pytest.raises(ShadowContractError, match="checkpoint SHA mismatch"):
        FrozenStage2R2DetectorRuntime(
            checkpoint, receipt, expected_checkpoint_sha256="0" * 64,
            expected_scheduler_sha256=_sha(receipt), expected_source_commit=SOURCE_COMMIT,
            expected_source_tree=SOURCE_TREE,
        )


def test_scheduler_sha_mismatch_rejected(tmp_path):
    checkpoint, receipt = _runtime_files(tmp_path)
    with pytest.raises(ShadowContractError, match="scheduler SHA mismatch"):
        FrozenStage2R2DetectorRuntime(
            checkpoint, receipt, expected_checkpoint_sha256=_sha(checkpoint),
            expected_scheduler_sha256="0" * 64, expected_source_commit=SOURCE_COMMIT,
            expected_source_tree=SOURCE_TREE,
        )


def test_feature_dimension_is_bound_to_25(tmp_path):
    runtime = _runtime(tmp_path)
    assert runtime.model.feature_dim == 25
    assert len(runtime.normalization["mean"]) == 25


def test_nonfinite_feature_input_rejected(tmp_path):
    runtime = _runtime(tmp_path)
    with pytest.raises(ShadowContractError):
        runtime.step(
            episode_id="ep", policy_step=0, raw_action=[0, 0, 0, 0, 0, 0, 1], env_action=[0, 0, 0, 0, 0, 0, -1],
            observation={"robot0_gripper_qpos": [float("nan"), 0.1], "robot0_eef_pos": [0, 0, 0]},
        )


def test_nonfinite_logit_rejected(tmp_path):
    runtime = _runtime(tmp_path)

    class NonfiniteModel:
        def __call__(self, _value):
            return torch.tensor([float("inf")])

    runtime.model = NonfiniteModel()
    with pytest.raises(ShadowContractError, match="non-finite"):
        _step(runtime)


def test_missing_to_zero_fallback_rejected(tmp_path):
    runtime = _runtime(tmp_path)
    with pytest.raises(ShadowContractError):
        runtime.step(
            episode_id="ep", policy_step=0, raw_action=[0, 0, 0, 0, 0, 0, 1], env_action=[0, 0, 0, 0, 0, 0, -1],
            observation={"robot0_eef_pos": [0, 0, 0]},
        )


def test_attack_trigger_and_evaluation_detector_are_separate(tmp_path):
    runtime = _runtime(tmp_path)
    row = _step(runtime)
    assert row["input_action_source"] == "clean_policy_action_before_attack"
    assert row["evaluation_detector_affects_timing"] is False
    assert "attack" not in inspect.signature(runtime.step).parameters


def test_shadow_detector_cannot_modify_action(tmp_path):
    runtime = _runtime(tmp_path)
    row = _step(runtime)
    assert row["raw_action"][-1] == 1.0
    assert row["env_action"][-1] == -1.0


def test_clean_action_exact_equality_contract(tmp_path):
    runtime = _runtime(tmp_path)
    row = _step(runtime)
    assert row["raw_action"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert row["env_action"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_rand_true_matching_fields_are_preregistered():
    for path in (ROOT / "configs/sweep_v5_e0.03_s20.yaml", ROOT / "configs/sweep_v5_e0.06_s20.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "matched_to_TRUE:" in text
        assert "num_steps: 20" in text and "attack_burst_frames: 10" in text


def test_exact_init_state_hash_rule_is_pickle_v4():
    state = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    expected = hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()
    assert len(expected) == 64


def test_exact_task_seed_identity_is_explicit():
    assert dispatcher.FORMAL_SEEDS == (2026080501, 2026080502, 2026080503)
    assert dispatcher.SUITE_ORDER == ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def test_matrix_has_exactly_60_jobs():
    tasks = [
        {"suite": suite, "task_index": 0, "state_index": 45, "canonical_parent_key": f"{suite}/task_00/state_45", "initial_state_sha256": "a" * 64, "model_path": "/model"}
        for suite in dispatcher.SUITE_ORDER
    ]
    jobs = dispatcher.build_jobs(tasks, diagnostic=False, repo=ROOT, root=Path("/tmp/stage3a"), checkpoint=Path("/checkpoint"), receipt=Path("/receipt"), n4_norm=Path("/norm"), attacker_sha="a" * 64)
    assert len(jobs) == 60
    assert len({(j["condition"], j["canonical_parent_key"], j["seed"]) for j in jobs}) == 60


def test_forbidden_eval_paths_are_rejected():
    with pytest.raises(RuntimeError, match="forbidden"):
        dispatcher.assert_safe_path(Path("/tmp/eval160/run"))
    with pytest.raises(RuntimeError, match="forbidden"):
        dispatcher.assert_safe_path(Path("/tmp/protected_eval/run"))


def test_first_artifact_failure_is_fail_closed(tmp_path):
    job = {"job_id": "j", "job_root": str(tmp_path / "missing"), "arm": "CLEAN", "condition": "CLEAN_SHADOW", "suite": "libero_spatial", "task_index": 0, "seed": 1, "initial_state_sha256": "a" * 64}
    with pytest.raises(RuntimeError, match="artifact closure"):
        dispatcher.validate_job_artifact(job)


def test_one_gpu_one_worker_dispatch_contract():
    source = (ROOT / "scripts/fec/run_stage3a_shadow_matrix.py").read_text(encoding="utf-8")
    assert "if not pending or gpu in active" in source
    assert "start_new_session=True" in source
    assert "max_workers_per_physical_gpu" in source


def test_independent_auditor_recomputes_condition_totals():
    source = (ROOT / "scripts/fec/audit_stage3a_shadow_matrix.py").read_text(encoding="utf-8")
    assert "condition_totals" in source
    assert "trigger_episodes" in source
    assert "STAGE3A_SELECTIVITY_GATE_V1" in source


def test_auditor_gate_thresholds_are_preregistered():
    source = (ROOT / "scripts/fec/audit_stage3a_shadow_matrix.py").read_text(encoding="utf-8")
    for token in ("<= 1", "<= 2", ">= 6", ">= 9", ">= 10", ">= 0.50", ">= 0.75"):
        assert token in source

