import json
import subprocess
import sys

import pytest

from scripts.stageb.layer3_exact_restore_runner import (
    ExactRestoreError,
    Layer3ParentDependencyManifest,
    build_mock_restore_case,
    compare_step_sequences,
    validate_clean_restore_pair,
)


def test_parent_manifest_requires_sha_dependencies():
    with pytest.raises(Exception, match="SHA256"):
        Layer3ParentDependencyManifest(
            suite="libero_spatial",
            task_idx=0,
            state_id=20,
            eval_seed=0,
            parent_key="x",
            openvla_model_sha256="not-a-sha",
            unnorm_key="libero_spatial",
            layer2_dataset_sha256="b" * 64,
            detector_checkpoint_sha256="c" * 64,
            tau_corridor=0.3,
            tau_release=0.3,
            libero_version="mock",
            mujoco_version="mock",
            task_instruction_sha256="d" * 64,
        )


def test_mock_restore_case_passes_five_step_reference_and_replays():
    case = build_mock_restore_case()
    result = validate_clean_restore_pair(
        snapshot=case["snapshot"],
        branch_records=case["branch_records"],
        reference=case["reference"],
        replay_a=case["replay_a"],
        replay_b=case["replay_b"],
    )
    assert result["clean_restore_pass"] is True
    assert result["restore_steps"] == 5
    assert result["reference_vs_replay_mismatch_count"] == 0
    assert result["replay_a_vs_replay_b_mismatch_count"] == 0


def test_restore_comparison_rejects_observation_mismatch():
    case = build_mock_restore_case()
    replay = list(case["replay_a"])
    replay[2].observation_sha256 = "0" * 64
    problems = compare_step_sequences(case["reference"], replay)
    assert "step2:observation_sha256_mismatch" in problems
    with pytest.raises(ExactRestoreError, match="reference_vs_replay"):
        validate_clean_restore_pair(
            snapshot=case["snapshot"],
            branch_records=case["branch_records"],
            reference=case["reference"],
            replay_a=replay,
            replay_b=case["replay_b"],
        )


def test_restore_comparison_rejects_detector_state_mismatch():
    case = build_mock_restore_case()
    replay = list(case["replay_b"])
    replay[1].detector_state_sha256 = "1" * 64
    problems = compare_step_sequences(case["replay_a"], replay)
    assert "step1:detector_state_sha256_mismatch" in problems


def test_mock_cli_writes_result(tmp_path):
    out = tmp_path / "mock_restore"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/stageb/layer3_exact_restore_runner.py",
            "--mock",
            "--output-dir",
            str(out),
        ]
    )
    result = json.loads((out / "mock_restore_result.json").read_text())
    assert result["clean_restore_pass"] is True
    assert result["restore_steps"] == 5

