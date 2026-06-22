import json
import random
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from scripts.stageb.layer3_exact_restore_runner import (
    EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE,
    EXPECTED_LAYER2_DATASET_SHA256,
    ExactRestoreSnapshotPayload,
    ExactRestoreError,
    Layer3RuntimeReceipt,
    Layer3ParentDependencyManifest,
    _MockEnv,
    _MockPolicy,
    _MockStudent,
    build_mock_restore_case,
    capture_env_internal_state,
    capture_feature_history,
    capture_mujoco_state,
    capture_policy_rng_state,
    capture_student_state,
    model_norm_stat_keys,
    parse_cuda_visible_devices,
    query_ordered_visible_gpu_uuids,
    get_observation_after_restore,
    compare_step_sequences,
    restore_env_internal_state,
    restore_feature_history,
    restore_mujoco_state,
    restore_policy_rng_state,
    restore_snapshot,
    restore_snapshot_and_recapture_observation,
    restore_student_state,
    rollout_clean_steps,
    update_student_for_step,
    validate_clean_restore_pair,
    validate_dependency_sha_values,
    validate_real_openvla_model_binding,
)


def make_parent(**overrides):
    data = dict(
        suite="libero_spatial",
        task_idx=0,
        state_id=20,
        eval_seed=0,
        parent_key="libero_spatial|0|20|0|CLEAN",
        openvla_model_sha256="a" * 64,
        unnorm_key="libero_spatial",
        layer2_dataset_sha256=EXPECTED_LAYER2_DATASET_SHA256,
        detector_checkpoint_sha256=EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE["libero_spatial"],
        tau_corridor=0.3,
        tau_release=0.3,
        libero_version="mock",
        mujoco_version="mock",
        task_instruction_sha256="d" * 64,
    )
    data.update(overrides)
    return Layer3ParentDependencyManifest(**data)


def test_parent_manifest_requires_sha_dependencies():
    with pytest.raises(Exception, match="SHA256"):
        make_parent(
            openvla_model_sha256="not-a-sha",
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"suite": "libero_object", "parent_key": "libero_object|0|20|0|CLEAN"}, "unsupported suite"),
        ({"layer2_dataset_sha256": "b" * 64}, "frozen v3 dataset"),
        ({"detector_checkpoint_sha256": "c" * 64}, "frozen libero_spatial M2 checkpoint"),
        ({"unnorm_key": "wrong"}, "unnorm_key"),
        ({"parent_key": "libero_spatial|1|20|0|CLEAN"}, "parent_key"),
        ({"tau_corridor": float("nan")}, "tau_corridor"),
        ({"tau_release": 1.1}, "tau_release"),
        ({"libero_version": ""}, "libero_version"),
        ({"mujoco_version": ""}, "mujoco_version"),
    ],
)
def test_parent_manifest_strict_validation(overrides, match):
    with pytest.raises(ExactRestoreError, match=match):
        make_parent(**overrides)


def test_real_openvla_binding_rejects_non_suite_matched_model_dir(tmp_path):
    model_dir = tmp_path / "openvla-7b-finetuned-libero-object"
    model_dir.mkdir()
    (model_dir / "dataset_statistics.json").write_text('{"libero_object": {}}\n', encoding="utf-8")

    with pytest.raises(ExactRestoreError, match="suite-matched"):
        validate_real_openvla_model_binding(
            suite="libero_goal",
            model_path=model_dir,
            unnorm_key="libero_goal",
        )


def test_real_openvla_binding_rejects_missing_unnorm_key(tmp_path):
    model_dir = tmp_path / "openvla-7b-finetuned-libero-goal"
    model_dir.mkdir()
    (model_dir / "dataset_statistics.json").write_text('{"libero_object": {}}\n', encoding="utf-8")

    assert model_norm_stat_keys(model_dir) == ["libero_object"]
    with pytest.raises(ExactRestoreError, match="unnorm_key libero_goal unavailable"):
        validate_real_openvla_model_binding(
            suite="libero_goal",
            model_path=model_dir,
            unnorm_key="libero_goal",
        )


def test_real_openvla_binding_accepts_suite_matched_stats(tmp_path):
    model_dir = tmp_path / "openvla-7b-finetuned-libero-goal"
    model_dir.mkdir()
    (model_dir / "dataset_statistics.json").write_text('{"libero_goal": {}}\n', encoding="utf-8")

    receipt = validate_real_openvla_model_binding(
        suite="libero_goal",
        model_path=model_dir,
        unnorm_key="libero_goal",
    )
    assert receipt["expected_model_dir"] == "openvla-7b-finetuned-libero-goal"
    assert receipt["available_norm_keys"] == ["libero_goal"]


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


def test_restore_comparison_rejects_step_offset_and_reports_numeric_diff():
    case = build_mock_restore_case()
    replay = list(case["replay_a"])
    replay[0] = replace(replay[0], step=replay[0].step + 1, qpos_values=[999.0, 999.0])
    problems = compare_step_sequences(case["reference"], replay)
    assert "step0:step_mismatch" in problems
    assert any(item.startswith("step0:qpos_max_abs_diff=") for item in problems)


def test_restore_comparison_rejects_detector_state_mismatch():
    case = build_mock_restore_case()
    replay = list(case["replay_b"])
    replay[1].detector_state_sha256 = "1" * 64
    problems = compare_step_sequences(case["replay_a"], replay)
    assert "step1:detector_state_sha256_mismatch" in problems


def test_global_and_policy_rng_round_trip():
    policy = _MockPolicy()
    random.seed(123)
    np.random.seed(123)
    saved = capture_policy_rng_state(policy)
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    policy.set_rng_state({"counter": 99})
    random.seed(999)
    np.random.seed(999)
    restore_policy_rng_state(policy, saved)
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert policy.rng_state() == {"counter": 0}


def test_feature_history_round_trip():
    student = _MockStudent()
    saved = capture_feature_history(student)
    student.feature_history.append({"step": 999})
    restore_feature_history(student, saved)
    assert student.snapshot_feature_history() == saved


def test_student_update_changes_state_and_feature_history_during_rollout():
    case = build_mock_restore_case()
    rows = case["reference"]
    assert len({row.detector_state_sha256 for row in rows}) == 5
    assert len({row.feature_history_sha256 for row in rows}) == 5


def test_missing_env_student_and_feature_hooks_fail_closed():
    class NoEnvHooks:
        pass

    class NoStudentHooks:
        pass

    with pytest.raises(ExactRestoreError, match="get_internal_state"):
        capture_env_internal_state(NoEnvHooks())
    with pytest.raises(ExactRestoreError, match="set_internal_state"):
        restore_env_internal_state(NoEnvHooks(), {})
    with pytest.raises(ExactRestoreError, match="snapshot_state"):
        capture_student_state(NoStudentHooks())
    with pytest.raises(ExactRestoreError, match="restore_state"):
        restore_student_state(NoStudentHooks(), {})
    with pytest.raises(ExactRestoreError, match="snapshot_feature_history"):
        capture_feature_history(NoStudentHooks())
    with pytest.raises(ExactRestoreError, match="restore_feature_history"):
        restore_feature_history(NoStudentHooks(), [])
    with pytest.raises(ExactRestoreError, match="per-step update"):
        update_student_for_step(NoStudentHooks(), step=0, obs={}, action=[0] * 7, tokens=[0] * 7)


def test_restore_observation_must_be_recaptured_from_env():
    case = build_mock_restore_case()
    snapshot = case["snapshot"]
    env = _MockEnv()
    student = _MockStudent()
    policy = _MockPolicy()
    restored_obs = restore_snapshot_and_recapture_observation(env, student, snapshot, policy)
    assert restored_obs == snapshot.observation
    env.internal["step"] = 999
    with pytest.raises(ExactRestoreError, match="restored observation hash"):
        get_observation_after_restore(env, snapshot)


def test_missing_restored_observation_hook_fail_closed():
    class NoObservationEnv(_MockEnv):
        def get_observation_after_restore(self):  # type: ignore[no-untyped-def]
            raise AttributeError("not available")

    class TrulyNoObservationEnv:
        def __init__(self):
            self.sim = _MockEnv().sim

    case = build_mock_restore_case()
    with pytest.raises(ExactRestoreError, match="get_observation_after_restore"):
        get_observation_after_restore(TrulyNoObservationEnv(), case["snapshot"])


def test_payload_prefix_hash_inconsistency_rejected():
    case = build_mock_restore_case()
    snapshot = case["snapshot"]
    with pytest.raises(ExactRestoreError, match="feature_history hash"):
        ExactRestoreSnapshotPayload(
            prefix=snapshot.prefix,
            parent_manifest=snapshot.parent_manifest,
            mujoco_state=snapshot.mujoco_state,
            env_internal_state=snapshot.env_internal_state,
            policy_rng_state=snapshot.policy_rng_state,
            student_state=snapshot.student_state,
            feature_history=[{"step": "tampered"}],
            observation=snapshot.observation,
            clean_action_t=snapshot.clean_action_t,
            clean_tokens_t=snapshot.clean_tokens_t,
        )


def test_recaptured_branch_record_uses_actual_post_restore_state():
    case = build_mock_restore_case()
    snapshot = case["snapshot"]
    policy = _MockPolicy()
    env = _MockEnv()
    student = _MockStudent()
    restore_snapshot(env, student, snapshot, policy)
    env.sim.data.qvel[:] = 42.0
    from scripts.stageb.layer3_exact_restore_runner import recapture_branch_record
    from scripts.stageb.layer3_exact_branching_contract import validate_branch_records

    record = recapture_branch_record(
        condition="CLEAN_REPLAY",
        snapshot=snapshot,
        env=env,
        student=student,
        policy=policy,
    )
    with pytest.raises(Exception, match="sim_state hash mismatch"):
        validate_branch_records(snapshot.prefix, [record], required_conditions=("CLEAN_REPLAY",))


def test_first_action_and_token_mismatch_rejected():
    case = build_mock_restore_case()
    snapshot = case["snapshot"]
    env = _MockEnv()
    student = _MockStudent()
    policy = _MockPolicy()
    restore_snapshot(env, student, snapshot, policy)
    with pytest.raises(ExactRestoreError, match="first token mismatch"):
        rollout_clean_steps(
            env=env,
            student=student,
            policy=policy,
            initial_obs=snapshot.observation,
            start_step=58,
            expected_first_action=snapshot.clean_action_t,
            expected_first_tokens=[1, 2, 3, 4, 5, 6, 7],
        )


def test_early_done_rejected():
    class EarlyDoneEnv(_MockEnv):
        def step(self, action):
            obs, reward, _done, info = super().step(action)
            return obs, reward, True, info

    case = build_mock_restore_case()
    snapshot = case["snapshot"]
    env = EarlyDoneEnv()
    student = _MockStudent()
    policy = _MockPolicy()
    restore_snapshot(env, student, snapshot, policy)
    with pytest.raises(ExactRestoreError, match="ended before"):
        rollout_clean_steps(
            env=env,
            student=student,
            policy=policy,
            initial_obs=snapshot.observation,
            start_step=58,
            expected_first_action=snapshot.clean_action_t,
            expected_first_tokens=snapshot.clean_tokens_t,
        )


def test_incomplete_restore_sequence_rejected():
    case = build_mock_restore_case()
    with pytest.raises(ExactRestoreError, match="exactly 5"):
        validate_clean_restore_pair(
            snapshot=case["snapshot"],
            branch_records=case["branch_records"],
            reference=case["reference"][:4],
            replay_a=case["replay_a"],
            replay_b=case["replay_b"],
        )


def test_mujoco_warmstart_and_applied_force_state_restored():
    env = _MockEnv()
    env.sim.data.qacc_warmstart[:] = [1.0, 2.0]
    env.sim.data.qfrc_applied[:] = [3.0, 4.0]
    env.sim.data.xfrc_applied[:] = 5.0
    env.sim.data.eq_active[:] = [0]
    state = capture_mujoco_state(env)
    env.sim.data.qacc_warmstart[:] = 0.0
    env.sim.data.qfrc_applied[:] = 0.0
    env.sim.data.xfrc_applied[:] = 0.0
    env.sim.data.eq_active[:] = [1]
    restore_mujoco_state(env, state)
    assert env.sim.data.qacc_warmstart.tolist() == [1.0, 2.0]
    assert env.sim.data.qfrc_applied.tolist() == [3.0, 4.0]
    assert float(env.sim.data.xfrc_applied[0, 0]) == 5.0
    assert env.sim.data.eq_active.tolist() == [0]


def test_dependency_sha_value_validation():
    parent = make_parent(openvla_model_sha256="a" * 64)
    result = validate_dependency_sha_values(
        parent,
        actual_openvla_model_sha256="a" * 64,
        actual_detector_checkpoint_sha256=EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE["libero_spatial"],
    )
    assert result["dependency_sha_validation_pass"] is True
    with pytest.raises(ExactRestoreError, match="OpenVLA model SHA"):
        validate_dependency_sha_values(
            parent,
            actual_openvla_model_sha256="b" * 64,
            actual_detector_checkpoint_sha256=EXPECTED_M2_CHECKPOINT_SHA256_BY_SUITE["libero_spatial"],
        )


def test_runtime_receipt_requires_deterministic_gpu_order_and_generation():
    receipt = Layer3RuntimeReceipt(
        cuda_visible_devices="1,3",
        ordered_gpu_uuids=["GPU-a", "GPU-b"],
        device_count=2,
        torch_version="mock",
        cuda_runtime="mock",
        driver_version="mock",
        libero_version="mock",
        mujoco_version="mock",
        openvla_generation_kwargs={"do_sample": False, "temperature": 0.0},
    )
    assert receipt.receipt_sha256
    with pytest.raises(ExactRestoreError, match="device_count"):
        Layer3RuntimeReceipt(
            cuda_visible_devices="1,3",
            ordered_gpu_uuids=["GPU-a"],
            device_count=2,
            torch_version="mock",
            cuda_runtime="mock",
            driver_version="mock",
            libero_version="mock",
            mujoco_version="mock",
            openvla_generation_kwargs={"do_sample": False, "temperature": 0.0},
        )


def test_ordered_gpu_uuid_query_uses_cuda_visible_devices_order(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        token = cmd[2]

        class Result:
            stdout = f"GPU-{token}\n"

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert parse_cuda_visible_devices("5,1") == ["5", "1"]
    assert query_ordered_visible_gpu_uuids("5,1") == ["GPU-5", "GPU-1"]
    assert calls[0][:3] == ["nvidia-smi", "-i", "5"]
    assert calls[1][:3] == ["nvidia-smi", "-i", "1"]


def test_ordered_gpu_uuid_query_rejects_missing_visible_devices():
    with pytest.raises(ExactRestoreError, match="CUDA_VISIBLE_DEVICES"):
        query_ordered_visible_gpu_uuids("")
    with pytest.raises(ExactRestoreError, match="do_sample"):
        Layer3RuntimeReceipt(
            cuda_visible_devices="1",
            ordered_gpu_uuids=["GPU-a"],
            device_count=1,
            torch_version="mock",
            cuda_runtime="mock",
            driver_version="mock",
            libero_version="mock",
            mujoco_version="mock",
            openvla_generation_kwargs={"do_sample": True, "temperature": 0.0},
        )


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
