import pytest

from scripts.stageb.layer3_exact_branching_contract import (
    DEFAULT_REQUIRED_PILOT_CONDITIONS,
    LAYER3_BRANCH_CONDITIONS,
    BranchRunRecord,
    ExactActionPrefixReplayPayload,
    Layer3BranchingContractError,
    PrefixBranchSnapshot,
    PrefixReplayStep,
    sha256_jsonable,
    arm_preservation_telemetry,
    make_gripper_only_executed_action,
    validate_branch_records,
)


def _snapshot() -> PrefixBranchSnapshot:
    return PrefixBranchSnapshot(
        suite="libero_spatial",
        task_idx=1,
        state_id=20,
        eval_seed=0,
        emit_step=58,
        observation_sha256="a" * 64,
        sim_state_sha256="b" * 64,
        policy_rng_sha256="c" * 64,
        detector_state_sha256="d" * 64,
        feature_history_sha256="e" * 64,
        source_episode_relpath="libero_spatial/task_01/state_20",
    )


def _records(snapshot: PrefixBranchSnapshot) -> list[BranchRunRecord]:
    return [
        BranchRunRecord(
            condition=condition,
            prefix_snapshot_sha256=snapshot.snapshot_sha256,
            branch_source="EXACT_PREFIX_RESTORE",
            restored_sim_state_sha256=snapshot.sim_state_sha256,
            restored_observation_sha256=snapshot.observation_sha256,
            restored_policy_rng_sha256=snapshot.policy_rng_sha256,
            restored_detector_state_sha256=snapshot.detector_state_sha256,
            restored_feature_history_sha256=snapshot.feature_history_sha256,
            trigger_step=snapshot.emit_step,
            first_env_step=snapshot.emit_step,
        )
        for condition in LAYER3_BRANCH_CONDITIONS
    ]


def test_gripper_only_action_preserves_clean_arm_and_attacked_gripper():
    clean = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]
    attacked = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 1.0]

    executed = make_gripper_only_executed_action(clean, attacked)

    assert executed[:6] == clean[:6]
    assert executed[-1] == attacked[-1]
    row = arm_preservation_telemetry(
        step=58,
        condition="VIS",
        clean_action=clean,
        attacked_decoded_action=attacked,
        executed_action=executed,
    )
    assert row["arm_preservation_pass"] is True
    assert row["arm_max_abs_diff"] == 0.0
    assert row["attacked_decoded_arm_0"] == 9.0
    assert row["executed_arm_0"] == clean[0]
    assert row["clean_gripper"] == -1.0
    assert row["attacked_gripper"] == 1.0
    assert row["executed_gripper"] == 1.0
    assert row["executed_gripper_expected_source"] == "attacked"
    assert row["gripper_abs_diff"] == 0.0


def test_clean_replay_requires_clean_gripper():
    clean = [0, 0, 0, 0, 0, 0, -1]
    attacked = [0, 0, 0, 0, 0, 0, 1]
    row = arm_preservation_telemetry(
        step=58,
        condition="CLEAN_REPLAY",
        clean_action=clean,
        attacked_decoded_action=attacked,
        executed_action=clean,
    )
    assert row["executed_gripper_expected_source"] == "clean"
    with pytest.raises(Layer3BranchingContractError, match="clean gripper"):
        arm_preservation_telemetry(
            step=58,
            condition="CLEAN_REPLAY",
            clean_action=clean,
            attacked_decoded_action=attacked,
            executed_action=make_gripper_only_executed_action(clean, attacked),
        )


def test_attack_conditions_reject_clean_gripper_execution():
    clean = [0, 0, 0, 0, 0, 0, -1]
    attacked = [0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(Layer3BranchingContractError, match="attacked gripper"):
        arm_preservation_telemetry(
            step=58,
            condition="VIS",
            clean_action=clean,
            attacked_decoded_action=attacked,
            executed_action=clean,
        )


def test_actions_must_be_exact_7d_and_finite():
    clean = [0, 0, 0, 0, 0, 0, -1]
    attacked = [0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(Layer3BranchingContractError, match="exactly 7"):
        make_gripper_only_executed_action(clean + [99], attacked)
    bad = list(clean)
    bad[0] = float("nan")
    with pytest.raises(Layer3BranchingContractError, match="non-finite"):
        make_gripper_only_executed_action(bad, attacked)


def test_arm_preservation_rejects_runtime_arm_drift():
    clean = [0, 0, 0, 0, 0, 0, -1]
    attacked = [0, 0, 0, 0, 0, 0, 1]
    executed = [0, 0, 0, 1e-4, 0, 0, 1]

    with pytest.raises(Layer3BranchingContractError, match="executed arm differs"):
        arm_preservation_telemetry(
            step=1,
            condition="RAND",
            clean_action=clean,
            attacked_decoded_action=attacked,
            executed_action=executed,
        )


def test_validate_branch_records_requires_all_conditions_same_snapshot():
    snapshot = _snapshot()
    result = validate_branch_records(snapshot, _records(snapshot))

    assert result["exact_prefix_branching_pass"] is True
    assert result["condition_count"] == 4
    assert result["required_conditions"] == list(DEFAULT_REQUIRED_PILOT_CONDITIONS)
    assert result["conditions"] == sorted(LAYER3_BRANCH_CONDITIONS)
    assert result["snapshot_boundary"] == "PRE_ACTION_OBS_T_AFTER_STUDENT_EMIT_BEFORE_ENV_STEP_T"


def test_validate_branch_records_accepts_minimal_3_condition_pilot_bundle():
    snapshot = _snapshot()
    records = [r for r in _records(snapshot) if r.condition in DEFAULT_REQUIRED_PILOT_CONDITIONS]
    result = validate_branch_records(snapshot, records)
    assert result["condition_count"] == 3
    assert result["required_conditions"] == list(DEFAULT_REQUIRED_PILOT_CONDITIONS)


def test_branch_record_accepts_exact_action_prefix_replay_contract():
    snapshot = _snapshot()
    record = BranchRunRecord(
        condition="VIS",
        prefix_snapshot_sha256=snapshot.snapshot_sha256,
        branch_source="EXACT_ACTION_PREFIX_REPLAY",
        restored_sim_state_sha256=snapshot.sim_state_sha256,
        restored_observation_sha256=snapshot.observation_sha256,
        restored_policy_rng_sha256=snapshot.policy_rng_sha256,
        restored_detector_state_sha256=snapshot.detector_state_sha256,
        restored_feature_history_sha256=snapshot.feature_history_sha256,
        trigger_step=snapshot.emit_step,
        first_env_step=snapshot.emit_step,
        prefix_trace_sha256="1" * 64,
        init_state_sha256="2" * 64,
        dummy_wait_contract_sha256="3" * 64,
        prefix_step_count=snapshot.emit_step,
        last_prefix_step=snapshot.emit_step - 1,
        pre_branch_sim_state_sha256="4" * 64,
        pre_branch_student_state_sha256="5" * 64,
        pre_branch_feature_history_sha256="6" * 64,
    )

    assert record.branch_source == "EXACT_ACTION_PREFIX_REPLAY"


def test_exact_action_prefix_replay_requires_prefix_provenance():
    snapshot = _snapshot()
    with pytest.raises(Layer3BranchingContractError, match="prefix_trace_sha256"):
        BranchRunRecord(
            condition="VIS",
            prefix_snapshot_sha256=snapshot.snapshot_sha256,
            branch_source="EXACT_ACTION_PREFIX_REPLAY",
            restored_sim_state_sha256=snapshot.sim_state_sha256,
            restored_observation_sha256=snapshot.observation_sha256,
            restored_policy_rng_sha256=snapshot.policy_rng_sha256,
            restored_detector_state_sha256=snapshot.detector_state_sha256,
            restored_feature_history_sha256=snapshot.feature_history_sha256,
            trigger_step=snapshot.emit_step,
            first_env_step=snapshot.emit_step,
        )


def _prefix_step(step: int = 0, *, token_offset: int = 0) -> PrefixReplayStep:
    tokens = tuple([31000 + token_offset, 31001, 31002, 31003, 31004, 31005, 31744])
    return PrefixReplayStep(
        step=step,
        raw_action_sha256="a" * 64,
        env_action_sha256="b" * 64,
        tokens=tokens,
        tokens_sha256=sha256_jsonable(list(tokens)),
        observation_sha256="c" * 64,
        policy_input_sha256="d" * 64,
        qpos_sha256="e" * 64,
        qvel_sha256="f" * 64,
        flat_sim_state_sha256="1" * 64,
        student_state_sha256="2" * 64,
        feature_history_sha256="3" * 64,
        reward=0.0,
        done=False,
    )


def _prefix_payload(**overrides) -> ExactActionPrefixReplayPayload:
    steps = tuple(_prefix_step(i, token_offset=i) for i in range(2))
    data = dict(
        protocol_version="C3_EXACT_ACTION_PREFIX_REPLAY_V1",
        parent_key="libero_goal|4|1|0|CLEAN",
        init_state_sha256="4" * 64,
        dummy_wait_contract_sha256="5" * 64,
        prefix_steps=steps,
        prefix_step_count=2,
        last_prefix_step=1,
        branch_step=2,
        prefix_trace_sha256="6" * 64,
        expected_branch_observation_sha256="7" * 64,
        expected_branch_policy_input_sha256="8" * 64,
        expected_branch_student_state_sha256="9" * 64,
        expected_branch_feature_history_sha256="a" * 64,
        expected_pre_branch_qpos_sha256="b" * 64,
        expected_pre_branch_qvel_sha256="c" * 64,
        expected_pre_branch_flat_sim_state_sha256="d" * 64,
    )
    data.update(overrides)
    return ExactActionPrefixReplayPayload(**data)


def test_exact_action_prefix_replay_payload_accepts_complete_provenance():
    payload = _prefix_payload()
    assert payload.prefix_step_count == 2
    assert payload.last_prefix_step == payload.branch_step - 1
    assert len(payload.payload_sha256) == 64


def test_exact_action_prefix_replay_payload_rejects_missing_prefix_trace_sha():
    with pytest.raises(Layer3BranchingContractError, match="prefix_trace_sha256"):
        _prefix_payload(prefix_trace_sha256="")


def test_exact_action_prefix_replay_payload_rejects_wrong_step_count():
    with pytest.raises(Layer3BranchingContractError, match="prefix_step_count"):
        _prefix_payload(prefix_step_count=3)


def test_exact_action_prefix_replay_payload_rejects_wrong_last_prefix_step():
    with pytest.raises(Layer3BranchingContractError, match="last_prefix_step"):
        _prefix_payload(last_prefix_step=0)


def test_exact_action_prefix_replay_payload_rejects_malformed_sha():
    with pytest.raises(Layer3BranchingContractError, match="init_state_sha256"):
        _prefix_payload(init_state_sha256="abc123")


def test_prefix_replay_step_rejects_mutated_token_sha():
    with pytest.raises(Layer3BranchingContractError, match="tokens_sha256"):
        PrefixReplayStep(
            step=0,
            raw_action_sha256="a" * 64,
            env_action_sha256="b" * 64,
            tokens=(1, 2, 3, 4, 5, 6, 7),
            tokens_sha256="c" * 64,
            observation_sha256="d" * 64,
            policy_input_sha256="e" * 64,
            qpos_sha256="f" * 64,
            qvel_sha256="1" * 64,
            flat_sim_state_sha256="2" * 64,
            student_state_sha256="3" * 64,
            feature_history_sha256="4" * 64,
            reward=0.0,
            done=False,
        )


def test_validate_branch_records_rejects_independent_restart():
    snapshot = _snapshot()
    records = _records(snapshot)
    bad = records[1]

    with pytest.raises(Layer3BranchingContractError, match="branch_source"):
        BranchRunRecord(
            condition=bad.condition,
            prefix_snapshot_sha256=bad.prefix_snapshot_sha256,
            branch_source="INDEPENDENT_RESTART",
            restored_sim_state_sha256=bad.restored_sim_state_sha256,
            restored_observation_sha256=bad.restored_observation_sha256,
            restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
            restored_detector_state_sha256=bad.restored_detector_state_sha256,
            restored_feature_history_sha256=bad.restored_feature_history_sha256,
            trigger_step=bad.trigger_step,
            first_env_step=bad.first_env_step,
        )


def test_validate_branch_records_rejects_prefix_hash_mismatch():
    snapshot = _snapshot()
    records = _records(snapshot)
    bad = records[2]
    records[2] = BranchRunRecord(
        condition=bad.condition,
        prefix_snapshot_sha256="f" * 64,
        branch_source=bad.branch_source,
        restored_sim_state_sha256=bad.restored_sim_state_sha256,
        restored_observation_sha256=bad.restored_observation_sha256,
        restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
        restored_detector_state_sha256=bad.restored_detector_state_sha256,
        restored_feature_history_sha256=bad.restored_feature_history_sha256,
        trigger_step=bad.trigger_step,
        first_env_step=bad.first_env_step,
    )

    with pytest.raises(Layer3BranchingContractError, match="frozen prefix"):
        validate_branch_records(snapshot, records)


def test_validate_branch_records_rejects_missing_condition():
    snapshot = _snapshot()
    records = [r for r in _records(snapshot) if r.condition != "RAND"]

    with pytest.raises(Layer3BranchingContractError, match="missing branch conditions"):
        validate_branch_records(snapshot, records)


def test_validate_branch_records_rejects_detector_and_feature_restore_mismatch():
    snapshot = _snapshot()
    records = _records(snapshot)
    bad = records[1]
    records[1] = BranchRunRecord(
        condition=bad.condition,
        prefix_snapshot_sha256=bad.prefix_snapshot_sha256,
        branch_source=bad.branch_source,
        restored_sim_state_sha256=bad.restored_sim_state_sha256,
        restored_observation_sha256=bad.restored_observation_sha256,
        restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
        restored_detector_state_sha256="0" * 64,
        restored_feature_history_sha256=bad.restored_feature_history_sha256,
        trigger_step=bad.trigger_step,
        first_env_step=bad.first_env_step,
    )
    with pytest.raises(Layer3BranchingContractError, match="detector state"):
        validate_branch_records(snapshot, records)

    records = _records(snapshot)
    bad = records[2]
    records[2] = BranchRunRecord(
        condition=bad.condition,
        prefix_snapshot_sha256=bad.prefix_snapshot_sha256,
        branch_source=bad.branch_source,
        restored_sim_state_sha256=bad.restored_sim_state_sha256,
        restored_observation_sha256=bad.restored_observation_sha256,
        restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
        restored_detector_state_sha256=bad.restored_detector_state_sha256,
        restored_feature_history_sha256="1" * 64,
        trigger_step=bad.trigger_step,
        first_env_step=bad.first_env_step,
    )
    with pytest.raises(Layer3BranchingContractError, match="feature history"):
        validate_branch_records(snapshot, records)


def test_validate_branch_records_rejects_first_env_step_off_by_one():
    snapshot = _snapshot()
    records = _records(snapshot)
    bad = records[0]
    records[0] = BranchRunRecord(
        condition=bad.condition,
        prefix_snapshot_sha256=bad.prefix_snapshot_sha256,
        branch_source=bad.branch_source,
        restored_sim_state_sha256=bad.restored_sim_state_sha256,
        restored_observation_sha256=bad.restored_observation_sha256,
        restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
        restored_detector_state_sha256=bad.restored_detector_state_sha256,
        restored_feature_history_sha256=bad.restored_feature_history_sha256,
        trigger_step=bad.trigger_step,
        first_env_step=bad.first_env_step + 1,
    )
    with pytest.raises(Layer3BranchingContractError, match="first_env_step"):
        validate_branch_records(snapshot, records)


def test_prefix_snapshot_requires_nonnegative_emit_step_and_hashes():
    with pytest.raises(Layer3BranchingContractError, match="emit_step"):
        PrefixBranchSnapshot(
            suite="libero_goal",
            task_idx=0,
            state_id=20,
            eval_seed=0,
            emit_step=-1,
            observation_sha256="a" * 64,
            sim_state_sha256="b" * 64,
            policy_rng_sha256="c" * 64,
            detector_state_sha256="d" * 64,
            feature_history_sha256="e" * 64,
            source_episode_relpath="x",
        )
