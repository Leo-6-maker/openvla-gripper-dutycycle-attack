import pytest

from scripts.stageb.layer3_exact_branching_contract import (
    LAYER3_BRANCH_CONDITIONS,
    BranchRunRecord,
    Layer3BranchingContractError,
    PrefixBranchSnapshot,
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
    assert result["conditions"] == list(LAYER3_BRANCH_CONDITIONS)


def test_validate_branch_records_rejects_independent_restart():
    snapshot = _snapshot()
    records = _records(snapshot)
    bad = records[1]

    with pytest.raises(Layer3BranchingContractError, match="EXACT_PREFIX_RESTORE"):
        BranchRunRecord(
            condition=bad.condition,
            prefix_snapshot_sha256=bad.prefix_snapshot_sha256,
            branch_source="INDEPENDENT_RESTART",
            restored_sim_state_sha256=bad.restored_sim_state_sha256,
            restored_observation_sha256=bad.restored_observation_sha256,
            restored_policy_rng_sha256=bad.restored_policy_rng_sha256,
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
        trigger_step=bad.trigger_step,
        first_env_step=bad.first_env_step,
    )

    with pytest.raises(Layer3BranchingContractError, match="frozen prefix"):
        validate_branch_records(snapshot, records)


def test_validate_branch_records_rejects_missing_condition():
    snapshot = _snapshot()
    records = _records(snapshot)[:-1]

    with pytest.raises(Layer3BranchingContractError, match="missing branch conditions"):
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
