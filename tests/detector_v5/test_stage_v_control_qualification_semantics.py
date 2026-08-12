from scripts.detector_v5.run_stage_v_control_qualification import (
    audit_qualification_row,
    qualifies,
)


SOURCE_COMMIT = "commit"
SOURCE_TREE = "tree"


def _result(key: str, *, terminal_outcome: str, terminal_state: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "exit_code": 0,
        "clean_success": True,
        "task_identity_valid": True,
        "snapshot_restore_valid": True,
        "runtime_valid": True,
        "metrics_finite": True,
        "artifact_validation_pass": True,
        "old_artifacts_reused": False,
        "remaining_horizon_complete": True,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "canonical_parent_key": key,
        "worker_gpu": 0,
        "assigned_gpu": 0,
        "key_state_identity_sha256": "same-initial-state",
        "terminal_outcome": terminal_outcome,
        "terminal_state_sha256": terminal_state,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
    }


def test_terminal_fields_are_descriptive_not_qualification_gates() -> None:
    key = "libero_goal/task_00/state_39"
    row = {"canonical_parent_key": key, "assigned_gpu": 0}
    a = _result(key, terminal_outcome="SUCCESS", terminal_state="terminal-a")
    b = _result(key, terminal_outcome="TASK_FAILURE", terminal_state="terminal-b")

    assert qualifies(row, a, b, SOURCE_COMMIT, SOURCE_TREE) == (True, [])
    assert audit_qualification_row(row, a, b, SOURCE_COMMIT, SOURCE_TREE) == (True, [])
