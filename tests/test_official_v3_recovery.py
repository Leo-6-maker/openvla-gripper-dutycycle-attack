from pathlib import Path

import pytest

from gripper_attack.official_v3_recovery import (
    AMBIGUOUS,
    CONTRADICTORY,
    EXACT_DIRECT,
    EXACT_LEASE,
    MISSING,
    WEAK,
    RecoveryContractViolation,
    build_recovery_rows,
    write_recovery_bundle,
)


def artifact(key="libero_object/task_00/state_00"):
    return {"canonical_parent_key": key, "artifact_root": "/clean/a", "artifact_recursive_sha256": "a" * 64, "worker_id": "w", "pid": "11", "collector_head": "c" * 40}


def worker(start="s1"):
    return {"start_uuid": start, "worker_id": "w", "pid": "11", "collector_head": "c" * 40, "manifest_sha256": "m" * 64, "manifest_sidecar_sha256": "n" * 64, "worker_start_gate_ack_sha256": "g" * 64}


def completion(**kwargs):
    value = {"canonical_parent_key": "libero_object/task_00/state_00", "selected_result": True, "quarantined": False, "artifact_recursive_sha256": "a" * 64, "completion_record_sha256": "z" * 64}
    value.update(kwargs)
    return value


def test_direct_start_chain_is_formal():
    rows, summary = build_recovery_rows([artifact()], [worker()], [], [completion(start_uuid="s1")])
    assert rows[0]["recovery_status"] == EXACT_DIRECT
    assert rows[0]["formal_eligible"] is True
    assert summary["formal_fit_exact_count"] == 1


def test_exact_lease_chain_is_formal():
    lease = {"canonical_parent_key": "libero_object/task_00/state_00", "lease_uuid": "l1", "lease_epoch": "2", "fencing_token": "f1", "start_uuid": "s1", "assignment_record_sha256": "q" * 64}
    rows, _ = build_recovery_rows([artifact()], [worker()], [lease], [completion(lease_uuid="l1", lease_epoch="2", fencing_token="f1")])
    assert rows[0]["recovery_status"] == EXACT_LEASE
    assert rows[0]["formal_eligible"] is True


def test_weak_pid_match_never_promotes():
    rows, _ = build_recovery_rows([artifact()], [worker()], [], [])
    assert rows[0]["recovery_status"] == WEAK
    assert rows[0]["formal_eligible"] is False


def test_missing_ambiguous_and_contradictory_are_fail_closed():
    missing_worker = worker("other")
    missing_worker["pid"] = "22"
    missing_worker["collector_head"] = "d" * 40
    missing, _ = build_recovery_rows([artifact()], [missing_worker], [], [])
    assert missing[0]["recovery_status"] == MISSING
    ambiguous, _ = build_recovery_rows([artifact()], [worker("s1"), worker("s2")], [], [])
    assert ambiguous[0]["recovery_status"] == AMBIGUOUS
    contradictory, _ = build_recovery_rows([artifact()], [worker()], [], [completion(start_uuid="s1", artifact_recursive_sha256="b" * 64)])
    assert contradictory[0]["recovery_status"] == CONTRADICTORY


def test_bundle_is_non_overwrite(tmp_path: Path):
    rows, summary = build_recovery_rows([artifact()], [worker()], [], [])
    output = tmp_path / "recovery"
    write_recovery_bundle(rows, summary, output)
    with pytest.raises(RecoveryContractViolation):
        write_recovery_bundle(rows, summary, output)
