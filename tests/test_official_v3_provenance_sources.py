from pathlib import Path

import pytest

from gripper_attack.official_v3_provenance_sources import (
    ProvenanceSourceViolation,
    normalize_final_ledger_rows,
    write_normalized_completion_bundle,
)


def artifact(key="libero_object/task_00/state_00"):
    return {
        "canonical_parent_key": key,
        "artifact_root": "/clean/a",
        "artifact_recursive_sha256": "a" * 64,
    }


def ledger(key="libero_object/task_00/state_00", status="PASS"):
    return {
        "canonical_parent_key": key,
        "suite": "libero_object",
        "task_idx": "0",
        "state_id": "0",
        "status": status,
        "result_status": status,
        "worker_start_uuid": "s1",
        "worker_start_manifest_sha256": "m" * 64,
        "artifact_root": "/clean/a",
    }


def test_final_ledger_normalizes_success_and_failure_without_filtering():
    rows, summary = normalize_final_ledger_rows(
        [ledger(), ledger(key="libero_object/task_00/state_01", status="TASK_FAILURE")],
        [artifact(), artifact(key="libero_object/task_00/state_01")],
        ledger_source_path="ledger.csv", ledger_source_sha256="l" * 64,
    )
    assert [row["task_success"] for row in rows] == ["true", "false"]
    assert all(row["selected_result"] == "true" for row in rows)
    assert summary["direct_start_uuid_row_count"] == 2


def test_missing_or_duplicate_identity_fails_closed():
    with pytest.raises(ProvenanceSourceViolation):
        normalize_final_ledger_rows([ledger(), ledger()], [artifact()])
    with pytest.raises(ProvenanceSourceViolation):
        normalize_final_ledger_rows([ledger()], [artifact(key="libero_object/task_00/state_01")])


def test_normalized_bundle_is_non_overwrite(tmp_path: Path):
    rows, summary = normalize_final_ledger_rows([ledger()], [artifact()])
    output = tmp_path / "sources"
    write_normalized_completion_bundle(rows, summary, output)
    with pytest.raises(ProvenanceSourceViolation):
        write_normalized_completion_bundle(rows, summary, output)
