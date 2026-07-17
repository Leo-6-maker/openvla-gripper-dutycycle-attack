import json
from pathlib import Path

import pytest

from gripper_attack.official_v3_sprint0 import (
    BRIDGE_PASS,
    EXACT_REMEDIATION_REQUIRED,
    Sprint0ContractViolation,
    audit_legacy_bridge,
    audit_stale_lease_recovery,
    build_fit_remediation_queue,
)


def _baseline():
    fields = [
        "artifact_schema", "runtime_valid", "source_contract_pass", "no_teacher_attack_files",
        "official_action_adapter", "official_horizon", "num_steps_wait", "collector_head",
        "worker_script_sha256", "adapter_sha256", "protocol_sha256", "model_tree_sha256",
        "processor_tree_sha256", "feature_order_sha256", "action_postprocess_sha256", "env_init_sha256",
    ]
    values = {
        "artifact_schema": "OPENVLA_OFFICIAL_CLEAN_EPISODE_V2",
        "runtime_valid": True,
        "source_contract_pass": True,
        "no_teacher_attack_files": True,
        "official_action_adapter": "OfficialOpenVLAActionAdapter.predict_action",
        "official_horizon": 520,
        "num_steps_wait": 10,
        "collector_head": "a" * 40,
        "worker_script_sha256": "b" * 64,
        "adapter_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "model_tree_sha256": "e" * 64,
        "processor_tree_sha256": "f" * 64,
        "feature_order_sha256": "1" * 64,
        "action_postprocess_sha256": "2" * 64,
        "env_init_sha256": "3" * 64,
    }
    return {
        "schema": "OFFICIAL_V3_LEGACY_BRIDGE_BASELINE_V1",
        "status": "FROZEN_PROVENANCE_ONLY",
        "bridge_fields": fields,
        "expected_values": values,
    }


def _row(key="libero_10/task_00/state_00", evidence="LEGACY_METADATA_ONLY"):
    row = {"canonical_parent_key": key, **_baseline()["expected_values"]}
    row.update({
        "artifact_recursive_sha256": "9" * 64,
        "generation_evidence": evidence,
        "worker_start_manifest_present": False,
        "provenance_class": "C_START_RECORD_MISSING",
        "active": False,
    })
    return row


def test_legacy_metadata_bridge_is_provenance_only():
    report = audit_legacy_bridge([_row()], _baseline(), expected_keys=[_row()["canonical_parent_key"]])
    assert report["overall_status"] == BRIDGE_PASS
    record = report["records"][0]
    assert record["legacy_25d_pilot_eligible"] is True
    assert record["official_v3_formal_eligible"] is False
    assert report["teacher_labels_read"] is False
    assert report["attack_results_read"] is False


def test_bridge_mismatch_requires_exact_identity_remediation():
    row = _row()
    row["model_tree_sha256"] = "0" * 64
    report = audit_legacy_bridge([row], _baseline())
    assert report["overall_status"] == EXACT_REMEDIATION_REQUIRED
    assert report["records"][0]["remediation_required"] is True


def test_bridge_rejects_scientific_result_columns():
    row = _row()
    row["task_success"] = True
    with pytest.raises(Sprint0ContractViolation):
        audit_legacy_bridge([row], _baseline())


def test_remediation_queue_is_exact_identity_and_ignores_active_rows():
    row = _row("libero_10/task_00/state_00")
    manifest = [{
        "canonical_parent_key": row["canonical_parent_key"], "suite": "libero_10", "task_idx": "0",
        "state_id": "0", "split": "FIT_TRAIN", "queue_rank": "10",
    }]
    bridge = audit_legacy_bridge([row], _baseline())
    queue, summary = build_fit_remediation_queue(
        manifest, bridge, [], queue_epoch_id="V3_REMEDIATION_EPOCH_1"
    )
    assert len(queue) == 0
    assert summary["exact_identity_replacement_only"] is True

    row["model_tree_sha256"] = "0" * 64
    bridge = audit_legacy_bridge([row], _baseline())
    queue, _ = build_fit_remediation_queue(
        manifest, bridge, [], queue_epoch_id="V3_REMEDIATION_EPOCH_2"
    )
    assert len(queue) == 1
    assert queue[0]["canonical_parent_key"] == manifest[0]["canonical_parent_key"]
    assert queue[0]["replacement_identity_policy"] == "EXACT_SAME_CANONICAL_IDENTITY"
    with pytest.raises(Sprint0ContractViolation):
        build_fit_remediation_queue(
            manifest, bridge,
            [{"canonical_parent_key": row["canonical_parent_key"], "status": "RUNNING"}],
            queue_epoch_id="V3_REMEDIATION_EPOCH_3",
        )


def test_stale_lease_recovery_requires_epoch_rotation_and_quarantine():
    key = "libero_10/task_06/state_11"
    ledger = [{
        "canonical_parent_key": key, "status": "RUNNING", "pid": "123",
        "lease_timestamp": "1000", "lease_uuid": "old-uuid", "lease_epoch_id": "7",
    }]
    processes = [{"pid": "123", "alive": False}]
    formal = [{"canonical_parent_key": key, "formal_selected": True, "formal_result_sha256": "a" * 64}]
    recovery = [{
        "canonical_parent_key": key, "old_lease_uuid": "old-uuid", "new_lease_uuid": "new-uuid",
        "old_lease_epoch_id": "7", "new_lease_epoch_id": "8", "fencing_token": "fence-8",
        "late_result_policy": "QUARANTINE",
    }]
    report = audit_stale_lease_recovery(
        ledger, processes, formal, recovery, now_epoch=2000,
        expected_stale_keys=[key],
    )
    assert report["status"] == "RECOVERY_SAFE"
    assert report["ledger_mutated"] is False

    recovery[0]["new_lease_epoch_id"] = "7"
    report = audit_stale_lease_recovery(
        ledger, processes, formal, recovery, now_epoch=2000,
        expected_stale_keys=[key],
    )
    assert report["status"] == "HOLD"


def test_stale_lease_without_formal_result_is_hold():
    key = "libero_10/task_06/state_12"
    report = audit_stale_lease_recovery(
        [{
            "canonical_parent_key": key, "status": "RUNNING", "pid": "123",
            "lease_timestamp": "1000", "lease_uuid": "old-uuid", "lease_epoch_id": "7",
        }],
        [{"pid": "123", "alive": False}],
        [],
        [{
            "canonical_parent_key": key, "old_lease_uuid": "old-uuid", "new_lease_uuid": "new-uuid",
            "old_lease_epoch_id": "7", "new_lease_epoch_id": "8", "fencing_token": "fence-8",
            "late_result_policy": "QUARANTINE",
        }],
        now_epoch=2000,
        expected_stale_keys=[key],
    )
    assert report["status"] == "HOLD"
    assert report["missing_formal_result_keys"] == [key]
