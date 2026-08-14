from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.stage_v_m4_q00_authority import (
    COUNTERS,
    PASS_STATUS,
    Q00AuthorityError,
    SCHEMA,
    sha256_file,
    validate_q00_authority,
)


REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def _authority(tmp_path: Path) -> dict:
    runtime_commit = "a" * 40
    runtime_tree = "b" * 40
    snapshot_commit = "c" * 40
    snapshot_tree = "d" * 40
    q00 = {
        "parent_key": "libero_10/task_01/state_42",
        "probe_id": "Q00",
        "probe_step": 46,
        "snapshot_manifest_sha256": "",
        "snapshot_source_commit": snapshot_commit,
        "snapshot_source_tree": snapshot_tree,
        "exact_plan_manifest_sha256": "",
        "clean_prefix_replay_allowed": True,
        "post_snapshot_primary_window_steps": 0,
    }
    snapshot = {
        "schema": "STAGE_V_CAUSAL_PROBE_SNAPSHOT_V2",
        "status": "SEALED_PROSPECTIVE_SNAPSHOT",
        "binding": {
            "parent_key": q00["parent_key"], "probe_id": q00["probe_id"],
            "source_commit": snapshot_commit, "source_tree": snapshot_tree, "step": 46,
        },
        "primary_input_authority": "loaded_frozen_canonical_bytes",
        "fresh_render_equality_gate_used": False,
        "payload": {
            "episode_start_rng_state": {}, "required_rng_state": {},
            "full_simulator_state": {}, "controller_and_wrapper_runtime_state": {},
            "clean_reference_action_window": [],
        },
    }
    snapshot_path = tmp_path / "CAUSAL_PROBE_SNAPSHOT_V2.json"
    q00["snapshot_manifest_sha256"] = _write(snapshot_path, snapshot)
    plan = {
        "schema": "STAGE_V_M4_EXACT_PROBE_AND_SNAPSHOT_MANIFEST_V1",
        "status": "PASS_EXACT_40X24_PLAN_ONLY",
        "selection_outcomes_read": False, "intervention_executed": False,
        "v_phys_generated": False, "teacher_predictions_read": False,
        "student_predictions_read": False, "protected_counters": COUNTERS,
        "probe_authorities": [
            {"canonical_parent_key": q00["parent_key"], "probe_id": "Q00", "probe_step": 46,
             "snapshot_manifest_sha256": q00["snapshot_manifest_sha256"], "arm": arm}
            for arm in ("CONTROL", "T3", "T5", "T10")
        ],
    }
    plan_path = tmp_path / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"
    q00["exact_plan_manifest_sha256"] = _write(plan_path, plan)
    provenance_path = tmp_path / "PROVENANCE.json"
    provenance_sha = _write(provenance_path, {
        "schema": "STAGE_V_EXTERNAL_RUNTIME_PROVENANCE_V1",
        "status": "PASS_RUNTIME_PROVENANCE_CAPTURED",
        "runtime_authorized": False, "outcomes_read": False,
        "intervention_executed": False, "protected_counters": COUNTERS,
        "source_worktree": {"commit": runtime_commit, "tree": runtime_tree, "status_porcelain": ""},
    })
    protocol_path = REPO / "configs/STAGE_V_M4_MATCHED_ACTION_PROTOCOL_V2.json"
    auditor_path = REPO / "scripts/detector_v5/run_stage_v_m4_zero_treatment_auditor.py"
    diff_path = REPO / "scripts/detector_v5/stage_v_runtime_diff.py"
    return {
        "schema": SCHEMA, "status": PASS_STATUS,
        "scope": "ZERO_TREATMENT_Q00_ONLY", "authorization_kind": "Q00_ZERO_TREATMENT_CANARY",
        "canary_authorized": True, "runtime_authorized": True, "owner_authorized": True,
        "requires_explicit_owner_authorization": True, "owner_authorization_basis": "test-owner-review",
        "formal_m4_authorized": False,
        "source_binding": {
            "runtime_commit": runtime_commit, "runtime_tree": runtime_tree,
            "snapshot_source_commit": snapshot_commit, "snapshot_source_tree": snapshot_tree,
        },
        "q00": q00,
        "resource_contract": {
            "minimum_free_memory_mib": 20480,
            "strict_comparison": "free_memory_mib > minimum_free_memory_mib",
            "maximum_project_workers_per_gpu": 1,
            "foreign_workload_allowed": True,
            "foreign_process_interference": False,
            "partial_fleet_allowed": True,
        },
        "zero_treatment": {
            "post_snapshot_primary_window_steps": 0, "treatment_steps": 0,
            "forced_open_steps": 0, "label_records": 0, "v_phys_generated": False,
            "intervention_executed": False, "outcomes_read": False,
            "protected_counters": COUNTERS,
            "primary_input_authority": "loaded_frozen_canonical_bytes",
            "fresh_render_primary_consumption": False,
        },
        "bindings": {
            "m4_v2_protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "exact_plan_manifest": {"path": str(plan_path), "sha256": q00["exact_plan_manifest_sha256"]},
            "snapshot_manifest": {"path": str(snapshot_path), "sha256": q00["snapshot_manifest_sha256"]},
            "runtime_provenance_receipt": {"path": str(provenance_path), "sha256": provenance_sha},
            "zero_treatment_auditor": {"path": str(auditor_path), "sha256": sha256_file(auditor_path)},
            "runtime_diff": {"path": str(diff_path), "sha256": sha256_file(diff_path)},
        },
    }


def test_q00_authority_binds_zero_scope_and_frozen_q00(tmp_path: Path) -> None:
    result = validate_q00_authority(_authority(tmp_path))
    assert result["status"] == PASS_STATUS
    assert result["probe_id"] == "Q00"
    assert result["probe_step"] == 46
    assert result["protected_counters"] == COUNTERS


def test_q00_design_must_not_be_launch_authorized(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    authority.update({"status": "FROZEN_PROSPECTIVE_NOT_AUTHORIZED", "canary_authorized": False, "runtime_authorized": False})
    assert validate_q00_authority(authority, require_launch=False)["status"] == "FROZEN_PROSPECTIVE_NOT_AUTHORIZED"
    with pytest.raises(Q00AuthorityError, match="Q00_AUTHORITY_STATUS"):
        validate_q00_authority(authority)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("formal_m4_authorized",), True, "Q00_FORMAL_M4_MUST_REMAIN_FALSE"),
        (("owner_authorized",), False, "Q00_OWNER_AUTHORIZED"),
        (("resource_contract", "strict_comparison"), "free_memory_mib >= minimum_free_memory_mib", "RESOURCE_COMPARISON"),
        (("zero_treatment", "treatment_steps"), 1, "ZERO_TREATMENT_TREATMENT_STEPS"),
    ],
)
def test_q00_authority_rejects_boundary_bypass(tmp_path: Path, path: tuple[str, ...], value: object, error: str) -> None:
    authority = _authority(tmp_path)
    target = authority
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(Q00AuthorityError, match=error):
        validate_q00_authority(authority)


def test_q00_authority_rejects_missing_arm_identity(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    plan_path = Path(authority["bindings"]["exact_plan_manifest"]["path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["probe_authorities"] = plan["probe_authorities"][:-1]
    plan_sha = _write(plan_path, plan)
    authority["bindings"]["exact_plan_manifest"]["sha256"] = plan_sha
    authority["q00"]["exact_plan_manifest_sha256"] = plan_sha
    with pytest.raises(Q00AuthorityError, match="Q00_EXACT_PLAN_ARM_CLOSURE_INVALID"):
        validate_q00_authority(authority)
