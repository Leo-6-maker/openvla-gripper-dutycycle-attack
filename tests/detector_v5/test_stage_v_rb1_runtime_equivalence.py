from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.detector_v5.stage_v_runner_binding_protocol import CONTRACT_FIELDS, canonical_sha256
from scripts.detector_v5.stage_v_rb1_runtime_equivalence import (
    COMMON_TRACE_HASH_FIELDS,
    NOOP_TRACE_HASH_FIELDS,
    RuntimeEquivalenceError,
    validate_pair,
    validate_protocol,
    verify_artifact_files,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v1.json").read_text())
V2_PROTOCOL = json.loads((ROOT / "configs/stage_v_rb1_runtime_equivalence_protocol_v2.json").read_text())


def _receipt(mode: str, scope: str = "CLEAN_PATH", root: Path | None = None) -> dict:
    contract = {field: f"{field}-value" for field in CONTRACT_FIELDS}
    contract.update({"seed": 7, "num_steps_wait": 10, "suite_horizon": 520})
    trace_fields = list(COMMON_TRACE_HASH_FIELDS)
    if scope == "NOOP_CONTINUATION":
        trace_fields += list(NOOP_TRACE_HASH_FIELDS)
    artifacts = {}
    for name in ["initial_state", "policy_token_trace", "postprocessed_action_trace", "observation_trace", "physical_state_trace"]:
        artifacts[name] = {"path": name + ".jsonl", "sha256": "a" * 64}
    if scope == "NOOP_CONTINUATION":
        for name in ("snapshot_restore_trace", "noop_action_trace"):
            artifacts[name] = {"path": name + ".jsonl", "sha256": "a" * 64}
    return {
        "schema": "STAGE_V_RB1_RUNTIME_RECEIPT_V1",
        "mode": mode,
        "comparison_scope": scope,
        "canonical_parent_key": "libero_10/task_00/state_47",
        "suite": "libero_10",
        "task_index": 0,
        "state_index": 47,
        "execution_contract": contract,
        "execution_contract_sha256": canonical_sha256(contract),
        "clean_core_sha256": contract["clean_core_sha256"],
        "initial_state_sha256": "b" * 64,
        "trace_step_count": 10,
        "termination_step": 9,
        "terminal_outcome": "SUCCESS",
        "trace_hashes": {field: "c" * 64 for field in trace_fields},
        "trace_artifacts": artifacts,
        "independent_recompute": {
            "status": "PASS",
            "recomputed": True,
            "auditor_source_commit": "auditor-commit",
            "auditor_source_tree": "auditor-tree",
            "auditor_sha256": "d" * 64,
            "protocol_sha256": "e" * 64,
        },
        "clean_prefix_shared": True,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "attack_rollouts": 0,
        "intervention_applied_steps": 0,
        "counterfactual_open_steps": 0,
    }


def _v2_receipt(mode: str) -> dict:
    receipt = _receipt(mode)
    receipt["diagnostic_trace_hashes"] = {
        "full_sim_state_trace_sha256": "d" * 64,
        "policy_rgb_224_trace_sha256": "e" * 64,
        "model_input_trace_sha256": "f" * 64,
    }
    receipt["diagnostic_trace_artifacts"] = {
        name: {"path": name + ".jsonl", "sha256": digest}
        for name, digest in (
            ("full_sim_state_trace", "d" * 64),
            ("policy_rgb_224_trace", "e" * 64),
            ("model_input_trace", "f" * 64),
        )
    }
    return receipt


def test_rb1_protocol_is_frozen_and_exact() -> None:
    assert validate_protocol(PROTOCOL)["tolerance_allowed"] is False


def test_rb1_v2_protocol_allows_only_declared_visual_input_differences() -> None:
    assert validate_protocol(V2_PROTOCOL)["schema"] == "STAGE_V_RB1_RUNTIME_EQUIVALENCE_PROTOCOL_V2"
    left = _v2_receipt("CLEAN_QUALIFICATION")
    right = _v2_receipt("COUNTERFACTUAL_CLEAN_PREFIX")
    right["trace_hashes"]["observation_trace_sha256"] = "a" * 64
    right["diagnostic_trace_hashes"]["policy_rgb_224_trace_sha256"] = "b" * 64
    right["diagnostic_trace_hashes"]["model_input_trace_sha256"] = "c" * 64
    result = validate_pair(left, right, V2_PROTOCOL, "RB1A_CLEAN_PATH")
    assert result["causal_execution_equivalence"] == "PASS"
    assert result["visual_input_difference_allowed"] is True


def test_rb1_v2_action_trace_mismatch_still_fails_closed() -> None:
    left = _v2_receipt("CLEAN_QUALIFICATION")
    right = _v2_receipt("COUNTERFACTUAL_CLEAN_PREFIX")
    right["trace_hashes"]["postprocessed_action_trace_sha256"] = "a" * 64
    with pytest.raises(RuntimeEquivalenceError, match="CAUSAL_TRACE_MISMATCH"):
        validate_pair(left, right, V2_PROTOCOL, "RB1A_CLEAN_PATH")


def test_rb1a_matching_trace_receipts_pass() -> None:
    result = validate_pair(
        _receipt("CLEAN_QUALIFICATION"),
        _receipt("COUNTERFACTUAL_CLEAN_PREFIX"),
        PROTOCOL,
        "RB1A_CLEAN_PATH",
    )
    assert result["runtime_trace_equivalence"] == "PASS"


def test_initial_state_mismatch_fails_closed() -> None:
    left = _receipt("CLEAN_QUALIFICATION")
    right = _receipt("COUNTERFACTUAL_CLEAN_PREFIX")
    right["initial_state_sha256"] = "f" * 64
    with pytest.raises(RuntimeEquivalenceError, match="initial_state_sha256"):
        validate_pair(left, right, PROTOCOL, "RB1A_CLEAN_PATH")


def test_action_trace_mismatch_fails_closed() -> None:
    left = _receipt("CLEAN_QUALIFICATION")
    right = _receipt("COUNTERFACTUAL_CLEAN_PREFIX")
    right["trace_hashes"]["postprocessed_action_trace_sha256"] = "f" * 64
    with pytest.raises(RuntimeEquivalenceError, match="trace_hashes"):
        validate_pair(left, right, PROTOCOL, "RB1A_CLEAN_PATH")


def test_trace_mismatch_reports_exact_field() -> None:
    left = _receipt("CLEAN_QUALIFICATION")
    right = _receipt("COUNTERFACTUAL_CLEAN_PREFIX")
    right["trace_hashes"]["observation_trace_sha256"] = "f" * 64
    with pytest.raises(RuntimeEquivalenceError, match="observation_trace_sha256"):
        validate_pair(left, right, PROTOCOL, "RB1A_CLEAN_PATH")


def test_rb1b_requires_restore_trace_and_probe_identity() -> None:
    left = _receipt("UNINTERRUPTED_CLEAN", "NOOP_CONTINUATION")
    right = _receipt("SNAPSHOT_RESTORE_NOOP", "NOOP_CONTINUATION")
    with pytest.raises(RuntimeEquivalenceError, match="PROBE_STEP"):
        validate_pair(left, right, PROTOCOL, "RB1B_NOOP_CONTINUATION")


def test_nonzero_intervention_boundary_fails_closed() -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    receipt["intervention_applied_steps"] = 1
    with pytest.raises(RuntimeEquivalenceError, match="PROTECTED_BOUNDARY"):
        from scripts.detector_v5.stage_v_rb1_runtime_equivalence import validate_receipt
        validate_receipt(receipt, PROTOCOL)


def test_artifact_bytes_are_independently_verified(tmp_path: Path) -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    trace_fields = {
        "policy_token_trace": "policy_token_trace_sha256",
        "postprocessed_action_trace": "postprocessed_action_trace_sha256",
        "observation_trace": "observation_trace_sha256",
        "physical_state_trace": "physical_state_trace_sha256",
    }
    for name, item in receipt["trace_artifacts"].items():
        path = tmp_path / item["path"]
        data = json.dumps({"initial_state_sha256": receipt["initial_state_sha256"], "identity": {field: receipt[field] for field in ("canonical_parent_key", "suite", "task_index", "state_index")}}).encode() if name == "initial_state" else b"trace"
        path.write_bytes(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
        if name in trace_fields:
            receipt["trace_hashes"][trace_fields[name]] = item["sha256"]
    verify_artifact_files(receipt, tmp_path, PROTOCOL)


def test_trace_hash_manifest_drift_fails_closed(tmp_path: Path) -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    for name, item in receipt["trace_artifacts"].items():
        data = json.dumps({"initial_state_sha256": receipt["initial_state_sha256"], "identity": {field: receipt[field] for field in ("canonical_parent_key", "suite", "task_index", "state_index")}}).encode() if name == "initial_state" else b"trace"
        (tmp_path / item["path"]).write_bytes(data)
        item["sha256"] = hashlib.sha256(b"trace").hexdigest()
    with pytest.raises(RuntimeEquivalenceError, match="TRACE_HASH_MANIFEST_MISMATCH"):
        verify_artifact_files(receipt, tmp_path, PROTOCOL)


def test_artifact_hash_drift_fails_closed(tmp_path: Path) -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    trace_fields = {
        "policy_token_trace": "policy_token_trace_sha256",
        "postprocessed_action_trace": "postprocessed_action_trace_sha256",
        "observation_trace": "observation_trace_sha256",
        "physical_state_trace": "physical_state_trace_sha256",
    }
    for name, item in receipt["trace_artifacts"].items():
        data = json.dumps({"initial_state_sha256": receipt["initial_state_sha256"], "identity": {field: receipt[field] for field in ("canonical_parent_key", "suite", "task_index", "state_index")}}).encode() if name == "initial_state" else b"trace"
        (tmp_path / item["path"]).write_bytes(data)
        if name in trace_fields:
            item["sha256"] = receipt["trace_hashes"][trace_fields[name]]
    with pytest.raises(RuntimeEquivalenceError, match="SHA256_MISMATCH"):
        verify_artifact_files(receipt, tmp_path, PROTOCOL)
