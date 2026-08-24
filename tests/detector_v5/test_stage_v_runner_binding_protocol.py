from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.detector_v5.stage_v_runner_binding_protocol import (
    CONTRACT_FIELDS,
    RunnerBindingError,
    canonical_sha256,
    validate_pair,
    validate_protocol,
    validate_receipt,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "configs/stage_v_runner_binding_protocol_v1.json").read_text())


def _receipt(mode: str) -> dict:
    contract = {field: f"{field}-value" for field in CONTRACT_FIELDS}
    contract["seed"] = 7
    contract["num_steps_wait"] = 10
    contract["suite_horizon"] = 520
    return {
        "schema": "STAGE_V_RUNNER_BINDING_RECEIPT_V1",
        "mode": mode,
        "execution_contract": contract,
        "execution_contract_sha256": canonical_sha256(contract),
        "clean_core_sha256": contract["clean_core_sha256"],
        "clean_prefix_shared": True,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
    }


def test_protocol_schema_is_frozen() -> None:
    assert validate_protocol(PROTOCOL)["status"].startswith("FROZEN_PROTOCOL_SCHEMA")


def test_matching_receipts_pass() -> None:
    result = validate_pair(_receipt("CLEAN_QUALIFICATION"), _receipt("COUNTERFACTUAL"), PROTOCOL)
    assert result["verdict"] == "PASS"


def test_contract_mismatch_fails_closed() -> None:
    qualification = _receipt("CLEAN_QUALIFICATION")
    science = _receipt("COUNTERFACTUAL")
    science["execution_contract"]["source_commit"] = "different"
    science["execution_contract_sha256"] = canonical_sha256(science["execution_contract"])
    with pytest.raises(RunnerBindingError, match="RUNNER_BINDING_MISMATCH"):
        validate_pair(qualification, science, PROTOCOL)


def test_digest_drift_fails_closed() -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    receipt["execution_contract"]["seed"] = 8
    with pytest.raises(RunnerBindingError, match="DIGEST_MISMATCH"):
        validate_receipt(receipt, expected_mode="CLEAN_QUALIFICATION")


def test_unexpected_contract_field_fails_closed() -> None:
    receipt = _receipt("CLEAN_QUALIFICATION")
    receipt["execution_contract"]["unfrozen_field"] = "drift"
    receipt["execution_contract_sha256"] = canonical_sha256(receipt["execution_contract"])
    with pytest.raises(RunnerBindingError, match="UNEXPECTED"):
        validate_receipt(receipt, expected_mode="CLEAN_QUALIFICATION")


def test_protected_boundary_fails_closed() -> None:
    qualification = _receipt("CLEAN_QUALIFICATION")
    science = _receipt("COUNTERFACTUAL")
    science["eval160_reads"] = 1
    with pytest.raises(RunnerBindingError, match="PROTECTED_BOUNDARY"):
        validate_pair(qualification, science, PROTOCOL)
