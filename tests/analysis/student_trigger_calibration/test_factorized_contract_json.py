"""Static JSON contract hygiene for Factorized V2 L3 analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONTRACTS = (
    ROOT / "analysis/student_trigger_calibration/factorized_l3_metric_contract.json",
    ROOT / "analysis/student_trigger_calibration/FACTORIZED_V2_L3_EVALUATION_PROTOCOL_V1.json",
    ROOT / "analysis/student_trigger_calibration/FACTORIZED_CALIBRATION_DESIGN_DECISION_V1.json",
    ROOT / "analysis/student_trigger_calibration/INDEPENDENT_FACTORIZED_CALIBRATION_REQUIREMENTS_V1.json",
)


def strict_load(path: Path):
    duplicates: list[str] = []

    def hook(pairs):
        seen = set()
        result = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(str(key))
            seen.add(key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    assert not duplicates, f"duplicate JSON keys in {path}: {duplicates}"
    return value


@pytest.mark.parametrize("path", CONTRACTS)
def test_contract_json_is_duplicate_free(path):
    assert isinstance(strict_load(path), dict)


def test_metric_contract_uses_exact_denominators():
    value = strict_load(CONTRACTS[0])
    assert value["contract"] == "FACTORIZED_V2_L3_METRIC_CONTRACT_V2"
    metrics = value["l3_metrics"]
    assert "total_emitted_episodes" not in metrics
    assert metrics["all_emit_precision"]["denominator"] == "total_emitted_all"
    assert metrics["verified_emit_precision"]["denominator"] == "total_emitted_verified"
    assert metrics["unverifiable_emit_fraction"]["denominator"] == "total_emitted_all"


def test_protocol_requires_real_adapter_and_three_way_identity_split():
    value = strict_load(CONTRACTS[1])
    assert value["status"] == "CODE_STATIC_CLOSURE_PASS_PRODUCTION_ARTIFACTS_PENDING"
    assert value["real_adapter_requirement"]["class"] == "FactorizedV2SchedulerAdapter"
    policy = value["identity_policy"]
    assert "calibrator_fit_identities" in policy["required_sets"]
    assert "policy_selection_identities" in policy["required_sets"]
    assert "heldout_evaluation_identities" in policy["required_sets"]
    assert "checkpoint_training_identities" in policy["required_sets"]
    assert value["formal_gate"]["attack"] == "HOLD"


def test_design_and_requirements_match_current_execution_boundary():
    design = strict_load(CONTRACTS[2])
    requirements = strict_load(CONTRACTS[3])
    assert design["status"] == "CODE_PATH_READY_AWAITING_SEALED_IDENTITY_ARTIFACTS"
    assert design["holding_rules"]["attack_authorization"] is False
    assert requirements["status"] == "CODE_IMPLEMENTED_PRODUCTION_ARTIFACTS_PENDING"
    assert requirements["threshold_selection"]["implementation_status"] == "IMPLEMENTED_AND_CPU_TESTED"
    assert requirements["authorization"] == {
        "training": False,
        "full_fit": False,
        "authoritative_l3": False,
        "attack": False,
    }
