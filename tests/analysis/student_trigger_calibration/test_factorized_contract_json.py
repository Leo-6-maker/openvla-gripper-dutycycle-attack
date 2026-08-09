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


def test_protocol_has_phase_b_gate_and_identity_closure():
    value = strict_load(CONTRACTS[1])
    assert value["status"] == "PHASE_B_IDENTITY_PASS_AWAITING_AUTHORITATIVE_K10_TEACHER_BUNDLES"
    assert "phase_summary" in value
    assert value["phase_summary"]["A_codex_static_handoff_and_seal"]["status"] == "COMPLETE"
    closure = value["identity_closure"]
    assert "five_roles" in closure
    for role in ["T", "C", "P", "H", "A"]:
        assert role in closure["five_roles"]
    assert value["formal_gate"]["attack"] == "HOLD"
    assert value["formal_gate"]["phase_b_identity_closure"] == "PASS_DETERMINISTIC_ALLOCATION"
    steps = value["phase_summary"]["C_production_inference_calibration_and_l3"]["steps"]
    c_index = next(i for i, step in enumerate(steps) if "C identities" in step)
    p_index = next(i for i, step in enumerate(steps) if "P identities" in step)
    freeze_index = next(i for i, step in enumerate(steps) if "external heldout-L3 authorization receipt" in step)
    h_index = next(i for i, step in enumerate(steps) if "heldout-L3 Student inference exactly once" in step)
    assert c_index < freeze_index and p_index < freeze_index < h_index


def test_design_and_requirements_match_current_execution_boundary():
    design = strict_load(CONTRACTS[2])
    requirements = strict_load(CONTRACTS[3])
    expected = "IDENTITY_CLOSURE_PASS_AWAITING_AUTHORITATIVE_TEACHER_COVERAGE"
    assert design["status"] == expected
    assert design["current_state"]["identity_audit"] == "PASS_DETERMINISTIC_ALLOCATION"
    assert design["scheme_decision_tree"]["NESTED_RETRAIN_REQUIRED"]["status"] == "NOT_CURRENTLY_REQUIRED"
    assert "five_way_disjointness" in design["holding_rules"]
    assert requirements["status"] == expected
    assert "two_independent_gates" in requirements
    assert "verdict_rules" in requirements
    auth = requirements["phase_c_authorization"]
    assert "CP_INFERENCE_AUTHORIZED" in auth
    assert "CP_INFERENCE_HOLD" in auth
    assert "HELDOUT_L3_DATA_READY" in auth
    assert "HELDOUT_L3_INFERENCE_AUTHORIZED" in auth
