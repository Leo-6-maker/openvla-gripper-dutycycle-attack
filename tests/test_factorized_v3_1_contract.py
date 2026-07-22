from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gripper_attack.factorized_scheduler_adapter import (
    FactorizedSchedulerAdapterError,
    FactorizedV2SchedulerAdapter,
    validate_calibration_v3,
)
from scripts.detector_v5.audit_factorized_calibration_design_feasibility import audit
from scripts.detector_v5.validate_factorized_v2_handoff import (
    _strict_json,
    receipt_binding_sha,
    validate_v3_1,
)


FIXTURE = Path("tests/fixtures/factorized_scheduler_api_v3_1_trace.json")
DUPLICATED_HANDOFF_FIXTURE = Path("tests/fixtures/factorized_v3_1_duplicate_key.json")


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_v3_diagnostic_contract_is_valid_and_authoritative_requires_independent_sources():
    fixture = _fixture()
    assert validate_calibration_v3(fixture["calibration"])["status"] == "DIAGNOSTIC"
    authoritative = copy.deepcopy(fixture["calibration"])
    authoritative.update(calibration_fit_authoritative=True, threshold_selection_authoritative=True, l3_evaluation_eligible=True, status="AUTHORITATIVE")
    with pytest.raises(FactorizedSchedulerAdapterError, match="L3_PROVENANCE"):
        validate_calibration_v3(authoritative)
    for name in ("grasp", "manipulation", "release"):
        authoritative[name]["provenance_class"] = "INDEPENDENT_CALIBRATION"
    assert validate_calibration_v3(authoritative)["status"] == "AUTHORITATIVE"


def test_adapter_binds_runtime_and_run_episode_first_emit():
    fixture = _fixture()
    adapter = FactorizedV2SchedulerAdapter(fixture["structure"], fixture["calibration"])
    result = adapter.run_episode(fixture["runtime_rows"])
    assert result["ever_emitted"] is True
    assert result["first_emit_step"] == fixture["expected"]["first_emit_step"]
    assert result["first_emit_trace"]["diagnostic_only"] is True
    assert sum(trace["emit"] for trace in result["per_step_trace"]) == 1
    assert result["per_step_trace"][1]["reason"] == "ONE_SHOT_LATCHED"


@pytest.mark.parametrize("field", ["checkpoint_sha256", "source_commit", "feature_order_sha256", "split", "scheduler_source_sha256", "structural_config_sha256"])
def test_adapter_rejects_runtime_contract_binding_mismatch(field: str):
    fixture = _fixture()
    row = dict(fixture["runtime_rows"][0])
    row[field] = "c" * (40 if field == "source_commit" else 64)
    adapter = FactorizedV2SchedulerAdapter(fixture["structure"], fixture["calibration"])
    with pytest.raises(FactorizedSchedulerAdapterError):
        adapter.step(row)


def test_threshold_changes_trace_without_refitting():
    fixture = _fixture()
    changed = copy.deepcopy(fixture["calibration"])
    changed["grasp"]["threshold"] = 0.99
    adapter = FactorizedV2SchedulerAdapter(fixture["structure"], changed)
    assert adapter.step(fixture["runtime_rows"][0])["emit"] is False


def test_teacher_and_offline_fields_are_rejected():
    fixture = _fixture()
    adapter = FactorizedV2SchedulerAdapter(fixture["structure"], fixture["calibration"])
    row = dict(fixture["runtime_rows"][0], strict_k10_feasible=True)
    with pytest.raises(FactorizedSchedulerAdapterError, match="FORBIDDEN_RUNTIME_FIELD"):
        adapter.step(row)


def test_identity_audit_is_fail_closed_without_roots():
    result = audit(None)
    assert result["verdict"] == "BLOCKED_ROOTS_NOT_MOUNTED"
    assert result["production_inference"] is False


def test_v3_1_canonical_handoff_is_static_pass():
    result = validate_v3_1(Path("reports/DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1.json"))
    assert result["status"] == "STATIC_INTEGRATION_PASS"
    assert result["expected_split_keys"] == [f"o{outer}_i{inner}" for outer in range(4) for inner in range(3)]


def test_v3_1_duplicate_key_fixture_is_rejected():
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY:schema"):
        _strict_json(DUPLICATED_HANDOFF_FIXTURE.read_text(encoding="utf-8"))


def test_v3_1_runtime_schema_name_and_file_are_canonical():
    value = _strict_json(Path("reports/DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1.json").read_text(encoding="utf-8"))
    runtime_bundle = value["runtime_bundle"]
    assert runtime_bundle["schema_name"] == "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_BUNDLE_V2"
    assert runtime_bundle["schema_file"]["path"] == "schemas/factorized_v2_runtime_scheduler_input.schema.json"
    assert len(runtime_bundle["schema_file"]["sha256"]) == 64


def test_v3_1_receipt_binds_corrected_handoff():
    handoff = _strict_json(Path("reports/DEEPSEEK_FACTORIZED_SCHEDULER_HANDOFF_V3_1.json").read_text(encoding="utf-8"))
    receipt = _strict_json(Path("reports/FACTORIZED_V3_1_HANDOFF_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["handoff_blob_sha256"] == handoff["handoff_blob_sha256"]
    assert handoff["production_receipt_requirements"]["handoff_receipt"]["sha256"] == receipt_binding_sha(Path("reports/FACTORIZED_V3_1_HANDOFF_RECEIPT.json"))
