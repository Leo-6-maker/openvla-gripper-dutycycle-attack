from __future__ import annotations

import json
from pathlib import Path

import pytest

from gripper_attack.factorized_scheduler import FactorizedSchedulerConfig, FactorizedV2OneShotScheduler
from gripper_attack.factorized_scheduler_adapter import (
    FactorizedSchedulerAdapterError,
    FactorizedV2SchedulerAdapter,
    apply_calibration,
    validate_calibration_v2,
)
from gripper_attack.factorized_calibration import (
    CalibrationPlanError,
    validate_execution_authorization_template_v2,
    validate_structured_inner_plan,
)
from scripts.detector_v5.rematerialize_factorized_v2_runtime_inputs import EXPECTED_SPLITS, rematerialize


SHA = "a" * 64
COMMIT = "b" * 40


def _head(method="RAW", threshold=0.5, a=1.0, b=0.0):
    return {
        "method": method,
        "a": a,
        "b": b,
        "threshold": threshold,
        "transform": "probability=sigmoid(a*raw_logit+b)",
        "method_valid": True,
        "transform_valid": True,
        "fit_data_valid": True,
        "provenance": "synthetic_test_only",
        "fit_manifest_sha256": SHA,
        "policy_selection_manifest_sha256": SHA,
    }


def _calibration(**changes):
    value = {
        "schema": "FACTORIZED_V2_CALIBRATION_AND_THRESHOLD_CONTRACT_V2",
        "checkpoint_sha256": SHA,
        "split": "o0_i0",
        "scheduler_source_sha256": SHA,
        "structural_config_sha256": SHA,
        "student_source_commit": COMMIT,
        "feature_order_sha256": SHA,
        "grasp": _head(threshold=0.5),
        "manipulation": _head(threshold=0.5),
        "release": _head(threshold=0.9),
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    }
    value.update(changes)
    return value


def _structure(**changes):
    value = json.loads(Path("configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json").read_text())
    value.update(candidate_dwell=1, persistence_window=1, persistence_required=1)
    value.update(changes)
    return value


def test_calibration_v2_supports_raw_intercept_and_platt_without_temperature():
    for method in ("RAW", "INTERCEPT_ONLY", "PLATT"):
        value = _calibration(grasp=_head(method=method, a=1.0, b=0.25))
        assert validate_calibration_v2(value)["schema"].endswith("V2")
        assert 0.0 < apply_calibration(value["grasp"], 0.0) < 1.0
    with pytest.raises(FactorizedSchedulerAdapterError):
        validate_calibration_v2(dict(_calibration(), temperature=1.0))


def test_adapter_applies_a_b_and_threshold_changes_trace():
    adapter = FactorizedV2SchedulerAdapter(_structure(), _calibration())
    row = {"step": 0, "candidate_close": True, "action_known": True, "student_valid": True, "route_supported": True, "grasp_logit": 4.0, "manipulation_logit": 4.0, "release_logit": -4.0}
    trace = adapter.step(row)
    assert trace["probabilities"]["grasp"] > 0.9
    assert trace["emit"] is True
    high = _calibration(manipulation=_head(threshold=0.99))
    assert FactorizedV2SchedulerAdapter(_structure(), high).step(row)["emit"] is False


def test_scheduler_api_fixture_has_deterministic_emit_step():
    fixture = json.loads(Path("tests/fixtures/factorized_scheduler_api_v1_trace.json").read_text())
    calibration = json.loads(Path("tests/fixtures/factorized_calibration_fixture.json").read_text())
    scheduler = FactorizedV2OneShotScheduler(FactorizedSchedulerConfig.from_mapping(fixture["config"], calibration))
    traces = [scheduler.step(row) for row in fixture["steps"]]
    assert [row["step"] for row in traces if row["emit"]] == fixture["expected"]["emit_steps"]
    assert sum(int(row["emit"]) for row in traces) == 1


def test_exact_split_names_fail_closed():
    eleven = [{"name": name} for name in EXPECTED_SPLITS[:-1]]
    with pytest.raises(Exception, match="EXACT_12_SPLIT_CLOSURE_REQUIRED"):
        rematerialize(eleven, Path("_never_created_v3"))
    wrong = [{"name": name} for name in EXPECTED_SPLITS[:-1]] + [{"name": "o3_i9"}]
    with pytest.raises(Exception, match="UNEXPECTED_SPLIT_NAME"):
        rematerialize(wrong, Path("_never_created_v3"))


def _job(split: str):
    outer, inner = int(split[1]), int(split[4])
    return {
        "split": split,
        "outer_fold": outer,
        "inner_fold": inner,
        "seed": 20260717,
        "predictor_module": "scripts.detector_v5.predict_factorized_v2_inner_cv",
        "predictor_script": "scripts/detector_v5/predict_factorized_v2_inner_cv.py",
        "checkpoint_path": f"/sealed/{split}/checkpoint.pt",
        "checkpoint_sha256": SHA,
        "identity_manifest_path": f"/sealed/{split}/identities.json",
        "identity_manifest_sha256": SHA,
        "inner_train_identities": [f"libero_object/task_{outer:02d}/state_00"],
        "heldout_identities": [f"libero_object/task_{outer:02d}/state_01"],
        "feature_root": f"/sealed/{split}/features",
        "feature_seal_sha256": SHA,
        "output_root": f"/new/{split}",
        "expected_output_schema": "FACTORIZED_V2_INNER_TRAIN_CALIBRATION_PREDICTION_V1",
    }


def test_structured_plan_rejects_arbitrary_commands_and_closes_splits():
    plan = {
        "schema": "OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_PLAN_V2",
        "status": "PLAN_ONLY",
        "jobs": [_job(split) for split in EXPECTED_SPLITS],
        "forbidden_roots": ["FIT-DEV", "CAL", "CHECK", "attack"],
        "formal_selection_eligible": False,
        "training_authorized": False,
        "attack_authorized": False,
    }
    assert validate_structured_inner_plan(plan)["split_names"] == list(EXPECTED_SPLITS)
    plan["commands"] = []
    with pytest.raises(CalibrationPlanError, match="STRUCTURED_PLAN_SCHEMA"):
        validate_structured_inner_plan(plan)


def test_null_execution_template_is_prepare_only():
    value = json.loads(Path("configs/OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V2.json").read_text())
    validate_execution_authorization_template_v2(value)
    value["execution_authorized"] = True
    with pytest.raises(CalibrationPlanError):
        validate_execution_authorization_template_v2(value)
