from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gripper_attack.factorized_runtime import (
    FactorizedRuntimeError,
    build_runtime_record,
    exact_runtime_step_join,
    validate_runtime_record,
)
from gripper_attack.factorized_scheduler import FactorizedSchedulerConfig, FactorizedV2OneShotScheduler
from scripts.detector_v5.rematerialize_factorized_v2_runtime_inputs import rematerialize


SHA = "a" * 64
COMMIT = "b" * 40


def _rows(raw=0.0):
    prediction = {
        "canonical_parent_key": "libero_object/task_00/state_00",
        "step": 0,
        "mechanism_route": "single_object_pick_place",
        "route_supported": True,
        "grasp_prob": 0.8,
        "manipulation_prob": 0.8,
        "release_prob": 0.1,
        "grasp_logit": 1.4,
        "manipulation_logit": 1.4,
        "release_logit": -1.4,
    }
    student = {
        "canonical_parent_key": prediction["canonical_parent_key"],
        "step": 0,
        "valid": True,
        "features_25d": [0.0] * 25,
    }
    runtime = {
        "canonical_parent_key": prediction["canonical_parent_key"],
        "step": 0,
        "clean_action_raw_7d": [0.0] * 6 + [raw],
    }
    return prediction, student, runtime


def _record(**changes):
    prediction, student, runtime = _rows()
    prediction.update(changes)
    return build_runtime_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA,
        source_commit=COMMIT,
        prediction_artifact_seal=SHA,
        runtime_artifact_seal=SHA,
        feature_order_sha256=SHA,
    )


def test_factorized_runtime_has_no_legacy_or_teacher_fields():
    record = _record()
    assert "utility_probability" not in record
    assert "regrasp_probability" not in record
    for key in ("event_id", "strict_k10_feasible", "grasp_known_mask"):
        mutated = dict(record, **{key: 1})
        with pytest.raises(FactorizedRuntimeError):
            validate_runtime_record(mutated)


def test_candidate_close_prefers_clean_raw_fallback_requires_certification_and_boundary_abstains():
    record = _record()
    assert record["candidate_close"] is True
    assert record["action_known"] is True
    assert record["raw_gripper_source_field"] == "clean_action_raw_7d[6]"

    prediction, student, runtime = _rows(raw=0.5)
    boundary = build_runtime_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA, source_commit=COMMIT,
        prediction_artifact_seal=SHA, runtime_artifact_seal=SHA, feature_order_sha256=SHA,
    )
    assert boundary["candidate_close"] is False and boundary["action_known"] is False

    prediction, student, runtime = _rows()
    runtime["action_raw"] = runtime.pop("clean_action_raw_7d")
    with pytest.raises(FactorizedRuntimeError, match="FALLBACK_RAW_ACTION_UNCERTIFIED"):
        build_runtime_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            prediction_artifact_seal=SHA, runtime_artifact_seal=SHA, feature_order_sha256=SHA,
        )
    cert = {
        "field_semantics": "OPENVLA_RAW_ACTION",
        "field_stage": "CLEAN_PRE_ATTACK_DECODE",
        "field_dimension": 7,
        "gripper_index": 6,
        "postprocessed": False,
        "attacked": False,
    }
    fallback = build_runtime_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA, source_commit=COMMIT,
        prediction_artifact_seal=SHA, runtime_artifact_seal=SHA, feature_order_sha256=SHA,
        runtime_manifest=cert,
    )
    assert fallback["raw_gripper_source_field"] == "action_raw[6]"


def test_no_utility_or_regrasp_proxy_and_no_zero_fallback():
    with pytest.raises(FactorizedRuntimeError, match="GRASP_PROBABILITY_MISSING"):
        _record(grasp_prob=None)
    with pytest.raises(FactorizedRuntimeError, match="LEGACY_HEADS_FORBIDDEN"):
        _record(utility_probability=0.9)


def test_runtime_scheduler_is_invariant_to_separate_offline_labels():
    config = FactorizedSchedulerConfig.from_files(
        Path("configs/FACTORIZED_V2_SCHEDULER_PROTOCOL_V1.json"),
        Path("tests/fixtures/factorized_calibration_fixture.json"),
    )
    base = {key: value for key, value in _record().items() if key in {
        "step", "candidate_close", "action_known", "student_valid", "route_supported",
        "grasp_probability", "manipulation_probability", "release_probability",
    }}
    scheduler_a = FactorizedV2OneShotScheduler(config)
    scheduler_b = FactorizedV2OneShotScheduler(config)
    assert scheduler_a.step(base) == scheduler_b.step(base)
    offline = {"event_id": 4, "grasp_known_mask": False, "strict_k10_feasible": True}
    assert offline and scheduler_b.state == scheduler_a.state


def test_exact_runtime_join_rejects_duplicate_missing_and_future_fields():
    prediction, student, runtime = _rows()
    with pytest.raises(FactorizedRuntimeError, match="STEP_SET_MISMATCH"):
        exact_runtime_step_join([prediction], [], [runtime])
    with pytest.raises(FactorizedRuntimeError, match="DUPLICATE_PREDICTION_STEP"):
        exact_runtime_step_join([prediction, copy.deepcopy(prediction)], [student], [runtime])


def test_rematerializer_requires_exact_twelve_and_is_nonoverwrite(tmp_path: Path):
    with pytest.raises(FactorizedRuntimeError, match="EXACT_12_SPLIT_CLOSURE_REQUIRED"):
        rematerialize([], tmp_path / "output")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        rematerialize([], output, require_twelve=False)
