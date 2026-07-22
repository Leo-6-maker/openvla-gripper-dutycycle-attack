from __future__ import annotations

import copy
import json
from pathlib import Path
from argparse import Namespace

import pytest

from gripper_attack.factorized_calibration import CalibrationPlanError, validate_authorization_template, validate_inner_train_plan
from gripper_attack.factorized_scheduler_bridge import (
    SchedulerBridgeError,
    build_scheduler_ready_record,
    exact_step_join,
    validate_scheduler_ready_record,
)
from gripper_attack.b3_training_protocol import verify_sealed_directory


SHA = "a" * 64
COMMIT = "b" * 40


def _rows(raw=0.0):
    prediction = {
        "canonical_parent_key": "libero_object/task_00/state_00",
        "step": 0,
        "mechanism_route": "single_object_pick_place",
        "route_supported": True,
        "utility_probability": 0.8,
        "utility_source": "DIRECT",
        "release_probability": 0.1,
        "release_source": "DIRECT",
        "regrasp_probability": 0.2,
        "regrasp_source": "DIRECT",
        "uncertainty_probability": 0.0,
        "uncertainty_source": "DERIVED",
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
    return build_scheduler_ready_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA, source_commit=COMMIT,
        input_artifact_seal=SHA, feature_order_sha256=SHA,
    )


def test_raw_gripper_close_classification_is_canonical():
    assert _record()["candidate_close"] is True
    assert _record()["candidate_close_source"] == "DERIVED"
    assert _record(raw_gripper=0.0)["candidate_close"] is True


def test_boundary_is_not_close():
    prediction, student, runtime = _rows(raw=0.5)
    row = build_scheduler_ready_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA, source_commit=COMMIT,
        input_artifact_seal=SHA, feature_order_sha256=SHA,
    )
    assert row["candidate_close"] is False
    assert row["action_intent"] == "BOUNDARY"


def test_certified_fallback_is_allowed_and_uncertified_fallback_is_rejected():
    prediction, student, runtime = _rows()
    runtime["action_raw"] = runtime.pop("clean_action_raw_7d")
    with pytest.raises(SchedulerBridgeError, match="FALLBACK_RAW_ACTION_UNCERTIFIED"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )
    cert = {
        "field_semantics": "OPENVLA_RAW_ACTION",
        "field_stage": "CLEAN_PRE_ATTACK_DECODE",
        "field_dimension": 7,
        "gripper_index": 6,
        "postprocessed": False,
        "attacked": False,
    }
    row = build_scheduler_ready_record(
        prediction, student, runtime,
        checkpoint_sha256=SHA, source_commit=COMMIT,
        input_artifact_seal=SHA, feature_order_sha256=SHA,
        runtime_manifest=cert,
    )
    assert row["raw_gripper_source_field"] == "action_raw[6]"


def test_two_raw_fields_must_match_and_attacked_field_is_rejected():
    prediction, student, runtime = _rows()
    runtime["action_raw"] = [0.0] * 6 + [1.0]
    with pytest.raises(SchedulerBridgeError, match="RAW_ACTION_FIELDS_MISMATCH"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )
    prediction, student, runtime = _rows()
    runtime["attacked_action"] = [0.0] * 7
    with pytest.raises(SchedulerBridgeError, match="ATTACKED_ACTION_FORBIDDEN"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )


def test_teacher_event_cannot_supply_runtime_candidate_gate():
    prediction, student, runtime = _rows()
    runtime.pop("clean_action_raw_7d")
    prediction["event_id"] = 7
    with pytest.raises(SchedulerBridgeError, match="RUNTIME_RAW_GRIPPER_MISSING"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )


def test_teacher_event_and_future_fields_are_rejected():
    for key in ("event_id", "teacher_phase", "window_end", "future_score"):
        record = _record()
        record[key] = 0
        with pytest.raises(SchedulerBridgeError):
            validate_scheduler_ready_record(record)


def test_utility_and_regrasp_proxy_substitution_is_rejected():
    with pytest.raises(SchedulerBridgeError, match="PROXY_HEAD_REJECTED"):
        _record(utility_source="grasp_prob")
    with pytest.raises(SchedulerBridgeError, match="PROXY_HEAD_REJECTED"):
        _record(regrasp_source="manipulation_prob")


def test_missing_heads_do_not_get_zero_defaults():
    prediction, student, runtime = _rows()
    prediction.pop("utility_probability")
    with pytest.raises(SchedulerBridgeError, match="UTILITY_MISSING"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )


def test_exact_step_join_rejects_missing_and_duplicate_steps():
    prediction, student, runtime = _rows()
    with pytest.raises(SchedulerBridgeError, match="STEP_SET_MISMATCH"):
        exact_step_join([prediction], [], [runtime])
    with pytest.raises(SchedulerBridgeError, match="DUPLICATE_PREDICTION_STEP"):
        exact_step_join([prediction, copy.deepcopy(prediction)], [student], [runtime])


def test_seal_and_checkpoint_bindings_are_validated():
    prediction, student, runtime = _rows()
    with pytest.raises(SchedulerBridgeError, match="CHECKPOINT_SHA_INVALID"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256="bad", source_commit=COMMIT,
            input_artifact_seal=SHA, feature_order_sha256=SHA,
        )
    with pytest.raises(SchedulerBridgeError, match="INPUT_SEAL_INVALID"):
        build_scheduler_ready_record(
            prediction, student, runtime,
            checkpoint_sha256=SHA, source_commit=COMMIT,
            input_artifact_seal="bad", feature_order_sha256=SHA,
        )


def test_deterministic_ready_record():
    assert _record() == _record()


def _plan_item(i):
    return {
        "outer_fold": i % 4,
        "inner_fold": i % 3,
        "seed": 42 + i,
        "checkpoint_sha256": f"{i:064x}",
        "identity_manifest_sha256": SHA,
        "inner_train_identities": [f"libero_object/task_{i % 10:02d}/state_{i % 20:02d}"],
        "heldout_identities": [f"libero_object/task_{i % 10:02d}/state_{(i + 1) % 20:02d}"],
        "checkpoint_root": f"/sealed/checkpoint_{i}",
        "identity_manifest_root": f"/sealed/identity_{i}",
        "feature_root": f"/sealed/feature_{i}",
    }


def test_inner_train_and_heldout_sets_must_be_disjoint():
    items = [_plan_item(i) for i in range(12)]
    summary = validate_inner_train_plan(items)
    assert summary["checkpoint_count"] == 12
    items[0]["heldout_identities"] = list(items[0]["inner_train_identities"])
    with pytest.raises(CalibrationPlanError, match="INNER_TRAIN_HELDOUT_OVERLAP"):
        validate_inner_train_plan(items)


def test_cal_check_paths_are_rejected():
    with pytest.raises(CalibrationPlanError, match="PROTECTED_SPLIT_PATH"):
        validate_inner_train_plan([_plan_item(i) for i in range(12)], forbidden_roots=["/sealed/CAL"])


def test_authorization_template_is_disabled():
    template = json.loads(Path("configs/OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V1.json").read_text())
    validate_authorization_template(template)
    assert template["execution_authorized"] is False
    assert template["formal_selection_eligible"] is False


def test_prepare_only_runner_seals_and_is_nonoverwrite(tmp_path: Path):
    from scripts.detector_v5.run_factorized_v2_inner_train_calibration import prepare

    plan = {"checkpoints": [_plan_item(i) for i in range(12)], "forbidden_roots": []}
    auth = json.loads(Path("configs/OFFLINE_FACTORIZED_V2_INNER_TRAIN_INFERENCE_AUTH_V1.json").read_text())
    plan_path = tmp_path / "plan.json"
    auth_path = tmp_path / "authorization.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    auth_path.write_text(json.dumps(auth), encoding="utf-8")
    output = tmp_path / "prepared"
    args = Namespace(plan=plan_path, authorization=auth_path, output_root=output, prepare_only=True)
    result = prepare(args)
    assert result["status"] == "PREPARATION_ONLY"
    verify_sealed_directory(output)
    with pytest.raises(FileExistsError):
        prepare(args)
