import importlib.util
from pathlib import Path

import pytest

from gripper_attack.b3_v3_attack_protocol import ATTACK_CONDITIONS


def _load(name):
    path = Path(__file__).parents[1] / "scripts" / "detector" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viability_matrix_is_24_runs(tmp_path):
    from gripper_attack.b3_training_protocol import build_fit_fold_manifest, write_fit_fold_bundle
    rows = [{"canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}", "suite": suite, "task_idx": task, "state_id": state, "split": "FIT_TRAIN"} for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10") for task in range(10) for state in range(20)]
    root = tmp_path / "folds"
    write_fit_fold_bundle(root, build_fit_fold_manifest(rows, registry_sha256="a" * 64))
    plan = _load("build_b3_v3_fit_viability_matrix.py").build_matrix_plan(root)
    assert plan["run_count"] == 24
    assert len({(item["fold_id"], item["variant"], item["seed"]) for item in plan["runs"]}) == 24
    assert all(item["formal_training_authorized"] is False for item in plan["runs"])


def test_fit_dev_and_cal_check_contracts_are_separate():
    selector = _load("select_b3_v3_fit_dev_model.py")
    common = {
        "split": "FIT_DEV", "fit_dev_identity_count": 160,
        "checkpoint_sha256": "a" * 64, "viability_report_sha256": "b" * 64,
        "fit_dev_identity_sha256": "c" * 64, "variant": "B3_25D", "seed": 20260717,
    }
    chosen = selector.select_candidate([
        {**common, "candidate_id": "b", "full_t10_event_hit_rate": .5, "negative_episode_any_emit_rate": .1},
        {**common, "candidate_id": "a", "full_t10_event_hit_rate": .6, "negative_episode_any_emit_rate": .2},
    ])
    assert chosen["status"] == "FIT_DEV_SELECTED" and chosen["candidate_id"] == "a"
    cal = _load("calibrate_b3_v3_thresholds.py")
    cal.validate_cal_records([{"split": "CAL", "state_id": 24, "attack_enabled": False}])
    with pytest.raises(ValueError, match="CAL"):
        cal.validate_cal_records([{"split": "CHECK", "state_id": 27}])
    check = _load("check_b3_v3_detector_once.py")
    receipt = check.build_check_receipt(checkpoint_status="FIT_DEV_SELECTED", check_access_key="x")
    assert receipt["check_executed"] is False


def test_attack_conditions_are_frozen_and_no_execution_authorized():
    assert len(ATTACK_CONDITIONS) == 6
    assert "RAND_VALID_T10" in ATTACK_CONDITIONS
