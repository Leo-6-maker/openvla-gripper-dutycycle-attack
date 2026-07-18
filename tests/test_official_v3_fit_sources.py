from __future__ import annotations

import csv
import copy
import importlib
import json
from pathlib import Path

import pytest


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_policy_and_privileged_contract_helpers_import() -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    assert module.POLICY_FIELDS[0] == "step"
    assert "object_state" in module.PRIVILEGED_FIELDS
    assert module.finite_list([1.0, 2.0], 2)
    assert not module.finite_list([1.0, float("nan")], 2)


def _policy_fixture(module):
    step = {"action_token_ids": [17]}
    intent = {
        "action_token_ids": [17],
        "score_head_summary": [{"top_token": 17}],
        "clean_policy_intent_9d": [0.0] * 9,
        "clean_open_probability_mass": 0.2,
        "clean_close_probability_mass": 0.7,
        "clean_top1_probability": 0.7,
        "clean_top1_is_open": False,
        "clean_top1_is_close": True,
        "clean_open_minus_close_log_mass": -1.0,
        "clean_action_token_entropy_normalized": 0.1,
        "clean_best_open_rank_normalized": 0.5,
        "clean_best_close_rank_normalized": 0.1,
        "generation_passes_per_step": 1,
        "single_generation_parity_pass": True,
        "score_adapter_parity_pass": True,
    }
    return step, intent


def _privileged_fixture():
    return {
        "canonical_parent_key": "libero_object/task_00/state_00",
        "suite": "libero_object",
        "task_idx": 0,
        "state_id": 0,
        "object_state": [0.0, 1.0],
        "mujoco_contact_pairs": [["robot0_gripper", "obj"]],
        "contact_count": 1,
        "contact_capture_valid": True,
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "robot0_eef_quat": [1.0, 0.0, 0.0, 0.0],
        "robot0_gripper_qpos": [0.0, 0.0],
        "eef_feature_pos": [0.0, 0.0, 0.0],
        "eef_alias_valid": True,
    }


def test_policy_record_validation_is_fail_closed() -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    step, intent = _policy_fixture(module)
    module.validate_policy_record(step, intent, "libero_object/task_00/state_00", 0)
    for field in ("clean_policy_intent_9d", "score_head_summary", "action_token_ids"):
        bad = copy.deepcopy(intent)
        if field == "clean_policy_intent_9d":
            bad[field] = [float("nan")] * 9
        elif field == "score_head_summary":
            bad[field] = [{"top_token": 99}]
        else:
            bad[field] = [18]
        with pytest.raises(ValueError):
            module.validate_policy_record(step, bad, "libero_object/task_00/state_00", 0)


def test_privileged_record_validation_checks_geometry_identity_and_contacts() -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    key = "libero_object/task_00/state_00"
    value = _privileged_fixture()
    module.validate_privileged_record(value, key, 0)
    cases = []
    bad = copy.deepcopy(value); bad["robot0_eef_quat"] = [1.0, 0.0, 0.0]; cases.append(bad)
    bad = copy.deepcopy(value); bad["contact_count"] = -1; cases.append(bad)
    bad = copy.deepcopy(value); bad["mujoco_contact_pairs"] = [["only_one_name"]]; cases.append(bad)
    bad = copy.deepcopy(value); bad["contact_count"] = 0; cases.append(bad)
    bad = copy.deepcopy(value); bad["state_id"] = 1; cases.append(bad)
    for invalid in cases:
        with pytest.raises(ValueError):
            module.validate_privileged_record(invalid, key, 0)


def test_privileged_task_conditional_dimensions_are_explicit() -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    rows = [{"canonical_parent_key": f"libero_object/task_{task:02d}/state_00", "object_state_dimensions": [28]} for task in range(40)]
    dimensions, passed = module.privileged_schema_dimensions(rows)
    assert len(dimensions) == 40
    assert passed is True
    dimensions["libero_object/task_00"] = [28, 56]
    assert all(len(value) == 1 for value in dimensions.values()) is False


def test_sealed_root_tamper_is_rejected(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    root = tmp_path / "sealed"
    root.mkdir()
    _write(root / "payload.txt", "immutable\n")
    module._write_recursive_seal(root)
    assert module.verify_sealed_root(root)
    (root / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_sealed_root(root)


def test_identity_parser_accepts_only_fit_rows(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.detector_v5.audit_official_v3_fit_sources")
    path = tmp_path / "registry.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_parent_key", "suite", "task_idx", "state_id", "split"])
        writer.writeheader()
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
            for task in range(10):
                for state in range(20):
                    writer.writerow({"canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}", "suite": suite, "task_idx": task, "state_id": state, "split": "FIT_TRAIN"})
        for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
            for task in range(10):
                for state in range(20, 50):
                    writer.writerow({"canonical_parent_key": f"{suite}/task_{task:02d}/state_{state:02d}", "suite": suite, "task_idx": task, "state_id": state, "split": "OTHER"})
    rows = module.load_fit(path)
    assert len(rows) == 800
    assert rows[0]["state_id"] == "0"
    assert rows[-1]["state_id"] == "19"
