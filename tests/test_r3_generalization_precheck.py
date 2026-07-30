import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_r3_generalization_precheck",
    ROOT / "scripts" / "detector_v5" / "run_r3_generalization_precheck.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_g3_requires_passing_g2_transition(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        MODULE._validate_transition(tmp_path)


def test_g3_active_and_inactive_heads_are_frozen():
    assert MODULE.ACTIVE_HEADS == ("physical_criticality", "k10_feasibility", "instability", "gripper_closing_state")
    assert MODULE.INACTIVE_HEADS == ("safe_release",)


def test_g3_exact_permission_boundary(tmp_path):
    transition = {
        "status": "PASS_G2_DEVELOPMENT_TRANSITION",
        "protected_reads": 0,
        "model_boundary": {
            "random_initialization_required": True,
            "all_670_engineering_checkpoint_allowed": False,
            "checkpoint_consumed": False,
            "privileged_oracle_nondeployable": True,
        },
        "permissions": dict(MODULE.EXPECTED_PERMISSION_MATRIX),
        "formal_training_authorized": False,
        "formal_inference_authorized": False,
        "shadow_offline_authorized": False,
        "shadow_live_authorized": False,
        "rollout_authorized": False,
        "attack_authorized": False,
        "teacher_privileged_fields_in_student": False,
        "consumable_for_scientific_promotion": False,
    }
    path = tmp_path / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V1.json"
    path.write_text(json.dumps(transition), encoding="utf-8")
    MODULE._write_seal(tmp_path)
    assert MODULE._validate_transition(tmp_path)["protected_reads"] == 0
    bad = dict(transition, permissions={**MODULE.EXPECTED_PERMISSION_MATRIX, "attack": True})
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE._write_seal(tmp_path)
        MODULE._validate_transition(tmp_path)
    bad_schema = dict(transition, schema="WRONG")
    path.write_text(json.dumps(bad_schema), encoding="utf-8")
    MODULE._write_seal(tmp_path)
    with pytest.raises(ValueError):
        MODULE._validate_transition(tmp_path)
    bad_type = dict(transition, permissions={**transition["permissions"], "attack": 0})
    path.write_text(json.dumps(bad_type), encoding="utf-8")
    MODULE._write_seal(tmp_path)
    with pytest.raises(ValueError):
        MODULE._validate_transition(tmp_path)
    bad_model = dict(transition, model_boundary={**transition["model_boundary"], "checkpoint_consumed": True})
    path.write_text(json.dumps(bad_model), encoding="utf-8")
    MODULE._write_seal(tmp_path)
    with pytest.raises(ValueError):
        MODULE._validate_transition(tmp_path)
    bad_zero = dict(transition, protected_reads=False)
    path.write_text(json.dumps(bad_zero), encoding="utf-8")
    MODULE._write_seal(tmp_path)
    with pytest.raises(ValueError):
        MODULE._validate_transition(tmp_path)


def test_g3_output_rejects_forbidden_path(tmp_path):
    with pytest.raises(ValueError):
        MODULE._output_root(tmp_path / "CAL", tmp_path)
