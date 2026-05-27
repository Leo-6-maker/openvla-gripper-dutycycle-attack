"""Test deployment detector feature schema — normalized_step and privileged fields forbidden."""
import pytest, yaml
from pathlib import Path

SCHEMA = Path(__file__).parents[2] / "configs" / "deployment_detector_feature_schema.yaml"


def test_schema_loads():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    assert "allowed_deployment_inputs" in schema
    assert "forbidden_deployment_inputs" in schema


def test_normalized_step_forbidden():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    forbidden = schema["forbidden_deployment_inputs"]
    assert "normalized_step" in forbidden
    assert "absolute_timestep" in forbidden


def test_privileged_forbidden():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    forbidden = schema["forbidden_deployment_inputs"]
    assert "object_pose" in forbidden
    assert "target_pose" in forbidden
    assert "object_to_target_distance" in forbidden


def test_teacher_window_forbidden():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    forbidden = schema["forbidden_deployment_inputs"]
    assert "teacher_window_start" in forbidden
    assert "teacher_window_end" in forbidden
    assert "teacher_anchor_step" in forbidden


def test_attack_outcomes_forbidden():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    forbidden = schema["forbidden_deployment_inputs"]
    assert "attack_outcome" in forbidden
    assert "oracle_outcome" in forbidden
    assert "random_outcome" in forbidden
    assert "manual_outcome" in forbidden


def test_future_forbidden():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    forbidden = schema["forbidden_deployment_inputs"]
    assert "future_done" in forbidden
    assert "success" in forbidden


def test_allowed_groups():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    allowed = schema["allowed_deployment_inputs"]
    assert "gripper" in allowed
    assert "eef" in allowed
    assert "action_history" in allowed
    assert "visual" in allowed
    assert "task_language" in allowed


def test_model_versions():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    versions = schema["model_versions"]
    assert versions["full_proprio_dev"]["uses_normalized_step"] == True
    assert versions["proprio_no_step"]["uses_normalized_step"] == False
    assert versions["visual_proprio_no_step"]["uses_normalized_step"] == False


def test_no_step_in_deployment_allowed():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    allowed = schema["allowed_deployment_inputs"]
    all_fields = []
    for group, fields in allowed.items():
        all_fields.extend(fields)
    assert "normalized_step" not in all_fields
    assert "absolute_timestep" not in all_fields
