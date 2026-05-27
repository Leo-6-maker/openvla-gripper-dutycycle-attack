"""Test artifact-rich runner schema invariants."""
import pytest, yaml, json
from pathlib import Path

SCHEMA = Path(__file__).parents[2] / "configs" / "artifact_rich_official_eval_schema.yaml"


def test_schema_loads():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    assert "run_manifest" in schema
    assert "step_records" in schema
    assert "episode_records" in schema


def test_run_manifest_required():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    required = schema["run_manifest"]["required"]
    assert "run_id" in required
    assert "action_path" in required
    assert "unnorm_key" in required
    assert "artifact_complete" in required
    assert "suite" in required


def test_action_path_valid():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    valid = set(schema["run_manifest"]["action_path_values"])
    assert "generate_manual_decode" in valid
    assert "predict_action" in valid
    assert "unknown_needs_audit" in valid
    assert "random_string" not in valid


def test_step_records_required():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    required = schema["step_records"]["required"]
    assert "run_id" in required
    assert "image_path" in required
    assert "gripper_command" in required
    assert "gripper_qpos" in required
    assert "gripper_width" in required
    assert "eef_x" in required
    assert "eef_y" in required
    assert "eef_z" in required
    assert "eef_vx" in required
    assert "eef_vy" in required
    assert "eef_vz" in required
    assert "raw_action" in required
    assert "env_action" in required
    assert "action_gripper" in required
    assert "done" in required
    assert "step_idx" in required
    assert "policy_step_idx" in required
    assert "phase" in required
    assert "image_path_available" in required


def test_teacher_only_privileged_not_in_required():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    teacher_fields = set(schema["step_records"]["teacher_only_privileged"])
    required_fields = set(schema["step_records"]["required"])
    overlap = teacher_fields & required_fields
    assert len(overlap) == 0, f"Teacher-only fields should not be in required: {overlap}"


def test_worker_shard_path_format():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    path = schema["worker_manifest_shard"]["path"]
    assert "{worker_id}" in path


def test_aggregate_episode_key_format():
    with open(SCHEMA) as f:
        schema = yaml.safe_load(f)
    key = schema["aggregate_manifest"]["episode_key"]
    assert "suite" in key
    assert "task_id" in key
    assert "state_id" in key
    assert "seed" in key
    assert "run_id" in key
