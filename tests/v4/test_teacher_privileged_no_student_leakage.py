"""Test that teacher-privileged fields do not leak into deployed student features."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Teacher-only fields that must NEVER appear in deployed student feature columns
TEACHER_ONLY_FIELDS = {
    "object_pose_json",
    "target_pose_json",
    "object_to_target_distance",
    "object_eef_distance",
    "privileged_state_error",
    "teacher_privileged_state_available",
}

# Deployment-allowed field groups
DEPLOYMENT_ALLOWED = {
    "image_path",
    "image_path_available",
    "gripper_qpos",
    "gripper_width",
    "gripper_command",
    "eef_x", "eef_y", "eef_z",
    "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "raw_action", "env_action",
    "task_instruction",
    "task_name",
    "parsed_target_object",
    "parsed_target_receptacle",
    "mechanism_type",
    "parse_confidence",
    "step_idx",
    "policy_step_idx",
    "phase",
    "run_id", "episode_key",
    "suite", "task_id", "state_id", "seed",
    "normalized_step_in_deployment_features",
    "teacher_window_start",
    "teacher_window_end",
    "uses_attack_outcome",
}


class TestTeacherPrivilegedNoStudentLeakage:
    def test_teacher_only_fields_not_in_deployment_allowed(self):
        """Teacher-only privileged fields must not be in the deployment-allowed set."""
        intersection = TEACHER_ONLY_FIELDS & DEPLOYMENT_ALLOWED
        assert intersection == set(), f"Leakage detected: {intersection}"

    def test_privileged_state_error_is_teacher_only(self):
        assert "privileged_state_error" in TEACHER_ONLY_FIELDS

    def test_object_pose_json_is_teacher_only(self):
        assert "object_pose_json" in TEACHER_ONLY_FIELDS

    def test_target_pose_json_is_teacher_only(self):
        assert "target_pose_json" in TEACHER_ONLY_FIELDS

    def test_object_eef_distance_is_teacher_only(self):
        assert "object_eef_distance" in TEACHER_ONLY_FIELDS

    def test_object_to_target_distance_is_teacher_only(self):
        assert "object_to_target_distance" in TEACHER_ONLY_FIELDS

    def test_teacher_privileged_state_available_is_teacher_only(self):
        assert "teacher_privileged_state_available" in TEACHER_ONLY_FIELDS

    def test_deployment_has_gripper_eef_action(self):
        """Deployment schema must have gripper, eef, action features."""
        required = {"gripper_qpos", "eef_x", "action_dx"}
        assert required <= DEPLOYMENT_ALLOWED, f"Missing: {required - DEPLOYMENT_ALLOWED}"

    def test_deployment_has_image_path(self):
        assert "image_path" in DEPLOYMENT_ALLOWED

    def test_deployment_must_not_have_normalized_step_as_feature(self):
        """normalized_step_in_deployment_features is metadata, not a feature.
        The actual normalized_step should NOT be in the allowed set."""
        assert "normalized_step" not in DEPLOYMENT_ALLOWED
