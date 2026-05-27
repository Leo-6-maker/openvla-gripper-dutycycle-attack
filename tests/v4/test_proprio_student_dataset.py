from src.utils.proprio_causal_student import CATEGORICAL_FEATURES, NUMERIC_FEATURES, encode_dataset


def _rows():
    return [
        {
            "run_id": "r",
            "suite": "libero_goal",
            "task_id": "task_a",
            "task_name": "put bowl on plate",
            "state_id": "0",
            "seed": "0",
            "episode_key": "e0",
            "step_idx": "0",
            "mechanism_type": "pick_place_transfer",
            "parse_confidence": "high",
            "gripper_command": "-1",
            "gripper_qpos": "",
            "gripper_width": "0.1",
            "eef_x": "",
            "eef_y": "",
            "eef_z": "0.2",
            "eef_vx": "",
            "eef_vy": "",
            "eef_vz": "0.01",
            "action_dx": "0",
            "action_dy": "0",
            "action_dz": "0",
            "action_gripper": "-1",
            "recent_close_streak": "0",
            "recent_open_streak": "1",
            "recent_gripper_flip_count": "0",
            "normalized_step": "0.1",
            "teacher_phase": "other",
            "teacher_hazard": "false",
            "teacher_release_safe": "false",
            "teacher_confidence": "low",
        },
        {
            "run_id": "r",
            "suite": "libero_goal",
            "task_id": "task_b",
            "task_name": "turn on stove",
            "state_id": "1",
            "seed": "0",
            "episode_key": "e1",
            "step_idx": "1",
            "mechanism_type": "articulated_object",
            "parse_confidence": "medium",
            "gripper_command": "1",
            "gripper_qpos": "0.01",
            "gripper_width": "0.02",
            "eef_x": "",
            "eef_y": "",
            "eef_z": "0.4",
            "eef_vx": "",
            "eef_vy": "",
            "eef_vz": "0.02",
            "action_dx": "0",
            "action_dy": "0",
            "action_dz": "0",
            "action_gripper": "1",
            "recent_close_streak": "1",
            "recent_open_streak": "0",
            "recent_gripper_flip_count": "1",
            "normalized_step": "0.2",
            "teacher_phase": "carry",
            "teacher_hazard": "true",
            "teacher_release_safe": "false",
            "teacher_confidence": "high",
        },
    ]


def test_dataset_loading_handles_missing_numeric_and_categorical():
    data = encode_dataset(_rows(), split_mode="episode_key", seed=0)
    assert data.x.shape[0] == 2
    assert data.x.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert data.phase.shape[0] == 2
    assert data.hazard.tolist() == [0.0, 1.0]

