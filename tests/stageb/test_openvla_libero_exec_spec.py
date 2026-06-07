import numpy as np

from gripper_attack.openvla_libero_exec_spec import (
    OFFICIAL_PROMPT_STYLE,
    OPEN_THRESHOLD_RAW,
    close_token_ids_from_decoded_action,
    env_gripper_is_close,
    env_gripper_is_open,
    official_prompt,
    open_token_ids_from_decoded_action,
    raw_gripper_is_close,
    raw_gripper_is_open,
    raw_gripper_to_env_gripper,
    validate_open_close_token_sets,
)


def test_raw_to_env_truth_table():
    assert raw_gripper_to_env_gripper(0.996) == -1.0
    assert env_gripper_is_open(-1.0)
    assert raw_gripper_is_open(0.996)

    assert raw_gripper_to_env_gripper(0.0) == 1.0
    assert env_gripper_is_close(1.0)
    assert raw_gripper_is_close(0.0)

    assert raw_gripper_to_env_gripper(0.5) == -1.0
    assert raw_gripper_is_open(0.5)


def test_official_prompt_shape():
    prompt = official_prompt("pick up the ketchup")
    assert prompt == "In: What action should the robot take to pick up the ketchup?\nOut:"
    assert OFFICIAL_PROMPT_STYLE == "official_in_out"


def test_token_helpers_follow_raw_threshold():
    token_action_map = {10: 0.0, 11: 0.499, 12: 0.5, 13: 0.996, 31744: 1.0, 31745: 0.999}
    open_ids = open_token_ids_from_decoded_action(token_action_map)
    close_ids = close_token_ids_from_decoded_action(token_action_map)
    assert open_ids == [12, 13, 31744, 31745]
    assert close_ids == [10, 11]
    validate_open_close_token_sets(open_ids, close_ids, token_action_map)


def test_image_rotation_only():
    from gripper_attack.openvla_libero_exec_spec import get_libero_image_official

    img = np.arange(12).reshape(2, 2, 3)
    out = get_libero_image_official({"agentview_image": img})
    assert np.array_equal(out, img[::-1, ::-1])
    assert OPEN_THRESHOLD_RAW == 0.5
