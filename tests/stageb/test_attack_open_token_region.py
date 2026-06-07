from gripper_attack.openvla_libero_exec_spec import (
    close_token_ids_from_decoded_action,
    boundary_token_ids_from_decoded_action,
    env_gripper_is_close,
    env_gripper_is_open,
    open_token_ids_from_decoded_action,
    raw_gripper_to_env_gripper,
    validate_open_close_token_sets,
)


def test_attack_open_token_region_classifies_physical_open_side():
    token_action_map = {
        100: 0.0,
        101: 0.25,
        102: 0.499,
        103: 0.5,
        104: 0.75,
        31744: 0.996,
        31745: 1.0,
    }
    open_ids = open_token_ids_from_decoded_action(token_action_map)
    close_ids = close_token_ids_from_decoded_action(token_action_map)
    boundary_ids = boundary_token_ids_from_decoded_action(token_action_map)

    assert set(open_ids) == {104, 31744, 31745}
    assert set(close_ids) == {100, 101, 102}
    assert set(boundary_ids) == {103}
    validate_open_close_token_sets(open_ids, close_ids, token_action_map)

    for tid in open_ids:
        assert env_gripper_is_open(raw_gripper_to_env_gripper(token_action_map[tid]))
    for tid in close_ids:
        assert env_gripper_is_close(raw_gripper_to_env_gripper(token_action_map[tid]))
