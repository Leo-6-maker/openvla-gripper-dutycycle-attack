"""Test Stage-B OpenVLA-LIBERO open convention."""

from gripper_attack.openvla_libero_exec_spec import (
    env_gripper_is_close,
    env_gripper_is_open,
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_to_env_gripper,
)


def test_open_convention():
    assert env_gripper_is_open(-1.0), "env_action=-1.0 should be OPEN"
    assert not env_gripper_is_open(1.0), "env_action=+1.0 should be CLOSE"
    assert env_gripper_is_close(1.0), "env_action=+1.0 should be CLOSE"
    assert not env_gripper_is_close(-1.0), "env_action=-1.0 should be OPEN"


def test_raw_boundary_convention():
    assert raw_gripper_is_open(0.996)
    assert not raw_gripper_is_open(0.5)
    assert not raw_gripper_is_close(0.5)
    assert not raw_gripper_is_open(0.0)
    assert raw_gripper_to_env_gripper(0.996) == -1.0
    assert raw_gripper_to_env_gripper(0.5) == 0.0
    assert raw_gripper_to_env_gripper(0.0) == 1.0


if __name__ == "__main__":
    test_open_convention()
    test_raw_boundary_convention()
    print("All open convention tests passed")
