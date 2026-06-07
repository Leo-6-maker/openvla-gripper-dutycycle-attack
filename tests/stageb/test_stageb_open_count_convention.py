from gripper_attack.openvla_libero_exec_spec import env_gripper_is_open, raw_gripper_is_open


def longest_streak(flags):
    cur = 0
    best = 0
    for flag in flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def test_env_action_open_count_uses_negative_one():
    env_actions = [-1.0, -1.0, 1.0, -1.0, 0.0]
    flags = [env_gripper_is_open(v) for v in env_actions]
    assert sum(flags) == 3
    assert longest_streak(flags) == 2


def test_raw_action_open_count_excludes_boundary():
    raw_actions = [0.996, 0.5, 0.499, 0.0, 0.8]
    flags = [raw_gripper_is_open(v) for v in raw_actions]
    assert flags == [True, False, False, False, True]
    assert sum(flags) == 2
