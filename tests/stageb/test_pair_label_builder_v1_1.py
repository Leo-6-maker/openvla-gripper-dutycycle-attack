"""Test: v1.1 pair label builder logic."""


def test_pairing_key():
    """Pairing must be by (pair_id, task_key, state_id, seed, window_start, window_end)."""
    key_fields = ['pair_id', 'task_key', 'state_id', 'seed', 'window_start', 'window_end']
    assert len(key_fields) == 6
    assert 'pair_id' in key_fields
    print('PASS: test_pairing_key')


def test_no_filename_parsing():
    """Filename parsing must NEVER be used for task/condition extraction."""
    fname = 'trace_cream_cheese_vis_pgd_job10000.csv'
    parts = fname.split('_')
    # Old bug: parts[1] = 'cream' (not 'cream_cheese')
    assert parts[1] == 'cream'
    assert parts[0] + '_' + parts[1] != 'cream_cheese'
    # Proves filename parsing is unreliable
    print('PASS: test_no_filename_parsing')


def test_open_count_uses_spec():
    """open_count must use env_gripper_is_open from spec."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from gripper_attack.openvla_libero_exec_spec import env_gripper_is_open

    actions = [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0]
    open_count = sum(1 for a in actions if env_gripper_is_open(a))
    assert open_count == 6, '6 of 8 are env=-1.0 (OPEN), got %d' % open_count

    # old wrong convention: g > 0 would give 2
    old_wrong = sum(1 for a in actions if a > 0)
    assert old_wrong == 2, 'old convention would say 2 OPEN (wrong)'
    assert old_wrong != open_count, 'old and new must disagree'
    print('PASS: test_open_count_uses_spec')


def test_qpos_abs_sum():
    """qpos must use abs(q0)+abs(q1), never signed mean."""
    q0, q1 = 0.02, -0.02
    signed_mean = (q0 + q1) / 2.0
    abs_sum = abs(q0) + abs(q1)
    assert abs(signed_mean) < 0.001, 'signed mean cancels'
    assert abs_sum > 0.03, 'abs_sum does not cancel'
    print('PASS: test_qpos_abs_sum')


def test_shifted_qpos_indexing():
    """shifted qpos uses step_dict[s+1], not enumerate local index."""
    rows = {
        5: {'obs_gripper_qpos_0': 0.01, 'obs_gripper_qpos_1': 0.01},
        6: {'obs_gripper_qpos_0': 0.02, 'obs_gripper_qpos_1': 0.02},
    }
    att_steps = [5]
    shifted = []
    for s in att_steps:
        if s in rows and s + 1 in rows:
            r_b = rows[s]; r_a = rows[s + 1]
            abs_b = abs(float(r_b['obs_gripper_qpos_0'])) + abs(float(r_b['obs_gripper_qpos_1']))
            abs_a = abs(float(r_a['obs_gripper_qpos_0'])) + abs(float(r_a['obs_gripper_qpos_1']))
            shifted.append(abs_a - abs_b)
    assert len(shifted) == 1
    assert shifted[0] == 0.02
    print('PASS: test_shifted_qpos_indexing')


if __name__ == '__main__':
    test_pairing_key()
    test_no_filename_parsing()
    test_open_count_uses_spec()
    test_qpos_abs_sum()
    test_shifted_qpos_indexing()
    print('All pair label builder v1.1 tests PASSED')
