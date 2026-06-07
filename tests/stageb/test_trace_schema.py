"""Test: Stage-B v1 trace schema validation."""
REQUIRED_COLS = [
    'step', 'in_window', 'attack_this_step', 'env_grip', 'arm_l2',
    'pgd_applied', 'attacks_applied', 'gripper_qpos', 'done',
    'env_action_0', 'env_action_1', 'env_action_2', 'env_action_3',
    'env_action_4', 'env_action_5', 'env_action_6',
    'obs_gripper_qpos_0', 'obs_gripper_qpos_1', 'qpos_source',
    'trace_version', 'runner_version',
    # v1.1 additions
    'pair_id', 'condition', 'task_key', 'state_id',
    'window_start', 'window_end',
]

def test_validate_full_schema():
    cols = set(REQUIRED_COLS)
    missing = [c for c in ['env_action_0', 'pair_id', 'qpos_source', 'obs_gripper_qpos_0'] if c not in cols]
    assert len(missing) == 0, f'Missing critical columns: {missing}'
    print('PASS: test_validate_full_schema (%d columns)' % len(REQUIRED_COLS))

def test_reject_missing_pair_id():
    cols = {'step', 'env_grip', 'done', 'obs_gripper_qpos_0', 'qpos_source'}
    assert 'pair_id' not in cols
    missing = [c for c in REQUIRED_COLS if c not in cols]
    assert 'pair_id' in missing, 'pair_id should be flagged as missing'
    print('PASS: test_reject_missing_pair_id')

def test_qpos_source_must_be_correct():
    valid = {'obs_robot0_gripper_qpos'}
    invalid = {'env.sim.data.qpos[-2:]', 'raw_action_gripper', 'unknown'}
    for src in invalid:
        assert src not in valid, f'{src} should not be valid qpos_source'
    print('PASS: test_qpos_source_must_be_correct')

if __name__ == '__main__':
    test_validate_full_schema()
    test_reject_missing_pair_id()
    test_qpos_source_must_be_correct()
    print('All trace schema tests passed')
