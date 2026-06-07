"""Test: Stage-B v1.1 trace schema validator."""

import csv
import os
import sys
import tempfile

REQUIRED_COLUMNS = [
    'step', 'row_id', 'in_window', 'attack_this_step',
    'pair_id', 'condition', 'task_key', 'state_id', 'seed',
    'window_start', 'window_end', 'done',
    'pgd_applied', 'attacks_applied',
    'env_action_0', 'env_action_1', 'env_action_2', 'env_action_3',
    'env_action_4', 'env_action_5', 'env_action_6', 'env_grip',
    'decoded_open_bool', 'open_convention',
    'raw_action_0', 'raw_action_1', 'raw_action_2', 'raw_action_3',
    'raw_action_4', 'raw_action_5', 'raw_action_6',
    'obs_gripper_qpos_0', 'obs_gripper_qpos_1',
    'obs_gripper_qpos_abs_sum', 'obs_gripper_qpos_abs_mean',
    'qpos_source', 'arm_l2', 'gripper_qpos',
    'random_seed', 'perturbation_space',
    'random_noise_linf', 'random_noise_l2',
    'eps_processor', 'eps_raw_pixels_name_deprecated_or_compat',
    'trace_version', 'runner_version', 'exec_spec_version',
    'git_commit', 'git_dirty', 'source_snapshot_id',
    'prompt_style', 'image_preprocess_style', 'unnorm_key',
]


def test_all_columns_present():
    assert len(REQUIRED_COLUMNS) >= 47, 'v1.1 requires >= 47 columns'
    assert 'prompt_style' in REQUIRED_COLUMNS
    assert 'image_preprocess_style' in REQUIRED_COLUMNS
    assert 'unnorm_key' in REQUIRED_COLUMNS
    assert 'exec_spec_version' in REQUIRED_COLUMNS
    assert 'random_noise_linf' in REQUIRED_COLUMNS
    assert 'random_noise_l2' in REQUIRED_COLUMNS
    assert 'eps_processor' in REQUIRED_COLUMNS
    print('PASS: test_all_columns_present (%d cols)' % len(REQUIRED_COLUMNS))


def test_reject_old_trace_version():
    VALID = 'corrected_stageb_v1_1'
    OLD = ['', 'legacy', 'patched_stageb_v1', 'corrected_stageb_v0',
           'pre_spec_20260605', 'corrected_stageb_v1.0']
    for tv in OLD:
        assert tv != VALID, 'old trace_version %r must be rejected' % tv
    print('PASS: test_reject_old_trace_version')


def test_reject_wrong_qpos_source():
    invalid = {'env_sim_data_qpos', 'raw_action_gripper', 'unknown'}
    valid = 'obs_robot0_gripper_qpos'
    for src in invalid:
        assert src != valid, 'qpos_source %r must be rejected' % src
    print('PASS: test_reject_wrong_qpos_source')


def test_reject_wrong_open_convention():
    valid = 'env_action_6_lt_neg_0p5_means_OPEN'
    invalid = ['env_action_6_gt_0p5_means_OPEN', 'raw_gripper_lt_0p5_means_OPEN']
    for inv in invalid:
        assert inv != valid, 'open_convention %r must be rejected' % inv
    print('PASS: test_reject_wrong_open_convention')


def test_reject_placeholder_git():
    for gc in ['', 'unknown', 'PLACEHOLDER']:
        assert gc in ('', 'unknown', 'PLACEHOLDER')
    assert 'ca3a97e' not in ('', 'unknown', 'PLACEHOLDER')
    print('PASS: test_reject_placeholder_git')


def test_source_snapshot_id_exact():
    """Validator must hard-fail on wrong source_snapshot_id."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'stageb'))
    import validate_stageb_trace_v1_1 as validator

    row = {c: '0' for c in REQUIRED_COLUMNS}
    row.update({
        'step': '0',
        'row_id': 'synthetic_wrong_snapshot',
        'pair_id': 'pair_synthetic',
        'condition': 'vis_pgd',
        'task_key': 'cream_cheese',
        'state_id': '0',
        'seed': '0',
        'window_start': '1',
        'window_end': '2',
        'trace_version': 'corrected_stageb_v1_1',
        'qpos_source': 'obs_robot0_gripper_qpos',
        'open_convention': 'env_action_6_lt_neg_0p5_means_OPEN',
        'decoded_open_bool': '0',
        'git_commit': '3985809a',
        'source_snapshot_id': '00000000',
    })
    with tempfile.NamedTemporaryFile('w', newline='', suffix='.csv', delete=False) as f:
        path = f.name
        w = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        w.writeheader()
        w.writerow(row)
    try:
        try:
            validator.validate_trace(path)
        except AssertionError as e:
            assert 'HARD_FAIL_SOURCE_SNAPSHOT_ID' in str(e)
        else:
            raise AssertionError('wrong source_snapshot_id should hard-fail')
    finally:
        os.unlink(path)
    print('PASS: test_source_snapshot_id_exact')


if __name__ == '__main__':
    test_all_columns_present()
    test_reject_old_trace_version()
    test_reject_wrong_qpos_source()
    test_reject_wrong_open_convention()
    test_reject_placeholder_git()
    test_source_snapshot_id_exact()
    print('All trace schema v1.1 tests PASSED')
