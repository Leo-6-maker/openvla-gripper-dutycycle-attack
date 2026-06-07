#!/usr/bin/env python3
"""Stage-B v1.1 trace validator -- hard fail on schema violations.

Usage:
  python scripts/stageb/validate_stageb_trace_v1_1.py --trace trace_xxx.csv
  python scripts/stageb/validate_stageb_trace_v1_1.py --dir /path/to/outputs
"""
import csv, os, sys, argparse, glob

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
    'git_commit', 'git_dirty',
    'prompt_style', 'image_preprocess_style', 'unnorm_key',
]

MIN_TRACE_VERSION = 'corrected_stageb_v1_1'
VALID_QPOS_SOURCE = 'obs_robot0_gripper_qpos'
VALID_OPEN_CONVENTION = 'env_action_6_lt_neg_0p5_means_OPEN'


def validate_trace(path):
    """Validate a single trace CSV. Returns (True, nrows, nwin, pair_id) or raises."""
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)

    # 1. Required columns (HARD FAIL)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise AssertionError('HARD_FAIL_MISSING_COLUMNS: %s' % missing)

    # 2. trace_version (HARD FAIL)
    tv = rows[0].get('trace_version', '') if rows else ''
    if str(tv) != MIN_TRACE_VERSION:
        raise AssertionError(
            'HARD_FAIL_TRACE_VERSION: got=%r require=%s' % (tv, MIN_TRACE_VERSION))

    # 3. qpos_source (HARD FAIL)
    qs = set(r.get('qpos_source', '') for r in rows)
    if qs != {VALID_QPOS_SOURCE}:
        raise AssertionError('HARD_FAIL_QPOS_SOURCE: %s' % qs)

    # 4. open_convention (HARD FAIL)
    oc = set(r.get('open_convention', '') for r in rows)
    if oc != {VALID_OPEN_CONVENTION}:
        raise AssertionError('HARD_FAIL_OPEN_CONVENTION: %s' % oc)

    # 5. decoded_open_bool must be '0' or '1' (HARD FAIL)
    for r in rows:
        v = r.get('decoded_open_bool', '')
        if v not in ('0', '1'):
            raise AssertionError(
                'HARD_FAIL_DECODED_OPEN_BOOL: step=%s val=%r'
                % (r.get('step', '?'), v))

    # 6. git_commit not placeholder (HARD FAIL)
    gc = rows[0].get('git_commit', '') if rows else ''
    if gc in ('', 'unknown', 'PLACEHOLDER'):
        raise AssertionError('HARD_FAIL_GIT_COMMIT: %r' % gc)

    # 7. pair_id consistency
    pids = set(r.get('pair_id', '') for r in rows if r.get('pair_id'))
    if len(pids) != 1:
        raise AssertionError('INCONSISTENT_PAIR_IDS: %s' % pids)

    # 8. in_window steps (WARNING if empty, not hard fail -- rollout may end early)
    in_win = [r for r in rows if r.get('in_window') == '1']
    nwin = len(in_win)
    if nwin == 0:
        print('WARNING: no in_window steps -- rollout may have ended before window')

    return True, len(rows), nwin, pids.pop() if pids else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', help='Single trace CSV path')
    ap.add_argument('--dir', help='Directory of trace CSVs')
    args = ap.parse_args()

    files = []
    if args.trace:
        files = [args.trace]
    elif args.dir:
        files = sorted(glob.glob(os.path.join(args.dir, 'trace_*.csv')))
    else:
        print('FATAL: --trace or --dir required'); sys.exit(1)

    ok = 0; fail = 0
    for fp in files:
        try:
            valid, nrows, nwin, pid = validate_trace(fp)
            ok += 1
            print('PASS %-50s rows=%d window=%d pair=%s'
                  % (os.path.basename(fp), nrows, nwin, pid))
        except AssertionError as e:
            fail += 1
            print('FAIL %-50s %s' % (os.path.basename(fp), e))

    print('\nValid: %d  Failed: %d' % (ok, fail))
    if fail > 0:
        print('HARD_FAIL: %d traces rejected' % fail)
        sys.exit(1)


if __name__ == '__main__':
    main()
