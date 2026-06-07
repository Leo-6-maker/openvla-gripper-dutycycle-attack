#!/usr/bin/env python3
"""Validate Stage-B v1 trace/summary output schema."""
import csv, json, os, sys, glob

REQUIRED_TRACE_COLS = [
    'step', 'in_window', 'attack_this_step', 'env_grip', 'arm_l2',
    'pgd_applied', 'attacks_applied', 'gripper_qpos', 'done',
    'env_action_0', 'env_action_1', 'env_action_2', 'env_action_3',
    'env_action_4', 'env_action_5', 'env_action_6',
    'obs_gripper_qpos_0', 'obs_gripper_qpos_1', 'qpos_source',
    'trace_version', 'runner_version',
]

def validate_trace(path):
    errors = []
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames)
        rows = list(reader)
    missing = [c for c in REQUIRED_TRACE_COLS if c not in cols]
    if missing: errors.append('MISSING_COLS: %s' % missing)
    if 'qpos_source' in cols:
        qs_vals = set(r.get('qpos_source', '') for r in rows)
        if qs_vals != {'obs_robot0_gripper_qpos'}:
            errors.append('BAD_QPOS_SOURCE: %s' % qs_vals)
    if 'obs_gripper_qpos_0' in cols:
        q0_vals = [float(r['obs_gripper_qpos_0']) for r in rows if r.get('in_window') == '1']
        if q0_vals and all(abs(v - 0.5) < 0.001 for v in q0_vals):
            errors.append('QPOS_PLACEHOLDER_05')
    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: errors.append('WINDOW_TOO_SHORT: %d rows' % len(win_rows))
    return len(errors) == 0, errors

def validate_summary(path):
    errors = []
    with open(path) as f: s = json.load(f)
    req_keys = ['task_key', 'state_id', 'window_start', 'window_end', 'condition',
                'decoded_open_count', 'infra_status', 'pair_id']
    missing = [k for k in req_keys if k not in s]
    if missing: errors.append('MISSING_KEYS: %s' % missing)
    return len(errors) == 0, errors

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    traces = sorted(glob.glob(d + '/trace_*.csv'))
    summaries = sorted(glob.glob(d + '/summary_*.json'))
    ok, fail = 0, 0
    for f in traces:
        passed, errs = validate_trace(f)
        if passed: ok += 1
        else: print('FAIL trace %s: %s' % (os.path.basename(f), errs)); fail += 1
    for f in summaries:
        passed, errs = validate_summary(f)
        if passed: ok += 1
        else: print('FAIL summary %s: %s' % (os.path.basename(f), errs)); fail += 1
    print('%d/%d passed, %d failed' % (ok, ok+fail, fail))
    sys.exit(0 if fail == 0 else 1)
