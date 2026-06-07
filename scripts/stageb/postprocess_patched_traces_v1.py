#!/usr/bin/env python3
"""Stage-B v1: Postprocess patched traces — recompute qpos from trace CSVs.
Ignores summary qpos_delta. Uses abs_sum, both shifted and unshifted."""
import csv, json, os, glob, sys
import numpy as np
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'src'))
from gripper_attack.gripper_semantics import env_gripper_is_open

qpos_rows, windows = [], defaultdict(dict)

for f in sorted(glob.glob(ROOT + '/trace_*.csv')):
    fname = os.path.basename(f).replace('.csv', '')
    # Determine condition
    cond = None
    for p in fname.split('_'):
        if p in ('vis_pgd', 'random_linf'): cond = p; break
    if cond is None: continue

    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    if 'obs_gripper_qpos_0' not in rows[0]: continue  # skip unpatched

    task = fname.split('_')[1] if len(fname.split('_')) > 1 else '?'
    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: continue
    ws = min(int(r['step']) for r in win_rows); we = max(int(r['step']) for r in win_rows)

    # Compute abs_qpos per step
    for r in rows:
        q0 = float(r.get('obs_gripper_qpos_0', 0)); q1 = float(r.get('obs_gripper_qpos_1', 0))
        r['qpos_abs_sum'] = abs(q0) + abs(q1); r['qpos_abs_mean'] = r['qpos_abs_sum'] / 2.0

    pre = [r for r in rows if ws - 3 <= int(r['step']) < ws]
    att = [r for r in win_rows]
    post = [r for r in rows if we < int(r['step']) <= we + 10]

    qpos_pre = np.mean([r['qpos_abs_sum'] for r in pre]) if pre else 0.0
    att_max_unshifted = max([r['qpos_abs_sum'] for r in att]) if att else 0.0
    post_max = max([r['qpos_abs_sum'] for r in post]) if post else 0.0

    # Shifted: action at t → qpos at t+1
    shifted = [rows[i+1]['qpos_abs_sum'] for i, r in enumerate(att)
               if i+1 < len(rows) and rows[i+1].get('in_window') == '1']
    att_max_shifted = max(shifted) if shifted else att_max_unshifted

    # OPEN convention: env_action_6 = -1.0 = OPEN (verified by env-only smoke)
    open_count = sum(1 for r in att if env_gripper_is_open(r.get('env_action_6', 0)))
    streak = max_streak = 0
    for r in att:
        if env_gripper_is_open(r.get('env_action_6', 0)): streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    delta_unshifted = att_max_unshifted - qpos_pre
    delta_shifted = att_max_shifted - qpos_pre
    delta_post = post_max - qpos_pre

    # Old signed delta from summary
    old_signed = 0.0
    sum_path = f.replace('trace_', 'summary_').replace('.csv', '.json')
    if os.path.exists(sum_path):
        with open(sum_path) as fh: old_signed = json.load(fh).get('qpos_delta', 0.0)

    cancelled = abs(old_signed) < 0.0001 and abs(delta_unshifted) > 0.001
    timing = delta_shifted > delta_unshifted + 0.001
    diag = []
    if cancelled: diag.append('SIGNED_MEAN_CANCELLATION')
    if timing: diag.append('PRESTEP_QPOS_TIMING_OFFSET')

    qpos_rows.append({
        'trace': fname, 'condition': cond, 'task_key': task,
        'window_start': str(ws), 'window_end': str(we),
        'qpos_pre': str(round(qpos_pre, 6)),
        'qpos_att_max_unshifted': str(round(att_max_unshifted, 6)),
        'qpos_att_max_shifted': str(round(att_max_shifted, 6)),
        'qpos_post_max': str(round(post_max, 6)),
        'delta_unshifted': str(round(delta_unshifted, 6)),
        'delta_shifted': str(round(delta_shifted, 6)),
        'delta_post': str(round(delta_post, 6)),
        'open_count': str(open_count), 'streak': str(max_streak),
        'old_signed_delta': str(round(old_signed, 6)),
        'diagnosis': '; '.join(diag) if diag else 'OK',
    })
    windows[(task, str(ws), str(we))][cond] = {
        'open_count': open_count, 'streak': max_streak,
        'delta_shifted': delta_shifted, 'delta_post': delta_post,
    }

# Write
out_csv = os.path.join(REPO, 'tables', 'stageb_v1_trace_metrics.csv')
if qpos_rows:
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(qpos_rows[0].keys()))
        w.writeheader(); w.writerows(qpos_rows)
    print('Wrote %d qpos rows' % len(qpos_rows))

# Write paired windows for label builder
win_csv = os.path.join(REPO, 'tables', 'stageb_v1_windows_for_labels.csv')
win_rows = []
for key, conds in windows.items():
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    task, ws, we = key; vs = conds['vis_pgd']; rs = conds['random_linf']
    win_rows.append({
        'task_key': task, 'window_start': ws, 'window_end': we,
        'vis_open': str(vs['open_count']), 'vis_streak': str(vs['streak']),
        'rand_open': str(rs['open_count']), 'rand_streak': str(rs['streak']),
        'vis_delta_shifted': str(round(vs['delta_shifted'], 6)),
        'rand_delta_shifted': str(round(rs['delta_shifted'], 6)),
        'vis_delta_post': str(round(vs['delta_post'], 6)),
    })
with open(win_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
    w.writeheader(); w.writerows(win_rows)
print('Wrote %d paired windows' % len(win_rows))
