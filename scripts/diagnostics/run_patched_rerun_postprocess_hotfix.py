#!/usr/bin/env python3
"""Hotfix postprocess for current 44-row rerun.
Fixes: read condition/task from summary JSON (not filename),
       shifted qpos via step_idx dict, pair by (task,state,ws,we),
       open convention: env_action_6 < -0.5 = OPEN.
"""
import csv, json, os, glob, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = '/data/liuyu/outputs/stageb_selective_rerun_patched_20260607'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'src'))
from gripper_attack.gripper_semantics import env_gripper_is_open

# ── Load all summaries as ground truth for metadata ──────────────
summaries = {}
for f in glob.glob(ROOT + '/summary_*.json'):
    with open(f) as fh: s = json.load(fh)
    jid = str(s.get('job_id', ''))
    summaries[jid] = s

# ── Process traces ───────────────────────────────────────────────
qpos_rows, windows = [], defaultdict(dict)

for f in sorted(glob.glob(ROOT + '/trace_*.csv')):
    fname = os.path.basename(f).replace('.csv', '')
    # Extract job_id from filename
    jid = None
    for p in fname.split('_'):
        if p.startswith('job'): jid = p.replace('job', ''); break
    if jid is None: continue
    if int(jid) >= 9000 and int(jid) < 10000: continue  # skip smoke traces

    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    if 'obs_gripper_qpos_0' not in rows[0]: continue

    # Get metadata from SUMMARY, not filename
    s = summaries.get(jid, {})
    cond = s.get('condition', '')
    task = s.get('task_key', '')
    sid = str(s.get('state_id', '0'))
    ws = s.get('window_start', 0); we = s.get('window_end', 0)
    if not cond or not task: continue

    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: continue
    att_steps = [int(r['step']) for r in win_rows]

    # Build step_dict for shifted qpos lookup
    step_dict = {int(r['step']): r for r in rows}

    # Compute abs_qpos per step
    for r in rows:
        q0 = float(r.get('obs_gripper_qpos_0', 0)); q1 = float(r.get('obs_gripper_qpos_1', 0))
        r['qpos_abs_sum'] = abs(q0) + abs(q1)

    pre = [step_dict[s] for s in range(ws - 3, ws) if s in step_dict]
    att = [step_dict[s] for s in att_steps if s in step_dict]
    post = [step_dict[s] for s in range(we + 1, we + 11) if s in step_dict]

    qpos_pre = np.mean([r['qpos_abs_sum'] for r in pre]) if pre else 0.0
    att_max_unshifted = max([r['qpos_abs_sum'] for r in att]) if att else 0.0
    post_max = max([r['qpos_abs_sum'] for r in post]) if post else 0.0

    # Shifted: action at step t → qpos at step t+1
    shifted = [step_dict[s+1]['qpos_abs_sum'] for s in att_steps if s+1 in step_dict]
    att_max_shifted = max(shifted) if shifted else att_max_unshifted

    # Open count: env_action_6 < -0.5 = OPEN
    open_count = sum(1 for r in att if env_gripper_is_open(r.get('env_action_6', 0)))
    streak = max_streak = 0
    for r in att:
        if env_gripper_is_open(r.get('env_action_6', 0)): streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    delta_unshifted = att_max_unshifted - qpos_pre
    delta_shifted = att_max_shifted - qpos_pre
    delta_post = post_max - qpos_pre

    old_signed = s.get('qpos_delta', 0.0)
    cancelled = abs(old_signed) < 0.0001 and abs(delta_unshifted) > 0.001
    timing = delta_shifted > delta_unshifted + 0.001

    qpos_rows.append({
        'trace': fname, 'condition': cond, 'task_key': task, 'state_id': sid,
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
        'diagnosis': ('SIGNED_MEAN_CANCELLATION;' if cancelled else '') + ('PRESTEP_QPOS_TIMING_OFFSET' if timing else 'OK' if not cancelled else ''),
    })

    # Pair by (task, state, ws, we)
    key = (task, sid, str(ws), str(we))
    windows[key][cond] = {
        'open_count': open_count, 'streak': max_streak,
        'delta_shifted': delta_shifted, 'delta_post': delta_post,
    }

# ── Pair labels ──────────────────────────────────────────────────
label_rows = []
for key, conds in windows.items():
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    vs = conds['vis_pgd']; rs = conds['random_linf']
    task, sid, ws, we = key

    cmd_pos = (vs['open_count'] >= 6 or vs['streak'] >= 6) and not (rs['open_count'] >= 6 or rs['streak'] >= 6)
    rand_conf = (rs['open_count'] >= 6 or rs['streak'] >= 6)
    phys_sens = vs['delta_shifted'] >= 0.01 or vs['delta_post'] >= 0.01
    phys_strict = vs['delta_shifted'] >= 0.02 or vs['delta_post'] >= 0.02
    vis_spec = phys_sens and not (rs['delta_shifted'] >= 0.01)

    label_rows.append({
        'task_key': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'vis_open_count': str(vs['open_count']), 'vis_streak': str(vs['streak']),
        'rand_open_count': str(rs['open_count']), 'rand_streak': str(rs['streak']),
        'vis_qpos_delta_shifted': str(round(vs['delta_shifted'], 6)),
        'rand_qpos_delta_shifted': str(round(rs['delta_shifted'], 6)),
        'cmd_susceptible': str(int(cmd_pos)), 'random_confounded': str(int(rand_conf)),
        'physical_response_sensitive': str(int(phys_sens)),
        'physical_response_strict': str(int(phys_strict)),
        'vis_specific_physical_response': str(int(vis_spec)),
    })

# ── Write ────────────────────────────────────────────────────────
out_qpos = os.path.join(REPO, 'tables', 'stageb_selective_rerun_qpos_hotfix.csv')
out_labels = os.path.join(REPO, 'tables', 'stageb_selective_rerun_labels_hotfix.csv')

with open(out_qpos, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(qpos_rows[0].keys()))
    w.writeheader(); w.writerows(qpos_rows)
print('Qpos: %d rows' % len(qpos_rows))

with open(out_labels, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
    w.writeheader(); w.writerows(label_rows)
print('Labels: %d paired rows' % len(label_rows))

# ── Readout ──────────────────────────────────────────────────────
cmd_pos = sum(1 for r in label_rows if r['cmd_susceptible'] == '1')
rand_conf = sum(1 for r in label_rows if r['random_confounded'] == '1')
phys_sens = sum(1 for r in label_rows if r['physical_response_sensitive'] == '1')
vis_spec = sum(1 for r in label_rows if r['vis_specific_physical_response'] == '1')

print('\n=== HOTFIX READOUT ===')
print('Paired: %d' % len(label_rows))
print('cmd_susceptible: %d (%.1f%%)' % (cmd_pos, cmd_pos/max(len(label_rows),1)*100))
print('random_confounded: %d (%.1f%%)' % (rand_conf, rand_conf/max(len(label_rows),1)*100))
print('physical_response_sensitive: %d (%.1f%%)' % (phys_sens, phys_sens/max(len(label_rows),1)*100))
print('vis_specific_physical: %d' % vis_spec)

vis_qpos = [r for r in qpos_rows if r['condition'] == 'vis_pgd']
vis_deltas = [float(r['delta_shifted']) for r in vis_qpos]
print('VIS qpos_delta_shifted: mean=%.6f median=%.6f >0.01=%d >0.02=%d' % (
    np.mean(vis_deltas), np.median(vis_deltas),
    sum(1 for d in vis_deltas if d > 0.01), sum(1 for d in vis_deltas if d > 0.02)))

task_counts = Counter(r['task_key'] for r in label_rows)
for t, c in task_counts.most_common():
    tr = [r for r in label_rows if r['task_key'] == t]
    print('  %s: %d paired, %d cmd_pos, %d phys_sens' % (t, c,
        sum(1 for r in tr if r['cmd_susceptible']=='1'),
        sum(1 for r in tr if r['physical_response_sensitive']=='1')))
