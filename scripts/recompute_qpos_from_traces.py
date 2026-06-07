#!/usr/bin/env python3
"""Recompute qpos from patched trace CSVs using abs_sum, fixing signed_mean cancellation."""
import csv, json, os, glob, sys
import numpy as np
from collections import defaultdict

TRACE_DIR = sys.argv[1] if len(sys.argv) > 1 else '/data/liuyu/outputs/stageb_selective_rerun_patched_20260607'
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260607'

# ── Load summaries ───────────────────────────────────────────────
summaries = []
for f in glob.glob(TRACE_DIR + '/summary_*.json'):
    with open(f) as fh: summaries.append(json.load(fh))
print('Summaries: %d' % len(summaries))

# ── Process each trace ───────────────────────────────────────────
qpos_rows = []
label_rows = []
windows = defaultdict(dict)

for f in sorted(glob.glob(TRACE_DIR + '/trace_*.csv')):
    try:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
    except: continue

    # Extract key info from filename
    fname = os.path.basename(f).replace('.csv', '')
    parts = fname.split('_')
    # Format: trace_<task>_<condition>_job<jobid>.csv
    cond = None
    for p in parts:
        if p in ('vis_pgd', 'random_linf'): cond = p; break
    if cond is None: continue

    task = parts[1] if len(parts) > 1 else '?'
    jid_str = [p for p in parts if p.startswith('job')][0] if any(p.startswith('job') for p in parts) else 'job?'
    jid = jid_str.replace('job', '')

    # Get window info from rows
    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: continue

    # Step numbers
    all_steps = [int(r['step']) for r in rows]
    win_steps = [int(r['step']) for r in win_rows]
    ws = min(win_steps); we = max(win_steps)

    # Compute abs_qpos for each step
    for r in rows:
        q0 = float(r.get('obs_gripper_qpos_0', 0))
        q1 = float(r.get('obs_gripper_qpos_1', 0))
        gq = float(r.get('gripper_qpos', 0))
        r['qpos_abs_sum'] = abs(q0) + abs(q1)
        r['qpos_abs_mean'] = r['qpos_abs_sum'] / 2.0
        r['qpos_signed_mean'] = gq  # the old field

    # Pre-attack qpos (3 steps before window)
    pre_rows = [r for r in rows if ws - 3 <= int(r['step']) < ws]
    pre_abs = np.mean([r['qpos_abs_mean'] for r in pre_rows]) if pre_rows else 0.0

    # Attack window qpos
    attack_rows = [r for r in win_rows if r.get('attack_this_step') == '1']
    attack_max = max([r['qpos_abs_mean'] for r in attack_rows]) if attack_rows else float(np.mean([r['qpos_abs_mean'] for r in win_rows]))

    # Post-attack qpos (up to 10 steps after window)
    post_rows = [r for r in rows if we < int(r['step']) <= we + 10]
    post_max = max([r['qpos_abs_mean'] for r in post_rows]) if post_rows else 0.0

    # Deltas
    delta_attack_abs = attack_max - pre_abs
    delta_post_abs = post_max - pre_abs

    # Old signed delta from summary
    old_delta = 0.0
    for s in summaries:
        if str(s.get('job_id', '')) == jid and s.get('condition') == cond:
            old_delta = s.get('qpos_delta', 0.0)
            break

    cancelled = abs(old_delta) < 0.0001 and abs(delta_attack_abs) > 0.001
    placeholder = all(abs(float(r.get('gripper_qpos', 0.5)) - 0.5) < 0.001 for r in rows)

    qpos_rows.append({
        'trace_file': fname, 'condition': cond,
        'task_key': task, 'job_id': jid,
        'window_start': str(ws), 'window_end': str(we),
        'n_pre_steps': str(len(pre_rows)), 'n_attack_steps': str(len(attack_rows)),
        'n_post_steps': str(len(post_rows)),
        'qpos_pre_abs_mean': str(round(pre_abs, 6)),
        'qpos_attack_max_abs_mean': str(round(attack_max, 6)),
        'qpos_post_max_abs_mean': str(round(post_max, 6)),
        'qpos_delta_attack_abs': str(round(delta_attack_abs, 6)),
        'qpos_delta_post_abs': str(round(delta_post_abs, 6)),
        'qpos_signed_mean_delta_old': str(round(old_delta, 6)),
        'qpos_placeholder_flag': str(placeholder),
        'qpos_cancelled_by_signed_mean': str(cancelled),
        'diagnosis': 'SIGNED_MEAN_CANCELLATION' if cancelled else ('PLACEHOLDER' if placeholder else 'OK'),
    })

    # Build label row
    win_key = (task, str(ws), str(we))  # approximate
    windows[(task, str(ws), str(we))][cond] = {
        'abs_pre': pre_abs, 'abs_attack_max': attack_max,
        'abs_post_max': post_max, 'delta_attack': delta_attack_abs,
        'delta_post': delta_post_abs,
    }

# ── Pair labels ──────────────────────────────────────────────────
for key, conds in windows.items():
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    vs = conds['vis_pgd']; rs = conds['random_linf']
    task, ws, we = key

    # Load decoded open from summary
    vis_open = 0; vis_streak = 0; rand_open = 0; rand_streak = 0
    for s in summaries:
        sk = (s['task_key'], str(s['window_start']), str(s['window_end']))
        if sk == key:
            if s['condition'] == 'vis_pgd':
                vis_open = s.get('decoded_open_count', 0)
                vis_streak = s.get('decoded_longest_open_streak', 0)
            elif s['condition'] == 'random_linf':
                rand_open = s.get('decoded_open_count', 0)
                rand_streak = s.get('decoded_longest_open_streak', 0)

    cmd_pos = (vis_open >= 6 or vis_streak >= 6) and not (rand_open >= 6 or rand_streak >= 6)
    phys_resp = vs['delta_attack'] > 0.005  # 0.005 = ~0.01mm threshold for physical response

    label_rows.append({
        'task_key': task, 'window_start': ws, 'window_end': we,
        'vis_open_count': str(vis_open), 'vis_streak': str(vis_streak),
        'rand_open_count': str(rand_open), 'rand_streak': str(rand_streak),
        'cmd_susceptible': str(int(cmd_pos)),
        'physical_response': str(int(phys_resp)),
        'qpos_delta_attack_abs': str(round(vs['delta_attack'], 6)),
        'qpos_delta_post_abs': str(round(vs['delta_post'], 6)),
        'qpos_pre_abs_mean': str(round(vs['abs_pre'], 6)),
        'qpos_attack_max_abs': str(round(vs['abs_attack_max'], 6)),
    })

# ── Write ────────────────────────────────────────────────────────
out_qpos = os.path.join(REPO, 'tables', 'stageb_selective_rerun_qpos_recomputed_from_trace.csv')
out_labels = os.path.join(REPO, 'tables', 'stageb_selective_rerun_patched_pair_labels.csv')

if qpos_rows:
    with open(out_qpos, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(qpos_rows[0].keys()))
        w.writeheader(); w.writerows(qpos_rows)
    print('Wrote %d qpos rows to %s' % (len(qpos_rows), out_qpos))

if label_rows:
    with open(out_labels, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        w.writeheader(); w.writerows(label_rows)
    print('Wrote %d label rows to %s' % (len(label_rows), out_labels))

# ── Quick audit ──────────────────────────────────────────────────
cancelled = sum(1 for r in qpos_rows if r['diagnosis'] == 'SIGNED_MEAN_CANCELLATION')
placeholder = sum(1 for r in qpos_rows if r['diagnosis'] == 'PLACEHOLDER')
ok_count = sum(1 for r in qpos_rows if r['diagnosis'] == 'OK')
print('\nQpos audit: cancelled=%d placeholder=%d ok=%d' % (cancelled, placeholder, ok_count))

if label_rows:
    cmd_pos_count = sum(1 for r in label_rows if r['cmd_susceptible'] == '1')
    phys_pos_count = sum(1 for r in label_rows if r['physical_response'] == '1')
    print('Labels: cmd_pos=%d phys_pos=%d (out of %d)' % (cmd_pos_count, phys_pos_count, len(label_rows)))
