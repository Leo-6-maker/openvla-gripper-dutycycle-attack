#!/usr/bin/env python3
"""Steps 3A-3D: Merge patched traces, recompute qpos, pair labels, mechanism readout."""
import csv, json, os, glob, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = '/data/liuyu/outputs/stageb_selective_rerun_patched_20260607'
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260607'
QUEUE_CSV = os.path.join(REPO, 'tables', 'stageb_selective_rerun_queue.csv')

# ── Step 3A: Strict manifest ─────────────────────────────────────
print('=== 3A: Building strict manifest ===')
summaries = []
for f in glob.glob(ROOT + '/summary_*.json'):
    with open(f) as fh: summaries.append(json.load(fh))

# Load rerun queue for validation
with open(QUEUE_CSV) as f:
    queue_rows = list(csv.DictReader(f))
queue_keys = set()
for r in queue_rows:
    queue_keys.add((r['task_key'].strip(), r['state_id'].strip(),
                    int(r['window_start']), int(r['window_end'])))

manifest = []
traces_ok = 0; traces_rejected = 0
for f in sorted(glob.glob(ROOT + '/trace_*.csv')):
    fname = os.path.basename(f).replace('.csv', '')
    jid_str = None
    for p in fname.split('_'):
        if p.startswith('job'): jid_str = p.replace('job', ''); break
    if jid_str is None: continue
    jid = int(jid_str)

    # Exclude non-44-row traces (job9xxx)
    if jid >= 9000 and jid < 10000:
        traces_rejected += 1; continue

    # Exclude old unpatched traces (no env_action_0)
    with open(f) as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames
        rows = list(reader)
    if 'env_action_0' not in cols or 'obs_gripper_qpos_0' not in cols:
        traces_rejected += 1; continue

    # Determine condition
    cond = None
    for p in fname.split('_'):
        if p in ('vis_pgd', 'random_linf', 'clean'): cond = p; break
    if cond is None: traces_rejected += 1; continue

    # Get task from rows
    task = rows[0].get('task_key', '') if 'task_key' in cols else '?'
    if not task:
        # Try from filename
        parts = fname.split('_')
        task = parts[1] if len(parts) > 1 else '?'

    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: traces_rejected += 1; continue
    ws = min(int(r['step']) for r in win_rows)
    we = max(int(r['step']) for r in win_rows)

    # Verify against queue
    key = (task, str(rows[0].get('state_id', '0')), ws, we) if 'state_id' in cols else None
    # Fuzzy match by task+ws+we
    match = None
    for qk in queue_keys:
        if qk[0] == task and abs(qk[2] - ws) <= 2 and abs(qk[3] - we) <= 2:
            match = qk; break

    summary_path = os.path.join(ROOT, fname.replace('trace_', 'summary_') + '.json')
    if not os.path.exists(summary_path): summary_path = ''

    manifest.append({
        'trace_path': f, 'summary_path': summary_path,
        'worker': '', 'condition': cond,
        'task_key': task, 'state_id': str(rows[0].get('state_id', '0')),
        'window_start': str(ws), 'window_end': str(we),
        'trace_version': 'patched',
        'mapping_status': 'OK' if match else 'WINDOW_MISMATCH',
        'exclude_reason': '' if match else 'window_not_in_queue_+/-2',
    })
    traces_ok += 1

print('Traces: %d ok, %d rejected' % (traces_ok, traces_rejected))

# Write manifest
MAN_CSV = os.path.join(REPO, 'tables', 'stageb_selective_rerun_patched_manifest.csv')
if manifest:
    with open(MAN_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)
    print('Wrote manifest: %d rows' % len(manifest))

# ── Step 3B: Recompute qpos from traces ──────────────────────────
print('\n=== 3B: Recomputing qpos ===')
qpos_rows = []
windows = defaultdict(dict)

for m in manifest:
    trace_path = m['trace_path']; cond = m['condition']
    task = m['task_key']; ws = int(m['window_start']); we = int(m['window_end'])

    with open(trace_path) as f:
        rows = list(csv.DictReader(f))

    win_rows = [r for r in rows if r.get('in_window') == '1']
    if len(win_rows) < 2: continue

    # Compute per-step abs qpos
    for i, r in enumerate(rows):
        q0 = float(r.get('obs_gripper_qpos_0', 0))
        q1 = float(r.get('obs_gripper_qpos_1', 0))
        gq = float(r.get('gripper_qpos', 0))
        r['qpos_abs_sum'] = abs(q0) + abs(q1)
        r['qpos_abs_mean'] = r['qpos_abs_sum'] / 2.0
        r['qpos_signed_mean'] = gq
        r['env_action_gripper'] = float(r.get('env_action_6', 0))
        r['decoded_open_bool'] = 1 if r['env_action_gripper'] > 0.5 else 0  # env_action[-1]=+1=OPEN

    # Pre (3 steps before), attack window, post (10 steps after)
    pre_steps = [r for r in rows if ws - 3 <= int(r['step']) < ws]
    attack_steps = [r for r in win_rows]
    post_steps = [r for r in rows if we < int(r['step']) <= we + 10]

    # Metrics
    n_pre = len(pre_steps); n_att = len(attack_steps); n_post = len(post_steps)

    qpos_pre = np.mean([r['qpos_abs_sum'] for r in pre_steps]) if pre_steps else 0.0
    qpos_att_max_unshifted = max([r['qpos_abs_sum'] for r in attack_steps]) if attack_steps else 0.0
    qpos_post_max = max([r['qpos_abs_sum'] for r in post_steps]) if post_steps else 0.0

    # Shifted: action at step t affects qpos at step t+1
    attack_qpos_shifted = []
    for i, r in enumerate(attack_steps):
        step_num = int(r['step'])
        # Find next step's qpos
        next_rows = [r2 for r2 in rows if int(r2['step']) == step_num + 1]
        if next_rows:
            attack_qpos_shifted.append(next_rows[0]['qpos_abs_sum'])
    qpos_att_max_shifted = max(attack_qpos_shifted) if attack_qpos_shifted else qpos_att_max_unshifted

    delta_unshifted = qpos_att_max_unshifted - qpos_pre
    delta_shifted = qpos_att_max_shifted - qpos_pre
    delta_post = qpos_post_max - qpos_pre

    # Open count from trace
    open_count = sum(1 for r in attack_steps if r['decoded_open_bool'] == 1)
    streak = 0; max_streak = 0
    for r in attack_steps:
        if r['decoded_open_bool'] == 1: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    # Old signed delta from summary
    old_signed = 0.0
    sum_path = m['summary_path']
    if sum_path and os.path.exists(sum_path):
        with open(sum_path) as fh:
            s = json.load(fh)
        old_signed = s.get('qpos_delta', 0.0)

    cancelled = abs(old_signed) < 0.0001 and abs(delta_unshifted) > 0.001
    timing_offset = delta_shifted > delta_unshifted + 0.001

    diagnoses = []
    if cancelled: diagnoses.append('SIGNED_MEAN_CANCELLATION')
    if timing_offset: diagnoses.append('PRESTEP_QPOS_TIMING_OFFSET')

    qpos_rows.append({
        'trace_path': os.path.basename(trace_path), 'condition': cond,
        'task_key': task, 'window_start': str(ws), 'window_end': str(we),
        'n_pre': str(n_pre), 'n_attack': str(n_att), 'n_post': str(n_post),
        'qpos_pre_abs_sum': str(round(qpos_pre, 6)),
        'qpos_attack_max_unshifted': str(round(qpos_att_max_unshifted, 6)),
        'qpos_attack_max_shifted': str(round(qpos_att_max_shifted, 6)),
        'qpos_post_max': str(round(qpos_post_max, 6)),
        'qpos_delta_unshifted': str(round(delta_unshifted, 6)),
        'qpos_delta_shifted': str(round(delta_shifted, 6)),
        'qpos_delta_post': str(round(delta_post, 6)),
        'open_count': str(open_count), 'longest_open_streak': str(max_streak),
        'old_signed_delta': str(round(old_signed, 6)),
        'diagnosis': '; '.join(diagnoses) if diagnoses else 'OK',
    })

    # Store for pairing
    win_key = (task, str(ws), str(we))
    windows[win_key][cond] = {
        'open_count': open_count, 'streak': max_streak,
        'delta_shifted': delta_shifted, 'delta_post': delta_post,
    }

QPOS_CSV = os.path.join(REPO, 'tables', 'stageb_selective_rerun_qpos_recomputed_from_trace.csv')
with open(QPOS_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(qpos_rows[0].keys()))
    w.writeheader(); w.writerows(qpos_rows)
print('Wrote qpos: %d rows' % len(qpos_rows))

# ── Step 3C: Pair labels ────────────────────────────────────────
print('\n=== 3C: Pairing labels ===')
label_rows = []

for key, conds in windows.items():
    if 'vis_pgd' not in conds or 'random_linf' not in conds: continue
    vs = conds['vis_pgd']; rs = conds['random_linf']
    task, ws, we = key

    cmd_pos = (vs['open_count'] >= 6 or vs['streak'] >= 6) and not (rs['open_count'] >= 6 or rs['streak'] >= 6)
    rand_conf = (rs['open_count'] >= 6 or rs['streak'] >= 6)
    phys_sens = vs['delta_shifted'] >= 0.01 or vs['delta_post'] >= 0.01
    phys_strict = vs['delta_shifted'] >= 0.02 or vs['delta_post'] >= 0.02
    vis_specific_phys = phys_sens and not (rs['delta_shifted'] >= 0.01 or rs['delta_post'] >= 0.01)

    label_rows.append({
        'task_key': task, 'window_start': ws, 'window_end': we,
        'vis_open_count': str(vs['open_count']), 'vis_streak': str(vs['streak']),
        'rand_open_count': str(rs['open_count']), 'rand_streak': str(rs['streak']),
        'vis_qpos_delta_shifted': str(round(vs['delta_shifted'], 6)),
        'rand_qpos_delta_shifted': str(round(rs['delta_shifted'], 6)),
        'vis_qpos_delta_post': str(round(vs['delta_post'], 6)),
        'cmd_susceptible': str(int(cmd_pos)),
        'random_confounded': str(int(rand_conf)),
        'physical_response_sensitive': str(int(phys_sens)),
        'physical_response_strict': str(int(phys_strict)),
        'vis_specific_physical_response': str(int(vis_specific_phys)),
    })

LAB_CSV = os.path.join(REPO, 'tables', 'stageb_selective_rerun_patched_pair_labels.csv')
with open(LAB_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
    w.writeheader(); w.writerows(label_rows)
print('Wrote labels: %d paired rows' % len(label_rows))

# ── Step 3D: Mechanism readout ───────────────────────────────────
print('\n=== 3D: Mechanism readout ===')
lines = []
lines.append('# Stage-B Patched Rerun — Mechanism Readout')
lines.append('')
lines.append('**Paired rows**: %d' % len(label_rows))
lines.append('')

# Summary counts
cmd_pos = sum(1 for r in label_rows if r['cmd_susceptible'] == '1')
rand_conf = sum(1 for r in label_rows if r['random_confounded'] == '1')
phys_sens = sum(1 for r in label_rows if r['physical_response_sensitive'] == '1')
phys_strict = sum(1 for r in label_rows if r['physical_response_strict'] == '1')
vis_spec = sum(1 for r in label_rows if r['vis_specific_physical_response'] == '1')
hard_neg_phys = sum(1 for r in label_rows if r['cmd_susceptible'] == '0' and r['random_confounded'] == '0' and r['physical_response_sensitive'] == '1')
rand_phys = sum(1 for r in label_rows if r['random_confounded'] == '1' and r['physical_response_sensitive'] == '1')

lines.append('## Label Distribution')
lines.append('')
lines.append('| Label | Count | Rate |')
lines.append('|---|---|---|')
lines.append('| cmd_susceptible | %d | %.1f%% |' % (cmd_pos, cmd_pos/max(len(label_rows),1)*100))
lines.append('| random_confounded | %d | %.1f%% |' % (rand_conf, rand_conf/max(len(label_rows),1)*100))
lines.append('| phys_response_sensitive | %d | %.1f%% |' % (phys_sens, phys_sens/max(len(label_rows),1)*100))
lines.append('| phys_response_strict | %d | %.1f%% |' % (phys_strict, phys_strict/max(len(label_rows),1)*100))
lines.append('| vis_specific_physical | %d | %.1f%% |' % (vis_spec, vis_spec/max(len(label_rows),1)*100))
lines.append('| hard_neg_phys_response | %d | |' % hard_neg_phys)
lines.append('| rand_conf_phys_response | %d | |' % rand_phys)
lines.append('')

lines.append('## Per-Task Distribution')
lines.append('')
lines.append('| Task | Paired | cmd_pos | phys_sens | vis_spec |')
lines.append('|---|---|---|---|---|')
task_counts = Counter(r['task_key'] for r in label_rows)
for t, c in task_counts.most_common():
    tr = [r for r in label_rows if r['task_key'] == t]
    lines.append('| %s | %d | %d | %d | %d |' % (t, c,
        sum(1 for r in tr if r['cmd_susceptible']=='1'),
        sum(1 for r in tr if r['physical_response_sensitive']=='1'),
        sum(1 for r in tr if r['vis_specific_physical_response']=='1')))
lines.append('')

lines.append('## Qpos Diagnosis')
lines.append('')
diag = Counter(r['diagnosis'] for r in qpos_rows if r['diagnosis'] != 'OK')
for d, c in diag.most_common():
    lines.append('- %s: %d traces' % (d, c))
lines.append('- OK: %d traces' % sum(1 for r in qpos_rows if r['diagnosis']=='OK'))
lines.append('')

# Qpos distribution
vis_qpos = [r for r in qpos_rows if r['condition'] == 'vis_pgd']
rand_qpos = [r for r in qpos_rows if r['condition'] == 'random_linf']
vis_deltas = [float(r['qpos_delta_shifted']) for r in vis_qpos]
rand_deltas = [float(r['qpos_delta_shifted']) for r in rand_qpos]
lines.append('## Qpos Delta Distribution (shifted)')
lines.append('')
lines.append('| Condition | N | Mean | Median | Min | Max | >0.01 | >0.02 |')
lines.append('|---|---|---|---|---|---|---|---|')
for label, deltas in [('VIS', vis_deltas), ('Random', rand_deltas)]:
    if deltas:
        lines.append('| %s | %d | %.6f | %.6f | %.6f | %.6f | %d | %d |' % (
            label, len(deltas), np.mean(deltas), np.median(deltas),
            min(deltas), max(deltas),
            sum(1 for d in deltas if d > 0.01), sum(1 for d in deltas if d > 0.02)))
lines.append('')

with open(os.path.join(REPO, 'reports', 'STAGEB_PATCHED_RERUN_MECHANISM_READOUT.md'), 'w') as f:
    f.write('\n'.join(lines))

# Per-task CSV
TASK_CSV = os.path.join(REPO, 'tables', 'stageb_selective_rerun_mechanism_by_task.csv')
with open(TASK_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task','paired','cmd_pos','rand_conf','phys_sens','phys_strict','vis_spec','hard_neg_phys','rand_phys'])
    for t, c in task_counts.most_common():
        tr = [r for r in label_rows if r['task_key'] == t]
        w.writerow([t, c,
            sum(1 for r in tr if r['cmd_susceptible']=='1'),
            sum(1 for r in tr if r['random_confounded']=='1'),
            sum(1 for r in tr if r['physical_response_sensitive']=='1'),
            sum(1 for r in tr if r['physical_response_strict']=='1'),
            sum(1 for r in tr if r['vis_specific_physical_response']=='1'),
            sum(1 for r in tr if r['cmd_susceptible']=='0' and r['random_confounded']=='0' and r['physical_response_sensitive']=='1'),
            sum(1 for r in tr if r['random_confounded']=='1' and r['physical_response_sensitive']=='1')])

print('\n'.join(lines))
print('Done')
