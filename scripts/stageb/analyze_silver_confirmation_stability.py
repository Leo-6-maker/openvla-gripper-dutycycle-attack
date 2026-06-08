#!/usr/bin/env python3
"""Silver confirmation stability analysis.

For each of 8 parents with 2 repeat seeds each:
- Compare original label vs repeat labels
- Classify stability: stable / partial / unstable
- Output stability report

Usage: Run AFTER confirmation completes (valiator PASS first).
"""
import csv, json, os, sys
from collections import defaultdict

CONFIRM_DIR = '/data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4'
LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_silver_confirmation_rc1a_e33b5e4/confirmation_stability.csv'

CMD_T = 6
PHYS_T = 0.01


def classify(vis_open, vis_streak, rand_open, rand_streak, vis_qpos, rand_qpos):
    vo = int(vis_open or 0); vs = int(vis_streak or 0)
    ro = int(rand_open or 0); rs = int(rand_streak or 0)
    vq = abs(float(vis_qpos or 0)); rq = abs(float(rand_qpos or 0))

    vis_cmd = (vo >= CMD_T or vs >= CMD_T)
    rand_cmd = (ro >= CMD_T or rs >= CMD_T)
    vis_phys = vq >= PHYS_T
    rand_phys = rq >= PHYS_T

    if vis_cmd and not rand_cmd:
        cmd_label = 'cmd_specific'
    elif rand_cmd and not vis_cmd:
        cmd_label = 'rand_command_sensitive'
    elif vis_cmd and rand_cmd:
        cmd_label = 'random_command_confounded'
    else:
        cmd_label = 'no_command'

    if vis_phys and not rand_phys:
        phys_label = 'vis_specific_phys'
    elif rand_phys and not vis_phys:
        phys_label = 'rand_phys_confound'
    elif vis_phys and rand_phys:
        phys_label = 'shared_qpos'
    else:
        phys_label = 'no_phys'

    return cmd_label, phys_label


# --- Load original 72-pair labels ---
orig_labels = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        orig_labels[r['pair_id']] = r

# --- Load confirmation summaries ---
pairs = defaultdict(list)
for fname in sorted(os.listdir(CONFIRM_DIR)):
    if not fname.startswith('summary_') or not fname.endswith('.json'):
        continue
    with open(os.path.join(CONFIRM_DIR, fname)) as f:
        d = json.load(f)
    pid = d.get('pair_id', '?')
    cond = 'VIS' if '_vis_pgd_' in fname else 'RAND'
    pairs[pid].append((cond, d))

# --- Match repeats to parents ---
# Group by parent (strip _r0/_r1 suffix)
parent_repeats = defaultdict(list)
for pid, entries in pairs.items():
    # pid format: silver_{cat}_{task}_s{sid}_w{ws}_{we}_seed{orig_seed}_r{repeat}
    base = pid.rsplit('_r', 1)[0]  # strip _r0 / _r1
    repeat_idx = int(pid.rsplit('_r', 1)[1])
    parent_repeats[base].append((repeat_idx, pid, entries))

# --- Map to original parent ---
# Confirmation parent → original 72-pair pair_id
# We need to find the matching original pair
parent_to_orig = {}
for orig_pid, orig_r in orig_labels.items():
    for base, repeats in parent_repeats.items():
        # base = silver_{cat}_{task}_s{sid}_w{ws}_{we}_seed{orig_seed}
        # Extract task, sid, ws, we
        parts = base.split('_')
        # Try to match by task + state + window
        if orig_r['task_key'] in base and orig_r['state_id'] in base:
            if orig_r['window_start'] in base and orig_r['window_end'] in base:
                parent_to_orig[base] = orig_pid

# --- Analyze stability ---
rows = []
for base, repeats in sorted(parent_repeats.items()):
    orig_pid = parent_to_orig.get(base, '?')
    orig = orig_labels.get(orig_pid, {})

    repeat_results = []
    for r_idx, pid, entries in sorted(repeats):
        vis_data = None; rand_data = None
        for cond, d in entries:
            if cond == 'VIS': vis_data = d
            elif cond == 'RAND': rand_data = d
        if not vis_data or not rand_data:
            repeat_results.append((r_idx, 'MISSING', 'MISSING', 0, 0, 0, 0, 0, 0))
            continue
        vo = vis_data.get('decoded_open_count', 0)
        vs = vis_data.get('decoded_longest_open_streak', 0)
        ro = rand_data.get('decoded_open_count', 0)
        rs = rand_data.get('decoded_longest_open_streak', 0)
        vq = vis_data.get('qpos_delta', 0)
        rq = rand_data.get('qpos_delta', 0)
        cmd_l, phys_l = classify(vo, vs, ro, rs, vq, rq)
        repeat_results.append((r_idx, cmd_l, phys_l, vo, vs, ro, rs,
                               round(abs(vq), 5), round(abs(rq), 5)))

    # Determine stability
    cmd_labels = [c for _, c, _, _, _, _, _, _, _ in repeat_results]
    phys_labels = [p for _, _, p, _, _, _, _, _, _ in repeat_results]

    # Original label
    orig_cmd_specific = orig.get('cmd_specific', '0')
    orig_rand_cmd = orig.get('rand_command_sensitive', '0')
    orig_confounded = orig.get('random_command_confounded', '0')
    orig_vis_phys = orig.get('vis_specific_phys', '0')
    orig_shared_qpos = orig.get('shared_qpos_response', '0')
    orig_rand_phys = orig.get('rand_phys_confound', '0')

    if orig_cmd_specific == '1':
        orig_cmd_class = 'cmd_specific'
    elif orig_rand_cmd == '1':
        orig_cmd_class = 'rand_command_sensitive'
    elif orig_confounded == '1':
        orig_cmd_class = 'random_command_confounded'
    else:
        orig_cmd_class = 'no_command'

    if orig_vis_phys == '1':
        orig_phys_class = 'vis_specific_phys'
    elif orig_shared_qpos == '1':
        orig_phys_class = 'shared_qpos'
    elif orig_rand_phys == '1':
        orig_phys_class = 'rand_phys_confound'
    else:
        orig_phys_class = 'no_phys'

    # Stability: how many repeats match original class?
    cmd_matches = sum(1 for c in cmd_labels if c == orig_cmd_class) if cmd_labels else 0
    phys_matches = sum(1 for p in phys_labels if p == orig_phys_class) if phys_labels else 0
    n_repeats = len(cmd_labels)

    if n_repeats == 0:
        stability = 'MISSING'
    elif cmd_matches >= n_repeats and phys_matches >= n_repeats:
        stability = 'stable'
    elif cmd_matches >= 1 and phys_matches >= 1:
        stability = 'partial'
    elif cmd_matches >= n_repeats or phys_matches >= n_repeats:
        stability = 'partial_one_head'
    else:
        stability = 'unstable'

    # Build row
    row = {
        'parent_base': base,
        'original_pair_id': orig_pid,
        'task_key': orig.get('task_key', '?'),
        'state_id': orig.get('state_id', '?'),
        'window_start': orig.get('window_start', '?'),
        'window_end': orig.get('window_end', '?'),
        'orig_cmd_class': orig_cmd_class,
        'orig_phys_class': orig_phys_class,
        'n_repeats': str(n_repeats),
        'stability': stability,
    }
    for r_idx, cmd_l, phys_l, vo, vs, ro, rs, vq, rq in repeat_results:
        row['r%d_cmd' % r_idx] = cmd_l
        row['r%d_phys' % r_idx] = phys_l
        row['r%d_VIS_open' % r_idx] = str(vo)
        row['r%d_RAND_open' % r_idx] = str(ro)
        row['r%d_VIS_qpos' % r_idx] = str(vq)
        row['r%d_RAND_qpos' % r_idx] = str(rq)

    rows.append(row)
    print('%-40s orig=(%s, %s)  r0=(%s,%s) r1=(%s,%s)  %s' %
          (base[:40], orig_cmd_class, orig_phys_class,
           cmd_labels[0] if len(cmd_labels) > 0 else '?',
           phys_labels[0] if len(phys_labels) > 0 else '?',
           cmd_labels[1] if len(cmd_labels) > 1 else '?',
           phys_labels[1] if len(phys_labels) > 1 else '?',
           stability))

# Write
if rows:
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    stab_counts = Counter(r['stability'] for r in rows)
    print('\nStability: %s' % dict(stab_counts))
    print('Output: %s' % OUT)
else:
    print('No complete pairs yet — confirmation still running?')
