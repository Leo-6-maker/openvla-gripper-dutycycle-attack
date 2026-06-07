#!/usr/bin/env python3
"""Run full logit-margin validation: clean rollouts for all candidates with logit saving,
then aggregate to window-level features. GPU 0,1.
"""
import csv, os, sys, subprocess, glob, time
from collections import defaultdict
from datetime import datetime
import numpy as np

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
OUT = '/data/liuyu/outputs/logit_margin_validation_20260606'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--gpu_pair', default='0,1')
_args, _unknown = _ap.parse_known_args()
GPU = _args.gpu_pair
SCRIPT = os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3_logit.py')

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    print('%s %s' % (t, msg))

# Load candidates
with open(os.path.join(REPO, 'tables/logit_margin_candidates_shard.csv')) as f:
    candidates = list(csv.DictReader(f))
log('Loaded %d candidates' % len(candidates))

os.makedirs(OUT, exist_ok=True)

success = 0
for i, c in enumerate(candidates):
    task = c['task_key'].strip()
    sid = c['state_id']
    ws = c['window_start']
    we = c['window_end']

    log('[%d/%d] %s s%s [%s,%s]' % (i+1, len(candidates), task, sid, ws, we))

    cmd = [PY, '-u', SCRIPT,
        '--task', task, '--state-id', str(sid), '--condition', 'clean',
        '--gpu_pair', GPU,
        '--perturb_start', str(ws), '--perturb_end', str(we),
        '--eps_raw_pixels', '6',
        '--objective', 'prefix_locked_gripper_open_margin',
        '--seed', '0']
    log_path = os.path.join(OUT, '%s_s%s_clean_w%s_%s.log' % (task, sid, ws, we))
    t0 = time.time()
    try:
        with open(log_path, 'w') as lf:
            rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, timeout=600).returncode
        rt = time.time() - t0
        if rc == 0:
            success += 1
            log('  OK (%.0fs) [%d/%d success]' % (rt, success, i+1))
        else:
            log('  FAILED rc=%d (%.0fs)' % (rc, rt))
    except subprocess.TimeoutExpired:
        log('  TIMEOUT')
    except Exception as e:
        log('  ERROR: %s' % e)
    time.sleep(3)

log('=== Phase 1 complete: %d/%d clean rollouts succeeded ===' % (success, len(candidates)))

# Phase 2: Aggregate logit features from traces
log('=== Phase 2: Aggregating logit features ===')
trace_rows = []
window_rows = []

for i, c in enumerate(candidates):
    task = c['task_key'].strip()
    sid = c['state_id']
    ws = int(c['window_start'])
    we = int(c['window_end'])

    # Find trace CSV
    ep_dir = os.path.join(OUT, '%s_s%s' % (task, sid))
    trace_files = glob.glob(os.path.join(ep_dir, 'traces', '*clean*trace.csv'))
    if not trace_files:
        continue

    trace_file = trace_files[0]
    with open(trace_file) as f:
        reader = csv.DictReader(f)
        steps = list(reader)

    window_steps = []
    for s in steps:
        try:
            step = int(float(s.get('step', -1)))
        except:
            continue
        if ws <= step <= we:
            window_steps.append(s)

            # Per-step trace row
            trace_rows.append({
                'task_key': task, 'state_id': str(sid),
                'window_start': str(ws), 'window_end': str(we),
                'step': str(step),
                'gripper_logit_open_mass': s.get('gripper_logit_open_mass', ''),
                'gripper_logit_close_mass': s.get('gripper_logit_close_mass', ''),
                'gripper_logit_margin': s.get('gripper_logit_margin', ''),
                'gripper_entropy': s.get('gripper_entropy', ''),
                'gripper_top2_margin': s.get('gripper_top2_margin', ''),
                'all_action_entropy': s.get('all_action_entropy', ''),
            })

    if len(window_steps) < 2:
        continue

    # Extract numeric values
    margins = []
    entropies = []
    open_masses = []
    close_masses = []
    top2s = []
    all_ents = []
    for s in window_steps:
        for name, lst in [('gripper_logit_margin', margins), ('gripper_entropy', entropies),
                          ('gripper_logit_open_mass', open_masses), ('gripper_logit_close_mass', close_masses),
                          ('gripper_top2_margin', top2s), ('all_action_entropy', all_ents)]:
            try:
                lst.append(float(s.get(name, 0)))
            except:
                pass

    if not margins:
        continue

    margins_arr = np.array(margins)
    n_low = int(np.sum(np.abs(margins_arr) < 0.1))
    streak = max_streak = 0
    for m in margins_arr:
        if abs(m) < 0.1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    window_rows.append({
        'task_key': task, 'state_id': str(sid),
        'window_start': str(ws), 'window_end': str(we),
        'n_steps': str(len(window_steps)),
        'logit_margin_min': str(round(float(np.min(margins_arr)), 8)),
        'logit_margin_mean': str(round(float(np.mean(margins_arr)), 8)),
        'logit_margin_std': str(round(float(np.std(margins_arr)), 8)),
        'entropy_mean': str(round(float(np.mean(entropies)), 8)) if entropies else '',
        'entropy_max': str(round(float(np.max(entropies)), 8)) if entropies else '',
        'open_mass_max': str(round(float(np.max(open_masses)), 8)) if open_masses else '',
        'close_mass_mean': str(round(float(np.mean(close_masses)), 8)) if close_masses else '',
        'top2_margin_min': str(round(float(np.min(top2s)), 8)) if top2s else '',
        'all_action_entropy_mean': str(round(float(np.mean(all_ents)), 8)) if all_ents else '',
        'low_margin_step_count': str(n_low),
        'longest_low_margin_streak': str(max_streak),
        'label_status': c.get('label_status', ''),
        'train_use': c.get('label_use', ''),
        'taxonomy': c.get('taxonomy', ''),
    })

# Write outputs
OUT_TRACE = os.path.join(REPO, 'tables/openvla_action_token_logit_trace_summary_shard45.csv')
OUT_WINDOW = os.path.join(REPO, 'tables/openvla_logit_margin_online_features_shard45.csv')

if trace_rows:
    with open(OUT_TRACE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    log('Wrote %d trace rows to %s' % (len(trace_rows), OUT_TRACE))

if window_rows:
    with open(OUT_WINDOW, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
        w.writeheader(); w.writerows(window_rows)
    log('Wrote %d window rows to %s' % (len(window_rows), OUT_WINDOW))

    # Quick audit
    all_margins = [float(r['logit_margin_mean']) for r in window_rows if r.get('logit_margin_mean')]
    all_entropy = [float(r['entropy_mean']) for r in window_rows if r.get('entropy_mean')]
    margin_unique = len(set(round(m, 6) for m in all_margins))
    log('Margin range: [%.4f, %.4f] unique=%d/%d' % (min(all_margins), max(all_margins), margin_unique, len(window_rows)))
    log('Entropy range: [%.6f, %.6f]' % (min(all_entropy), max(all_entropy)))
    log('ARE LOGIT FEATURES DEGENERATE? %s' % ('YES (constant)' if margin_unique <= 2 else 'NO (has variance)'))
else:
    log('WARNING: No window rows generated')
