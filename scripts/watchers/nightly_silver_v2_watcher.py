#!/usr/bin/env python3
"""Nightly Silver v2 watcher — HARD/SOFT gates, utility backlog, capped expansion.

HARD DATA GATE failures → STOP_ALL_GPU immediately.
SOFT SCIENCE GATE failures → continue GPU with utility backlog queues.
Never exceeds 36 expansion windows without user review.
"""
import csv, os, sys, time, json, subprocess
from collections import defaultdict, Counter
from datetime import datetime

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SILVER_O = '/data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608'
BRONZE_O = '/data/liuyu/outputs/stageb_v1_1_bronze_batch_rc1a_20260607'
DETECTOR_O = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608'
RESCUE_O = '/data/liuyu/outputs/stageb_v1_1_stability_rescue_rc1a_20260608'
EXPANSION_O = '/data/liuyu/outputs/stageb_v1_1_bronze_expansion_rc1a_20260608'
P1B_O = '/data/liuyu/outputs/stageb_v1_1_silver_p1b_rc1a_20260608'
BRONZE_LABELS = '/tmp/bronze_labels.csv'
RUNNER = REPO + '/scripts/run_stageb_vis_labeling.py'
os.makedirs(RESCUE_O, exist_ok=True)
os.makedirs(EXPANSION_O, exist_ok=True)
os.makedirs(P1B_O, exist_ok=True)

LOG_PATH = os.path.join(SILVER_O, 'nightly_v2.log')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = '%s %s' % (ts, msg)
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=120):
    log('RUN: %s' % ' '.join(cmd[:6]))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=timeout)
    out = r.stdout[-300:] if len(r.stdout) > 300 else r.stdout
    log(out)
    if r.returncode != 0:
        log('FAIL rc=%d: %s' % (r.returncode, r.stderr[:200]))
    return r.returncode, r.stdout

def check_hard_gate():
    """Check HARD DATA GATE. Returns (ok, failures_list)."""
    failures = []
    # Check a sample trace for metadata
    traces = [f for f in os.listdir(SILVER_O) if f.startswith('trace_') and f.endswith('.csv')]
    if traces:
        with open(os.path.join(SILVER_O, traces[0])) as f:
            r = list(csv.DictReader(f))[0]
        if r.get('source_snapshot_id','') != 'f9840cb1': failures.append('source_snapshot_id')
        if r.get('trace_version','') != 'corrected_stageb_v1_1': failures.append('trace_version')
        if r.get('prompt_style','') != 'official_in_out': failures.append('prompt_style')
        if 'official_rot180' not in r.get('image_preprocess_style',''): failures.append('image_preprocess')
        if r.get('open_convention','') != 'env_action_6_lt_neg_0p5_means_OPEN': failures.append('open_convention')
        if r.get('qpos_source','') != 'obs_robot0_gripper_qpos': failures.append('qpos_source')
    return len(failures) == 0, failures

def count_summaries():
    return len([f for f in os.listdir(SILVER_O) if f.startswith('summary_')])

def count_workers():
    r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    return sum(1 for line in r.stdout.split('\n') if 'silver_p1a_w' in line and 'grep' not in line)

def generate_worker_script(queue_csv, gpu_pair, output_root, log_prefix, script_path):
    """Generate a worker bash script from a queue CSV."""
    with open(queue_csv) as f:
        jobs = list(csv.DictReader(f))
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        for j in jobs:
            jid = 95000 + int(j.get('silver_job_id', len(jobs)))
            cond = j.get('condition', 'vis_pgd')
            task = j.get('task_key', j.get('task', '?'))
            sid = j.get('state_id', j.get('state-id', '0'))
            ws = j.get('window_start', j.get('window-start', '1'))
            we = j.get('window_end', j.get('window-end', '10'))
            seed = j.get('random_seed', j.get('seed', '42'))
            pid = j.get('pair_id_silver', j.get('pair_id', 'auto'))
            f.write('echo "$(date +%%H:%%M:%%S) %s %s s%s" >> %s/%s.log\n' % (cond, task, sid, output_root, log_prefix))
            f.write('CUDA_VISIBLE_DEVICES=%s %s -u %s --task %s --state-id %s --window_start %s --window_end %s --condition %s --gpu_pair 0,1 --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --seed %s --pair_id %s --job_id %d --output_dir %s >> %s/w_%s.log 2>&1\n' % (gpu_pair, PY, RUNNER, task, sid, ws, we, cond, seed, pid, jid, output_root, output_root, log_prefix))
        f.write('echo "$(date +%%H:%%M:%%S) DONE" >> %s/%s.log\n' % (output_root, log_prefix))
    log('Generated: %s (%d jobs)' % (script_path, len(jobs)))

def launch_worker(script_path):
    subprocess.Popen(['nohup', 'bash', script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log('Launched: %s' % script_path)

def postprocess_and_label(input_dir, qpos_csv, labels_csv):
    rc, _ = run([PY, REPO + '/scripts/stageb/postprocess_traces_v1_1.py',
                 '--input-dir', input_dir, '--output-csv', qpos_csv])
    if rc != 0:
        return False
    # Clean
    with open(qpos_csv) as f:
        rows = [r for r in csv.DictReader(f) if int(r.get('n_window_steps',0)) > 0]
    pairs = defaultdict(list)
    for r in rows: pairs[r['pair_id']].append(r)
    complete = []
    for pid, items in pairs.items():
        if len(items) == 2 and set(i['condition'] for i in items) == {'vis_pgd', 'random_linf'}:
            complete.extend(items)
    with open(qpos_csv + '.clean', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(complete)
    rc, _ = run([PY, REPO + '/scripts/stageb/build_pair_labels_v1_1.py',
                 '--qpos-csv', qpos_csv + '.clean', '--output-csv', labels_csv])
    return rc == 0

# ── MAIN ──
log('===== NIGHTLY V2 WATCHER START =====')

# Wait for P1a
EXPECTED_P1A = 84
while True:
    n = count_summaries()
    w = count_workers()
    log('P1a: %d/%d summaries, %d workers' % (n, EXPECTED_P1A, w))
    if n >= EXPECTED_P1A and w == 0:
        break
    time.sleep(300)

log('P1a complete: %d summaries' % n)

# HARD GATE check
hard_ok, hard_fails = check_hard_gate()
if not hard_ok:
    log('HARD DATA GATE FAIL: %s — STOP_ALL_GPU' % hard_fails)
    with open(os.path.join(SILVER_O, 'HARD_BLOCKED.txt'), 'w') as f:
        f.write('HARD_GATE_FAIL: %s\n' % hard_fails)
    sys.exit(1)
log('HARD DATA GATE: PASS')

# Validator
rc, _ = run([PY, REPO + '/scripts/stageb/validate_stageb_trace_v1_1.py', '--dir', SILVER_O])
if rc != 0:
    log('VALIDATOR FAIL — STOP')
    sys.exit(1)

# Postprocess + labels
log('=== P1A POSTPROCESS ===')
ok = postprocess_and_label(SILVER_O, '/tmp/silver_p1a_qpos.csv', '/tmp/silver_p1a_labels.csv')
if not ok:
    log('POSTPROCESS FAIL')
    sys.exit(1)

# Label stability
log('=== P1A STABILITY ===')
with open(BRONZE_LABELS) as f:
    bronze = {r['pair_id']: r for r in csv.DictReader(f)}
with open('/tmp/silver_p1a_labels.csv') as f:
    silver_labels = list(csv.DictReader(f))

# Group by parent
silver_by_parent = defaultdict(list)
for r in silver_labels:
    pid = r['pair_id']
    parent = '_'.join(pid.split('_r')[0].split('_')[1:]) if '_r' in pid else pid
    silver_by_parent[parent].append(r)

stable_cmd = 0; stable_phys = 0; stable_rand = 0; unstable = 0
for parent, repeats in silver_by_parent.items():
    n_v = sum(1 for r in repeats if r['condition']=='vis_pgd')
    n_r = sum(1 for r in repeats if r['condition']=='random_linf')
    vc = sum(1 for r in repeats if r['condition']=='vis_pgd' and r['cmd_susceptible']=='1')
    vp = sum(1 for r in repeats if r['condition']=='vis_pgd' and r['physical_response_sensitive']=='1')
    rc_c = sum(1 for r in repeats if r['condition']=='random_linf' and r['cmd_susceptible']=='1')
    vr = vc / max(n_v, 1); rr = rc_c / max(n_r, 1)
    pr = vp / max(n_v, 1)
    if vr >= 0.67 and rr <= 0.33: stable_cmd += 1
    if pr >= 0.67: stable_phys += 1
    if rr >= 0.67: stable_rand += 1
    if vr <= 0.33 and rr <= 0.33: pass  # hard neg
    elif not (vr >= 0.67 or rr >= 0.67): unstable += 1

log('Stability: stable_cmd=%d stable_phys=%d stable_rand=%d unstable=%d' %
    (stable_cmd, stable_phys, stable_rand, unstable))

pos_stability = stable_cmd / max(len([1 for r in silver_by_parent.values() if any(x['cmd_susceptible']=='1' for x in r)]), 1)
log('positive_stability_rate=%.2f' % pos_stability)

# SOFT GATE: determine next action
soft_fail = (stable_cmd < 4 or stable_phys < 2 or pos_stability < 0.50)

if soft_fail:
    log('SOFT GATE WEAK — switching to Stability Rescue Queue')
    # Generate stability rescue queue...
    log('Stability Rescue: would launch ~40-60 jobs')
    # For now: just log intent, wait for morning
    log('NIGHTLY V2: SOFT GATE — awaiting morning review for rescue launch')
else:
    log('SOFT GATE PASS — would launch P1b + capped expansion')
    log('NIGHTLY V2: PASS — awaiting morning review')

log('===== NIGHTLY V2 WATCHER DONE =====')
log('Summary: P1a done, hard_gate=PASS, stable_cmd=%d, stable_phys=%d' % (stable_cmd, stable_phys))
