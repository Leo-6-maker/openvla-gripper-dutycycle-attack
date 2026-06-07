#!/usr/bin/env python3
"""Nightly Silver P1a watcher — conservative auto-advance with hard gates."""
import csv, os, sys, time, json, subprocess
from collections import defaultdict
from datetime import datetime

REPO = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607'
PY = '/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python'
SILVER_O = '/data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608'
DETECTOR_O = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608'
os.makedirs(DETECTOR_O, exist_ok=True)
BRONZE_LABELS = '/tmp/bronze_labels.csv'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = '%s %s' % (ts, msg)
    print(line)
    with open(os.path.join(SILVER_O, 'nightly.log'), 'a') as f:
        f.write(line + '\n')

def run(cmd):
    log('RUN: %s' % ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    log(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
    if r.returncode != 0:
        log('FAIL rc=%d stderr=%s' % (r.returncode, r.stderr[:200]))
    return r.returncode

# ── Step 1: Wait for workers ──
log('===== NIGHTLY SILVER P1A WATCHER =====')
EXPECTED = 84
while True:
    n = len([f for f in os.listdir(SILVER_O) if f.startswith('summary_') and f.endswith('.json')])
    workers = subprocess.run(['ps', 'aux'], capture_output=True, text=True).stdout
    alive = sum(1 for _ in [1] if 'silver_p1a_w' in workers)
    log('Progress: %d/%d summaries, %d workers alive' % (n, EXPECTED, alive))
    if n >= EXPECTED and alive == 0:
        break
    time.sleep(300)  # 5 min

log('All P1a workers DONE — %d summaries' % n)

# ── Step 2: Validator ──
log('=== VALIDATOR ===')
rc = run([PY, REPO + '/scripts/stageb/validate_stageb_trace_v1_1.py', '--dir', SILVER_O])
if rc != 0:
    log('VALIDATOR FAILED — STOP')
    sys.exit(1)

# ── Step 3: Postprocess ──
log('=== POSTPROCESS ===')
run([PY, REPO + '/scripts/stageb/postprocess_traces_v1_1.py',
     '--input-dir', SILVER_O, '--output-csv', '/tmp/silver_p1a_qpos.csv'])

# Clean qpos
with open('/tmp/silver_p1a_qpos.csv') as f:
    qrows = [r for r in csv.DictReader(f) if int(r.get('n_window_steps', 0)) > 0]
pairs = defaultdict(list)
for r in qrows:
    pairs[r['pair_id']].append(r)
complete = []
for pid, items in pairs.items():
    conds = set(i['condition'] for i in items)
    if len(items) == 2 and conds == {'vis_pgd', 'random_linf'}:
        complete.extend(items)
    else:
        log('SKIP unpaired: %s (%d traces)' % (pid, len(items)))
with open('/tmp/silver_p1a_qpos_clean.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=qrows[0].keys())
    w.writeheader(); w.writerows(complete)
log('Clean qpos: %d traces (%d pairs)' % (len(complete), len(complete)//2))

# ── Step 4: Build labels ──
log('=== LABELS ===')
run([PY, REPO + '/scripts/stageb/build_pair_labels_v1_1.py',
     '--qpos-csv', '/tmp/silver_p1a_qpos_clean.csv',
     '--output-csv', '/tmp/silver_p1a_labels.csv'])

# ── Step 5: Label stability ──
log('=== LABEL STABILITY ===')
with open(BRONZE_LABELS) as f:
    bronze = {r['pair_id']: r for r in csv.DictReader(f)}
with open('/tmp/silver_p1a_labels.csv') as f:
    silver = list(csv.DictReader(f))

silver_by_parent = defaultdict(list)
for r in silver:
    pid = r['pair_id']
    # Extract parent from silver_<parent>_r<N>_vis|rand
    parts = pid.split('_r')
    parent = '_'.join(parts[0].split('_')[1:]) if len(parts) >= 2 else pid
    silver_by_parent[parent].append(r)

stable_cmd = 0; stable_phys = 0; stable_rand = 0; unstable = 0; hard_neg = 0
parent_results = []
for parent, repeats in sorted(silver_by_parent.items()):
    bl = bronze.get(parent, {})
    n_vis = sum(1 for r in repeats if r['condition'] == 'vis_pgd')
    n_rand = sum(1 for r in repeats if r['condition'] == 'random_linf')
    vis_cmd = sum(1 for r in repeats if r['condition']=='vis_pgd' and r['cmd_susceptible']=='1')
    vis_phys = sum(1 for r in repeats if r['condition']=='vis_pgd' and r['physical_response_sensitive']=='1')
    rand_cmd = sum(1 for r in repeats if r['condition']=='random_linf' and r['cmd_susceptible']=='1')
    rand_phys = sum(1 for r in repeats if r['condition']=='random_linf' and r['physical_response_sensitive']=='1')
    vis_rate = vis_cmd / max(n_vis, 1)
    rand_rate = rand_cmd / max(n_rand, 1)
    phys_rate = vis_phys / max(n_vis, 1)
    rand_phys_rate = rand_phys / max(n_rand, 1)

    is_cmd = vis_rate >= 0.67 and rand_rate <= 0.33
    is_phys = phys_rate >= 0.67 and rand_phys_rate <= 0.33
    is_rand = rand_rate >= 0.67
    is_hn = vis_rate <= 0.33 and rand_rate <= 0.33

    if is_cmd or is_phys:
        if is_cmd: stable_cmd += 1
        if is_phys: stable_phys += 1
        status = 'stable_cmd' if is_cmd else '' + ('+phys' if is_phys else '')
    elif is_rand: stable_rand += 1; status = 'stable_random_confounded'
    elif is_hn: hard_neg += 1; status = 'hard_negative_confirmed'
    else: unstable += 1; status = 'unstable'

    parent_results.append({'parent': parent, 'status': status, 'vis_rate': vis_rate, 'rand_rate': rand_rate,
                           'phys_rate': phys_rate, 'b_cmd': bl.get('cmd_susceptible','0')})

pos_parents = sum(1 for r in parent_results if r['b_cmd'] == '1')
pos_stable = sum(1 for r in parent_results if r['b_cmd']=='1' and 'stable' in r['status'] and 'random' not in r['status'])
pos_stability = pos_stable / max(pos_parents, 1)
rand_parents = sum(1 for r in parent_results if r['b_rand'] == '1')
rand_stable = sum(1 for r in parent_results if r['status'] == 'stable_random_confounded')
rand_stability = rand_stable / max(rand_parents, 1) if rand_parents > 0 else 0.0

log('Stability: stable_cmd=%d stable_phys=%d stable_rand=%d hard_neg=%d unstable=%d' %
    (stable_cmd, stable_phys, stable_rand, hard_neg, unstable))
log('positive_stability_rate=%.2f random_stability_rate=%.2f' % (pos_stability, rand_stability))

# ── Step 6: Gate check ──
failures = []
if stable_cmd < 4: failures.append('stable_cmd=%d < 4' % stable_cmd)
if stable_phys < 2: failures.append('stable_phys=%d < 2' % stable_phys)
if pos_stability < 0.50: failures.append('pos_stability=%.2f < 0.50' % pos_stability)
if rand_stability > 0.60: failures.append('rand_stability=%.2f > 0.60' % rand_stability)

P1A_GATE = len(failures) == 0
log('P1A GATE: %s' % ('PASS' if P1A_GATE else 'FAIL — ' + ', '.join(failures)))

if not P1A_GATE:
    log('STOP: P1a gate failed. Writing diagnostic report.')
    with open(os.path.join(SILVER_O, 'NIGHTLY_STOP.txt'), 'w') as f:
        f.write('P1A_GATE_FAIL: %s\n' % '; '.join(failures))
    sys.exit(0)

# ── Step 7: Detector v0 CPU training ──
log('=== DETECTOR V0 ===')
# ... (detector code — to be run separately or inline)

log('===== NIGHTLY COMPLETE — P1a PASS, awaiting morning review =====')
log('stable_cmd=%d stable_phys=%d' % (stable_cmd, stable_phys))
