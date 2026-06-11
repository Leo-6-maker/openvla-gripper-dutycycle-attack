#!/usr/bin/env python3
"""Generate S20f labeling job queues for Track B and Track C GPU workers.
Phase-stratified sampling from v0.3.1 candidate universe, ~15 jobs per GPU pair."""
import csv, os, json
from pathlib import Path
from collections import defaultdict

UNIVERSE = '/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv'
OUT_DIR = '/data/liuyu/outputs/stageb_s20f_queues_20260611'
os.makedirs(OUT_DIR, exist_ok=True)

# Load universe
with open(UNIVERSE) as f:
    candidates = list(csv.DictReader(f))

# Build complete labeling job list: RAND + VIS pairs
label_jobs = []
seen_windows = set()
jid = 966000

for c in candidates:
    cid = c['candidate_id']
    task = c['task']; sid = c['state_id']
    ws = int(c['window_start']); we = int(c['window_end'])
    phase = c['phase_id']

    # Skip windows we already have RAND results for
    if cid in seen_windows:
        continue
    seen_windows.add(cid)

    # RAND job
    jid += 1
    label_jobs.append({
        'job_id': str(jid), 'task': task, 'state_id': sid,
        'window_start': str(ws), 'window_end': str(we),
        'condition': 'random_linf', 'attack_seed': '80', 'random_control_seed': '80',
        'seed': '0', 'phase': phase, 'candidate_id': cid, 'status': 'pending',
    })

# Prioritize: finish remaining 10-batch VIS first, then add new RAND+VIS
# Sort by: 1) existing RAND results first (VIS needed), 2) phase, 3) task
def sort_key(j):
    # VIS jobs first (follow-up to existing RAND)
    phase_order = {'approach': 0, 'grasp_transition': 1, 'early_transport': 2,
                   'transport': 3, 'preplace': 4, 'place_or_done': 5}
    return (j['condition'] != 'random_linf', phase_order.get(j.get('phase', 'unknown'), 9), j['task'], j['window_start'])

label_jobs.sort(key=sort_key)

# Split into 2 queues for GPU(2,6) and GPU(4,5)
# Round-robin by condition (keep RAND+VIS pairs on same GPU)
queue26 = []; queue45 = []
for i, job in enumerate(label_jobs):
    if i % 2 == 0:
        queue26.append(job)
    else:
        queue45.append(job)

# Write queues
for qname, qjobs, gpu, render in [
    ('queue_gpu26.csv', queue26, '2,6', '2'),
    ('queue_gpu45.csv', queue45, '4,5', '4'),
]:
    qpath = os.path.join(OUT_DIR, qname)
    fields = ['job_id', 'task', 'state_id', 'window_start', 'window_end',
              'condition', 'attack_seed', 'random_control_seed', 'seed',
              'phase', 'candidate_id', 'status']
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(qjobs)
    print('%s: %d jobs (GPU %s, render=%s)' % (qname, len(qjobs), gpu, render))
    # Phase breakdown
    phases = defaultdict(int)
    for j in qjobs:
        phases[j.get('phase', '?')] += 1
    for p, n in sorted(phases.items()):
        print('  %s: %d' % (p, n))

print('\nOutput: %s' % OUT_DIR)
