#!/usr/bin/env python3
"""S20M3b: Multiseed confirmation of S20M3a positive parents.
cream_cheese_s2_w80-90 (CMD_POSITIVE) + salad_dressing_s2_w100-110 (TASK_EFFECT).
seeds 94,95 — RAND first, VIS only if RAND clean."""
import csv, json, os
from collections import Counter

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O = '/data/liuyu/outputs/stageb_s20m3b_multiseed_confirmation_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O+'/queues', exist_ok=True)

PARENTS = [
    {'candidate_id': 'cream_cheese_s2_w80_90',   'task': 'cream_cheese',  'state_id': '2',
     'window_start': '80', 'window_end': '90', 'phase': 'early_transport',
     's20m3a_rand_open': 0, 's20m3a_rand_streak': 0, 's20m3a_rand_label': 'RAND_STRICT',
     's20m3a_vis_open': 8, 's20m3a_vis_streak': 8, 's20m3a_class': 'CMD_POSITIVE'},
    {'candidate_id': 'salad_dressing_s2_w100_110', 'task': 'salad_dressing', 'state_id': '2',
     'window_start': '100', 'window_end': '110', 'phase': 'place_or_done',
     's20m3a_rand_open': 0, 's20m3a_rand_streak': 0, 's20m3a_rand_label': 'RAND_STRICT',
     's20m3a_vis_open': 3, 's20m3a_vis_streak': 1, 's20m3a_class': 'TASK_EFFECT'},
]

SEEDS = ['94', '95']

# Build jobs: RAND then VIS for each (parent, seed) pair
jobs = []; jid = 310000
for p in PARENTS:
    for seed in SEEDS:
        jid += 1
        # RAND first
        jobs.append({
            'job_id': str(jid), 'task': p['task'], 'state_id': p['state_id'],
            'window_start': p['window_start'], 'window_end': p['window_end'],
            'condition': 'random_linf', 'attack_seed': seed, 'random_control_seed': seed,
            'seed': '0', 'candidate_id': p['candidate_id'],
            'tier': 'M3b_'+p['s20m3a_class'], 'track': 'S20M3b', 'status': 'pending',
        })
        jid += 1
        # VIS second (same seed → worker finds matched RAND)
        jobs.append({
            'job_id': str(jid), 'task': p['task'], 'state_id': p['state_id'],
            'window_start': p['window_start'], 'window_end': p['window_end'],
            'condition': 'vis_pgd', 'attack_seed': seed, 'random_control_seed': '',
            'seed': '0', 'candidate_id': p['candidate_id'],
            'tier': 'M3b_'+p['s20m3a_class'], 'track': 'S20M3b', 'status': 'pending',
        })

# Split: cream_cheese → GPU 0,1; salad_dressing → GPU 2,3
queues = {}
for j in jobs:
    gpu = 'gpu0' if j['task'] == 'cream_cheese' else 'gpu2'
    if gpu not in queues: queues[gpu] = []
    queues[gpu].append(j)

for gpu, gj in queues.items():
    qp = O+'/queues/s20m3b_%s.csv' % gpu
    with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(gj)
    print('%s: %d jobs (RAND+VIS paired)' % (gpu, len(gj)))

# Write manifest
with open(T+'/s20m3b_multiseed_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['candidate_id','task','state_id','window_start','window_end',
        'phase','s20m3a_class','s20m3a_rand_label','s20m3a_rand_open','s20m3a_rand_streak',
        's20m3a_vis_open','s20m3a_vis_streak','seeds'])
    w.writeheader()
    for p in PARENTS:
        w.writerow({**p, 'seeds': ','.join(SEEDS)})

# Gate audit template (to be filled after RAND completes)
with open(T+'/s20m3b_rand_gate_audit.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['candidate_id','seed','condition','status','rand_open',
        'rand_streak','rand_label','rand_done','gate_passed','note'])
    w.writeheader()

print()
print('S20M3b multiseed confirmation:')
print('  Parents: 2 (cream_cheese_s2_w80-90, salad_dressing_s2_w100-110)')
print('  Seeds: 94, 95')
print('  Jobs: %d (RAND+VIS paired per seed)' % len(jobs))
print('  Strategy: RAND first, VIS only if RAND STRICT/USABLE')
print()
print('  Manifest: %s/s20m3b_multiseed_manifest.csv' % T)
print('  Queues: %s/queues/' % O)

# Expected totals for audit
print()
print('Expected per parent: 2 RAND + up to 2 VIS = up to 4')
print('Total: up to 8 jobs (4 RAND + up to 4 VIS)')
print()
print('Confirmation criteria:')
print('  CONFIRMED_CMD: >=2/3 seeds VIS-RAND open/streak gap >=3')
print('  CONFIRMED_TASK_EFFECT: >=2/3 seeds RAND clean, VIS timeout/failure')
