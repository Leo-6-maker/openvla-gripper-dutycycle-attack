#!/usr/bin/env python3
"""S20M4 clean scan: LIBERO Object states 10-49, eval_seed=0.
States 0-9 already covered in S20K/S20I clean expansion.
Full coverage scan — many states will fail, only success states become candidates."""
import csv, json, glob, os
from collections import Counter

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O = '/data/liuyu/outputs/stageb_s20m4_clean_scan_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O+'/queues', exist_ok=True)

OBJECT_TASKS = ['alphabet_soup','bbq_sauce','butter','chocolate_pudding',
                'cream_cheese','ketchup','milk','orange_juice','salad_dressing','tomato_sauce']

# Check existing clean (task,state) pairs
existing_clean = set()
clean_dirs = [
    '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean',
    '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612',
    '/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
]
for d in clean_dirs:
    if not os.path.exists(d): continue
    for f in glob.glob(d+'/summary_*clean*.json'):
        try:
            s = json.load(open(f))
            existing_clean.add((s['task'], str(s['state_id'])))
        except: pass

# Also exclude any already-running/in-progress
for f in glob.glob(O+'/summary_*.json'):
    try:
        s = json.load(open(f))
        existing_clean.add((s['task'], str(s['state_id'])))
    except: pass

print('Existing clean (task,state) pairs: %d' % len(existing_clean))

# Build queue: states 10-49
jobs = []; jid = 330000
scanned = Counter()

for task in OBJECT_TASKS:
    for state_id in range(10, 50):
        if (task, str(state_id)) in existing_clean:
            continue
        jid += 1
        jobs.append({
            'job_id': str(jid), 'task': task, 'state_id': str(state_id),
            'window_start': '0', 'window_end': '0',
            'condition': 'clean', 'attack_seed': '0', 'random_control_seed': '',
            'seed': '0', 'candidate_id': '%s_s%d' % (task, state_id),
            'tier': 'clean_scan', 'track': 'S20M4_clean', 'status': 'pending',
        })
        scanned[task] += 1

print('\nClean scan jobs: %d (states 10-49)' % len(jobs))
for task in OBJECT_TASKS:
    print('  %s: %d states to scan' % (task, scanned[task]))

# Split across GPUs for efficiency
# GPU 6,7 first (immediately available), rest join after S20M4a finishes
if jobs:
    for gpuid, fname in [('gpu6', 's20m4_clean_gpu6.csv'),
                          ('gpu7', 's20m4_clean_gpu7_reserve.csv')]:
        pass  # just one queue for now, restart with multiple when S20M4a done

    qp = O+'/queues/s20m4_clean_gpu6.csv'
    with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(jobs)
    print('\nQueue: %s (%d jobs, ~%.1f hrs on 1 GPU pair)' %
          (qp, len(jobs), len(jobs)*2.5/60))
    print('Estimated: %d min = %.1f hrs (2-3 min per clean rollout)' %
          (len(jobs)*2.5, len(jobs)*2.5/60))
else:
    print('\nAll states already covered!')
