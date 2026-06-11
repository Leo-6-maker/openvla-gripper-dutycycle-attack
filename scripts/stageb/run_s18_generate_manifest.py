#!/usr/bin/env python3
"""Generate S18 overnight census manifest — 100 jobs, deterministic IDs."""
import csv, os, hashlib, sys

# Verified actual LIBERO Object task order (10 tasks, idx 0-9)
ALL_TASKS = [
    'alphabet_soup', 'cream_cheese', 'salad_dressing', 'bbq_sauce',
    'ketchup', 'tomato_sauce', 'butter', 'milk',
    'chocolate_pudding', 'orange_juice',
]

WINDOWS = [(50,60), (70,80), (90,100), (150,160), (230,240)]
STATE_ID = 0
ATTACK_SEED = 70
EPS = 6
PGD_STEPS = 20
OPEN_DURATION = 10

# GPU allocation: consistent, tomato_sauce prioritized early on GPU26
GPU_TASKS = {
    'gpu10': ALL_TASKS[0:4],   # alphabet_soup, cream_cheese, salad_dressing, bbq_sauce
    'gpu26': ALL_TASKS[4:7],   # ketchup, tomato_sauce, butter
    'gpu45': ALL_TASKS[7:10],  # milk, chocolate_pudding, orange_juice
}

# Deterministic job_id: hash of unique job key → integer in range
def make_job_id(task, ws, we, condition):
    key = f's18_{task}_s0_w{ws}_{we}_{condition}_seed{ATTACK_SEED}'
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(h, 16) % 900000 + 100000  # 100000-999999 range

rows = []
for gpu, tasks in sorted(GPU_TASKS.items()):
    for task in tasks:
        for ws, we in WINDOWS:
            for condition in ['vis_pgd', 'random_linf']:
                job_id = make_job_id(task, ws, we, condition)
                job_uid = f's18_{task}_s{STATE_ID}_w{ws}_{we}_{condition}_seed{ATTACK_SEED}'

                # RAND seed string (must be deterministic per job)
                random_seed_str = str(ATTACK_SEED + job_id)

                pair_uid = f's18_{task}_s{STATE_ID}_w{ws}_{we}_seed{ATTACK_SEED}'

                rows.append({
                    'gpu_group': gpu,
                    'task': task,
                    'state_id': STATE_ID,
                    'window_start': ws,
                    'window_end': we,
                    'condition': condition,
                    'attack_seed': ATTACK_SEED,
                    'job_id': job_id,
                    'job_uid': job_uid,
                    'pair_uid': pair_uid,
                    'random_seed_str': random_seed_str,
                    'eps_raw_pixels': EPS,
                    'pgd_steps': PGD_STEPS,
                    'open_duration': OPEN_DURATION,
                })

# Write manifest
out = os.path.join(os.path.dirname(__file__), '..', '..', 'tables', 's18_jobs_manifest.csv')
out = os.path.abspath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)

fieldnames = ['gpu_group','task','state_id','window_start','window_end','condition',
              'attack_seed','job_id','job_uid','pair_uid','random_seed_str',
              'eps_raw_pixels','pgd_steps','open_duration']

with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f'Manifest: {len(rows)} jobs → {out}')

# Summary
from collections import Counter
by_gpu = Counter(r['gpu_group'] for r in rows)
by_task = Counter(r['task'] for r in rows)
print(f'By GPU: {dict(by_gpu)}')
print(f'By task: {dict(by_task)}')
print(f'Pairs: {len(set(r["pair_uid"] for r in rows))}')
