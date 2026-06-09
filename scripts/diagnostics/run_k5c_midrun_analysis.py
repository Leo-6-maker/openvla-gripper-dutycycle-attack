#!/usr/bin/env python3
"""K5c mid-run analysis."""
import json, glob, os, sys
from collections import defaultdict, Counter

DIR = '/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e'
summaries = sorted(glob.glob(os.path.join(DIR, 'summary_*.json')))
print(f'Summaries: {len(summaries)}')

# Early termination check
print('\n=== Early termination ===')
early = []
for sp in summaries:
    with open(sp) as f:
        j = json.load(f)
    n_steps = j.get('n_total_steps', 0)
    we = j.get('window_end', 0)
    ws = j.get('window_start', 0)
    req = we + 5
    if n_steps < req:
        early.append(j)
        print(f"  {j.get('pair_id')} job={j.get('job_id')} cond={j.get('condition')} "
              f"atk={j.get('attack_seed')} ws={ws} we={we} n_steps={n_steps} < {req} "
              f"task={j.get('task_key')} infra={j.get('infra_status')}")

if not early:
    print('  None')

# Infra failures
infra_bad = [j for sp in summaries for j in [json.load(open(sp))] if j.get('infra_status') != 'ok']
print(f'\n=== Infra failures: {len(infra_bad)} ===')
for j in infra_bad:
    print(f"  {j.get('pair_id')} job={j.get('job_id')} infra={j.get('infra_status')}")

# Per-parent
print('\n=== Per-parent decoded_open_count ===')
by_parent = defaultdict(list)
for sp in summaries:
    with open(sp) as f:
        j = json.load(f)
    by_parent[j.get('pair_id', '?')].append(j)

for pk in sorted(by_parent.keys()):
    jobs = by_parent[pk]
    vis = [j.get('decoded_open_count', 0) for j in jobs if j.get('condition') == 'vis_pgd']
    rand = [j.get('decoded_open_count', 0) for j in jobs if j.get('condition') == 'random_linf']
    task = jobs[0].get('task_key', '?')
    n_vis = len(vis)
    n_rand = len(rand)
    vis_str = ','.join(str(v) for v in vis)
    rand_str = ','.join(str(r) for r in rand)
    cat = '?'
    if 'rand_' in pk: cat = 'rand_sens'
    elif 'cmd_' in pk: cat = 'cmd_contrast'
    elif 'phys_' in pk: cat = 'strict_phys'
    elif 'sentinel' in pk: cat = 'sentinel'
    print(f"  {pk:35s} {task:15s} {cat:14s} VIS=[{vis_str:20s}] RAND=[{rand_str:20s}]")

# Task distribution
print('\n=== By task ===')
task_counts = Counter()
for pk, jobs in by_parent.items():
    task = jobs[0].get('task_key', '?')
    task_counts[task] += 1
for t, c in task_counts.most_common():
    print(f'  {t}: {c} parents ({c*10} jobs)')
