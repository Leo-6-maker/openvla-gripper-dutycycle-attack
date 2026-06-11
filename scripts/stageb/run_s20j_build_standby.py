#!/usr/bin/env python3
"""S20J-B: Build standby refill queues. Task/phase balanced, gate-controlled."""
import csv, os, numpy as np
from collections import Counter, defaultdict

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613/queues'
os.makedirs(OUT, exist_ok=True)

# Load eligible candidates
eligible = list(csv.DictReader(open(TABLES + '/s20i_v031_non_random_sensitive_candidates.csv')))
# Load expansion candidates
expansion = list(csv.DictReader(open(TABLES + '/s20i_clean_expansion_candidate_universe.csv')))

# Load current S20J queue keys (exclude already running)
current_keys = set()
for gpu in ['gpu10', 'gpu26', 'gpu45']:
    qpath = OUT + '/s20j_%s.csv' % gpu
    if os.path.exists(qpath):
        with open(qpath) as f:
            for r in csv.DictReader(f):
                current_keys.add((r['task'], r['state_id'], r['window_start'], r['window_end'], r['attack_seed'], r['condition']))

held_out = {('tomato_sauce', '0', '70', '80'), ('ketchup', '0', '150', '160')}

# Build candidate pool with tier and task info
pool = []
for e in eligible:
    key = (e['task'], e['state_id'], e['window_start'], e['window_end'])
    if key in held_out: continue
    if (e['task'], e['state_id'], e['window_start'], e['window_end'], '86', 'random_linf') in current_keys: continue
    pool.append({'task': e['task'], 'state_id': e['state_id'], 'ws': e['window_start'], 'we': e['window_end'],
                 'phase': e['phase'], 'tier': e['eligible_tier'], 'source': 's20j_screened'})

# Add expansion candidates (non-random-sensitive only)
for e in expansion:
    key = (e['task'], e['state_id'], e['window_start'], e['window_end'])
    if key in held_out: continue
    if (e['task'], e['state_id'], e['window_start'], e['window_end'], '86', 'random_linf') in current_keys: continue
    phase = e['phase']
    open_frac = float(e.get('clean_open_frac', 0) or 0)
    if phase in ('transport', 'preplace') and open_frac <= 0.2:
        tier = 'eligible_usable'
    elif phase in ('grasp_transition', 'early_transport') and open_frac <= 0.3:
        tier = 'eligible_usable'
    else:
        tier = 'predicted_random_sensitive'
    pool.append({'task': e['task'], 'state_id': e['state_id'], 'ws': e['window_start'], 'we': e['window_end'],
                 'phase': phase, 'tier': tier, 'source': 's20i_expansion'})

print('Pool: %d candidates (%d eligible_strict, %d eligible_usable, %d disagree, %d pred_rand)' % (
    len(pool),
    sum(1 for p in pool if p['tier'] == 'eligible_strict'),
    sum(1 for p in pool if p['tier'] == 'eligible_usable'),
    sum(1 for p in pool if p['tier'] == 'disagreement'),
    sum(1 for p in pool if p['tier'] == 'predicted_random_sensitive')))

# Sampling targets
targets = {
    'eligible_strict': 18,
    'eligible_usable': 8,
    'disagreement': 5,
    'predicted_random_sensitive': 4,
}

# Constraints: max 5 per task, max 35% per phase, prefer expansion tasks
preferred_tasks = ['cream_cheese', 'salad_dressing', 'orange_juice', 'alphabet_soup', 'milk']
limited_tasks = ['ketchup', 'tomato_sauce']
max_per_task = 5
max_phase_frac = 0.35

selected = []
task_counts = Counter()
phase_counts = Counter()
tier_counts = Counter()

# Flatten: pick from all tiers, preferring strict > usable > disagree > pred_rand
tier_order = ['eligible_strict', 'eligible_usable', 'disagreement', 'predicted_random_sensitive']
pool.sort(key=lambda p: (
    tier_order.index(p['tier']) if p['tier'] in tier_order else 99,
    p['task'] not in preferred_tasks,
    p['task'] in limited_tasks,
))
np.random.seed(42)
# Shuffle within same priority
groups = defaultdict(list)
for p in pool:
    priority = (tier_order.index(p['tier']) if p['tier'] in tier_order else 99,
               p['task'] not in preferred_tasks)
    groups[priority].append(p)
for g in groups.values():
    np.random.shuffle(g)
# Flatten back
pool = []
for priority in sorted(groups.keys()):
    pool.extend(groups[priority])

target_total = 35
for p in pool:
    if len(selected) >= target_total:
        break
    if task_counts[p['task']] >= max_per_task:
        continue
    if phase_counts[p['phase']] >= target_total * max_phase_frac:
        continue
    selected.append(p)
    task_counts[p['task']] += 1
    phase_counts[p['phase']] += 1
    tier_counts[p['tier']] += 1

print('\nSelected: %d candidates' % len(selected))
print('Tasks: %s' % dict(task_counts))
print('Phases: %s' % dict(phase_counts))
print('Tiers: %s' % dict(Counter(p['tier'] for p in selected)))

# Build jobs
jobs = []
jid = 242000
for p in selected:
    cid = '%s_s%s_w%s_%s' % (p['task'], p['state_id'], p['ws'], p['we'])
    tier = p['tier']
    jid += 1; jobs.append({'job_id':str(jid),'task':p['task'],'state_id':p['state_id'],'window_start':p['ws'],'window_end':p['we'],'condition':'random_linf','attack_seed':'87','random_control_seed':'87','seed':'0','candidate_id':cid,'tier':'JB_'+tier,'track':'S20JB_standby','status':'pending'})
    if tier != 'predicted_random_sensitive':
        jid += 1; jobs.append({'job_id':str(jid),'task':p['task'],'state_id':p['state_id'],'window_start':p['ws'],'window_end':p['we'],'condition':'vis_pgd','attack_seed':'87','random_control_seed':'','seed':'0','candidate_id':cid,'tier':'JB_'+tier,'track':'S20JB_standby','status':'pending'})

# Split across 3 GPUs
queues = {'gpu10': [], 'gpu26': [], 'gpu45': []}
gpus = ['gpu10', 'gpu26', 'gpu45']
pairs = [(jobs[i], jobs[i+1]) for i in range(0, len(jobs)-1, 2) if i+1 < len(jobs) and jobs[i+1]['condition'] == 'vis_pgd']
for i, (rj, vj) in enumerate(pairs):
    queues[gpus[i % 3]].append(rj)
    queues[gpus[i % 3]].append(vj)

# Add orphan RAND-only jobs
orphans = [j for j in jobs if j['condition'] == 'random_linf' and not any(j is rj or j is vj for rj, vj in pairs for pair_jobs in [(rj, vj)] if j is rj or j is vj)]
for i, j in enumerate(orphans):
    queues[gpus[i % 3]].append(j)

for gpu, gpu_jobs in queues.items():
    if not gpu_jobs: continue
    qpath = OUT + '/s20j_%s_refill_001.csv' % gpu
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(gpu_jobs)
    n_rand = sum(1 for j in gpu_jobs if j['condition'] == 'random_linf')
    print('%s refill: %d jobs (%d RAND + %d VIS)' % (gpu, len(gpu_jobs), n_rand, len(gpu_jobs) - n_rand))

# Write master standby manifest
with open(TABLES + '/s20j_standby_refill_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)

print('\nTotal standby: %d jobs, %d candidates' % (len(jobs), len(selected)))
print('Launch gate: eligible_strict RAND-clean precision >= 60% (>=20 RAND jobs)')
