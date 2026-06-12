#!/usr/bin/env python3
"""S20M3b freeze + S20M4a RAND-stability screen builder.
1. Write S20M3b confirmation summary & Layer3 registry
2. Select 30 candidates from S20M2+M1 RAND-clean pool
3. Build multi-seed RAND stability queue (seeds 96,97,98)"""
import csv, json, glob, os, numpy as np
from collections import Counter

T = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
O4 = '/data/liuyu/outputs/stageb_s20m4_rand_stability_20260613'
os.makedirs(T, exist_ok=True); os.makedirs(O4+'/queues', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PART A: S20M3b freeze — confirmation summary + Layer3 registry
# ═══════════════════════════════════════════════════════════════

M3B_RESULTS = [
    {'parent_id': 'cream_cheese_s2_w80_90', 'task': 'cream_cheese', 'state_id': '2',
     'window_start': 80, 'window_end': 90, 'phase': 'early_transport',
     's20m3a_class': 'CMD_POSITIVE',
     'seed92_rand': 'STRICT_o0_s0', 'seed93_vis': 'CMD_o8_s8',
     'seed94_rand': 'STRICT_o0_s0', 'seed94_vis': 'NO_EFFECT_o0_s0',
     'seed95_rand': 'STRICT_o0_s0', 'seed95_vis': 'NO_EFFECT_o0_s0',
     'rand_stability': 'STABLE (3/3 STRICT)',
     'vis_reproducibility': '1/3 CMD_POSITIVE',
     'status': 'NOT_CONFIRMED_CMD',
     'reason': 'VIS effect in 1/3 seeds only despite stable RAND baseline',
     'next_action': 'deprioritize; mechanism audit only (window sensitivity)'},
    {'parent_id': 'salad_dressing_s2_w100_110', 'task': 'salad_dressing', 'state_id': '2',
     'window_start': 100, 'window_end': 110, 'phase': 'place_or_done',
     's20m3a_class': 'TASK_EFFECT',
     'seed92_rand': 'STRICT_o0_s0', 'seed93_vis': 'TASK_EFFECT_o3_s1_timeout',
     'seed94_rand': 'BORDERLINE_o7_s4', 'seed94_vis': 'SKIPPED',
     'seed95_rand': 'BORDERLINE_o7_s3', 'seed95_vis': 'SKIPPED',
     'rand_stability': 'UNSTABLE (1/3 STRICT, 2/3 BORDERLINE)',
     'vis_reproducibility': 'cannot assess (2/3 seeds gate-failed)',
     'status': 'NOT_CONFIRMED_TASK_EFFECT',
     'reason': 'RAND baseline unstable; 2/3 seeds BORDERLINE open=7',
     'next_action': 'exclude from claim pool; RAND-unstable parent'}]

with open(T+'/s20m3b_confirmation_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(M3B_RESULTS[0].keys()))
    w.writeheader(); w.writerows(M3B_RESULTS)

with open(T+'/s20m3b_layer3_registry.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['parent_id','task','phase','s20m3a_class','status',
        'rand_stability','vis_reproducibility','reason','next_action'])
    w.writeheader()
    for r in M3B_RESULTS:
        w.writerow({k: r[k] for k in w.fieldnames})

print('=== S20M3b FREEZE ===')
print('Layer3 registry: %s/s20m3b_layer3_registry.csv' % T)
print('Confirmation: %s/s20m3b_confirmation_summary.csv' % T)
for r in M3B_RESULTS:
    print('  %s: %s — %s' % (r['parent_id'], r['status'], r['reason']))

# ═══════════════════════════════════════════════════════════════
# PART B: S20M4a — RAND-stability candidate selection
# ═══════════════════════════════════════════════════════════════

EXCLUDED = {('cream_cheese','2',80,90), ('salad_dressing','2',100,110)}
HELD_OUT = {('tomato_sauce','0',70,80), ('ketchup','0',150,160)}

# Load all existing RAND-tested windows (for exclusion)
existing_rand = set()
rand_dirs = [
    '/data/liuyu/outputs/stageb_s20m1_randonly_calibration_20260613',
    '/data/liuyu/outputs/stageb_s20m2_frozen_forward_20260613',
    '/data/liuyu/outputs/stageb_s20m3b_multiseed_confirmation_20260613',
    '/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
    '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
    '/data/liuyu/outputs/stageb_s20l_v2_randonly_20260613',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
    '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613',
]
for d in rand_dirs:
    if not os.path.exists(d): continue
    for f in glob.glob(d+'/summary_*.json'):
        try:
            s = json.load(open(f))
            existing_rand.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))
        except: pass
    for subd in [d+'/output']:
        if not os.path.exists(subd): continue
        for f in glob.glob(subd+'/summary_*.json'):
            try:
                s = json.load(open(f))
                existing_rand.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))
            except: pass

# Load v0.3.2 frozen scored candidates
scored_map = {}
if os.path.exists(T+'/s20m2_frozen_scored_candidates.csv'):
    with open(T+'/s20m2_frozen_scored_candidates.csv') as f:
        for r in csv.DictReader(f):
            scored_map[(r['task'], r['state_id'], int(r['ws']), int(r['we']))] = r

# Also load S20M1 manifest
m1_map = {}
if os.path.exists(T+'/s20m1_calibration_manifest.csv'):
    with open(T+'/s20m1_calibration_manifest.csv') as f:
        for r in csv.DictReader(f):
            m1_map[(r['task'], r['state_id'], int(r['ws']), int(r['we']))] = r

# Phase detection (shared)
def detect_phases(rows):
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d
    is_open = [g(r, 'decoded_open_bool') for r in rows]
    fc = None; stk = 0
    for i, o in enumerate(is_open):
        if o == 0:
            if stk == 0: fc = i
            stk += 1
            if stk >= 3: break
        else: stk = 0; fc = None
    if stk < 3: fc = None
    bo = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    be = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift = None
    for i in range(fc or 0, len(rows)):
        if g(rows[i], 'obj_z')-bo >= 0.015 or g(rows[i], 'eef_z')-be >= 0.03:
            lift = i; break
    pp = None; pz = None; ps = None
    if lift is not None:
        for i in range(lift, len(rows)):
            z = g(rows[i], 'eef_z')
            if pz is None or z > pz: pz = z; ps = i
            if ps is not None and i > ps+3 and pz-z >= 0.005:
                pp = i; break
    done = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary','0') or '0') == 1 or int(r.get('success_done','0') or '0') == 1:
            done = i; break
    if done is None: done = len(rows)-1
    return {'fc': fc, 'lift': lift, 'pp': pp, 'done': done, 'ms': len(rows)}

def phase_id(ws, we, ph):
    wc = (ws+we)/2.0
    if wc >= ph['done']-5: return 'place_or_done'
    if ph['pp'] is not None and wc >= ph['pp']: return 'preplace'
    if ph['lift'] is not None and wc >= ph['lift']+5: return 'transport'
    if ph['lift'] is not None and wc >= ph['lift']: return 'early_transport'
    if ph['fc'] is not None and wc >= ph['fc']: return 'grasp_transition'
    return 'approach'

# Priority tasks (val+test, exclude train)
PRIORITY_TASKS = {'alphabet_soup','bbq_sauce','butter','chocolate_pudding',
                   'cream_cheese','orange_juice','salad_dressing'}
TRAIN_TASKS = {'ketchup','milk','tomato_sauce'}

WINDOW = 10; STRIDE = 5
clean_dirs = [
    '/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
    '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612',
    '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean',
]

# Collect all untested candidates with priority scoring
candidates = []
for d in clean_dirs:
    if not os.path.exists(d): continue
    for sf in sorted(glob.glob(d+'/summary_*clean*.json')):
        s = json.load(open(sf))
        if not s.get('success_done_any', False): continue
        task = s['task']; sid = str(s['state_id'])
        if task in TRAIN_TASKS: continue  # focus on fresh tasks
        tp = d+'/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv'%(task, sid)
        tr = sorted(glob.glob(tp))
        if not tr: continue
        with open(tr[0]) as f: rows = list(csv.DictReader(f))
        ph = detect_phases(rows)
        for ws in range(5, ph['ms']-WINDOW, STRIDE):
            we = ws+WINDOW
            if we > ph['done']+5: continue
            key = (task, sid, ws, we)
            if key in HELD_OUT: continue
            if key in EXCLUDED: continue
            if key in existing_rand: continue

            phase = phase_id(ws, we, ph)
            window_rows = [r for r in rows if ws <= int(r['step']) < we]
            def g(row, key_col, d=0.0):
                try: return float(row.get(key_col, d) or d)
                except: return d
            clean_open = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
            clean_frac = clean_open/WINDOW

            # Priority scoring
            score = 0
            if task in PRIORITY_TASKS: score += 10
            if clean_open == 0: score += 5
            elif clean_open <= 1: score += 3
            if phase in ('transport','preplace'): score += 4
            elif phase in ('grasp_transition','early_transport'): score += 2

            # Check v0.3.2 score if available
            sc = scored_map.get(key, {})
            p_rand = float(sc.get('p_random_sensitive', 0.5) or 0.5)
            m1 = m1_map.get(key, {})
            tier = sc.get('tier', m1.get('tier', '?'))

            candidates.append({
                'task': task, 'state_id': sid, 'ws': ws, 'we': we,
                'phase': phase, 'clean_open': clean_open, 'clean_frac': clean_frac,
                'p_rand': p_rand, 'tier': tier, 'priority_score': score,
            })

print('\n=== S20M4a CANDIDATE POOL ===')
print('Total untested fresh candidates: %d' % len(candidates))
task_counts = Counter(c['task'] for c in candidates)
print('By task: %s' % dict(task_counts))

# Sort by priority score (desc), then p_rand (asc)
candidates.sort(key=lambda c: (-c['priority_score'], c['p_rand']))

# Select: target 30, max 6/task, max 3/(task,state)
TARGET = 30
MAX_PER_TASK = 6
selected = []; task_n = Counter(); adj_n = Counter()

for c in candidates:
    if len(selected) >= TARGET: break
    if task_n[c['task']] >= MAX_PER_TASK: continue
    adj_key = (c['task'], c['state_id'])
    if adj_n[adj_key] >= 3: continue
    # Phase diversity: prefer under-represented phases
    phase_counts = Counter(s['phase'] for s in selected)
    if phase_counts[c['phase']] >= TARGET * 0.30: continue

    selected.append(c)
    task_n[c['task']] += 1; adj_n[adj_key] += 1

# If short, relax constraints
if len(selected) < 24:
    print('Relaxing constraints for coverage...')
    for c in candidates:
        if len(selected) >= 30: break
        if any(s['task']==c['task'] and s['state_id']==c['state_id'] and s['ws']==c['ws'] for s in selected):
            continue
        if task_n[c['task']] >= 8: continue
        adj_key = (c['task'], c['state_id'])
        if adj_n[adj_key] >= 4: continue
        selected.append(c)
        task_n[c['task']] += 1; adj_n[adj_key] += 1

print('\nS20M4a selected: %d candidates' % len(selected))
print('Tasks: %s' % dict(task_n))
print('Phases: %s' % dict(Counter(c['phase'] for c in selected)))
print('Tiers: %s' % dict(Counter(c['tier'] for c in selected)))

# Write manifest
with open(T+'/s20m4a_rand_stability_manifest.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['candidate_id','task','state_id','ws','we','phase',
        'clean_open','clean_frac','p_rand','tier','priority_score'])
    w.writeheader()
    for c in selected:
        cid = '%s_s%s_w%d_%d' % (c['task'], c['state_id'], c['ws'], c['we'])
        w.writerow({**c, 'candidate_id': cid})

# ═══════════════════════════════════════════════════════════════
# Build RAND-only queues (seeds 96,97,98 per candidate)
# ═══════════════════════════════════════════════════════════════
SEEDS = ['96', '97', '98']
jobs = []; jid = 320000

for c in selected:
    cid = '%s_s%s_w%d_%d' % (c['task'], c['state_id'], c['ws'], c['we'])
    for seed in SEEDS:
        jid += 1
        jobs.append({
            'job_id': str(jid), 'task': c['task'], 'state_id': c['state_id'],
            'window_start': str(c['ws']), 'window_end': str(c['we']),
            'condition': 'random_linf', 'attack_seed': seed, 'random_control_seed': seed,
            'seed': '0', 'candidate_id': cid,
            'tier': 'M4a_'+c['tier'], 'track': 'S20M4a', 'status': 'pending',
        })

# Split across 3 GPUs
queues = {'gpu0': [], 'gpu2': [], 'gpu4': []}
gpus = ['gpu0', 'gpu2', 'gpu4']
for i, j in enumerate(jobs):
    queues[gpus[i % 3]].append(j)

for gpu, gj in queues.items():
    qp = O4+'/queues/s20m4a_rand_%s.csv' % gpu
    with open(qp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(gj)
    print('%s: %d RAND jobs' % (gpu, len(gj)))

# Summary
print()
print('='*60)
print('S20M4a RAND STABILITY SCREEN')
print('='*60)
print('Candidates: %d × 3 seeds = %d RAND-only jobs' % (len(selected), len(jobs)))
print('Seeds: 96, 97, 98')
print('GPUs: 0,1 | 2,3 | 4,5')
print()
print('Stability criteria:')
print('  RAND_STABLE_STRICT: 3/3 STRICT, open<=1, streak<=1, no timeout')
print('  RAND_STABLE_USABLE: >=2/3 STRICT/USABLE, no RS seed')
print('  RAND_UNSTABLE: any 2/3 BORDERLINE or RS')
print()
print('Manifest: %s/s20m4a_rand_stability_manifest.csv' % T)
print('Queues: %s/queues/' % O4)
print()
print('=== S20M3b FREEZE (above) + S20M4a QUEUES (below) ===')
