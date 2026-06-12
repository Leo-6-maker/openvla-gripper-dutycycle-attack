#!/usr/bin/env python3
"""S20L: Build randhead-screened queue from unified candidate universe (S20K + all prior)."""
import csv, json, glob, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
OUT = '/data/liuyu/outputs/stageb_s20l_randhead_screened_20260613'
os.makedirs(OUT + '/queues', exist_ok=True)

WINDOW_LEN = 10; STRIDE = 5

def detect_phases(rows):
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d
    is_open = [g(r, 'decoded_open_bool') for r in rows]
    fc = None; streak = 0
    for i, o in enumerate(is_open):
        if o == 0:
            if streak == 0: fc = i
            streak += 1
            if streak >= 3: break
        else: streak = 0; fc = None
    if streak < 3: fc = None
    base_obj = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    base_eef = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift = None
    for i in range(fc or 0, len(rows)):
        if g(rows[i], 'obj_z') - base_obj >= 0.015 or g(rows[i], 'eef_z') - base_eef >= 0.03:
            lift = i; break
    pp = None; peak_z = None; peak_step = None
    if lift is not None:
        for i in range(lift, len(rows)):
            z = g(rows[i], 'eef_z')
            if peak_z is None or z > peak_z: peak_z = z; peak_step = i
            if peak_step is not None and i > peak_step + 3 and peak_z - z >= 0.005:
                pp = i; break
    done = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary','0') or '0') == 1: done = i; break
    if done is None:
        for i, r in enumerate(rows):
            if int(r.get('success_done','0') or '0') == 1: done = i; break
    if done is None: done = len(rows) - 1
    return {'fc': fc, 'lift': lift, 'pp': pp, 'done': done, 'max_steps': len(rows)}

def phase_id(ws, we, ph):
    wc = (ws + we) / 2.0
    if wc >= ph['done'] - 5: return 'place_or_done'
    if ph['pp'] is not None and wc >= ph['pp']: return 'preplace'
    if ph['lift'] is not None and wc >= ph['lift'] + 5: return 'transport'
    if ph['lift'] is not None and wc >= ph['lift']: return 'early_transport'
    if ph['fc'] is not None and wc >= ph['fc']: return 'grasp_transition'
    return 'approach'

# Collect ALL clean traces
clean_dirs = [
    '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean',
    '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612',
    '/data/liuyu/outputs/stageb_s20k_clean_expansion_20260613',
]

held_out = {('tomato_sauce','0',70,80), ('ketchup','0',150,160)}
all_candidates = []; clean_summary = []

for d in clean_dirs:
    for summary_f in sorted(glob.glob(d + '/summary_*clean*.json')):
        s = json.load(open(summary_f))
        if not s.get('success_done_any', False): continue
        task = s['task']; sid = str(s['state_id'])
        trace_pat = d + '/trace_%s_s%s_w0_10_s20d_clean_seed0_job*.csv' % (task, sid)
        traces = sorted(glob.glob(trace_pat))
        if not traces: continue
        with open(traces[0]) as f: rows = list(csv.DictReader(f))
        ph = detect_phases(rows)
        clean_summary.append({'task':task,'state_id':sid,'done':ph['done'],'max':ph['max_steps'],'fc':ph['fc'],'lift':ph['lift'],'pp':ph['pp'],'source':d})
        for ws in range(5, ph['max_steps'] - WINDOW_LEN, STRIDE):
            we = ws + WINDOW_LEN
            if we > ph['done'] + 5: continue
            phase = phase_id(ws, we, ph)
            wc = (ws+we)/2.0
            window_rows = [r for r in rows if ws <= int(r['step']) < we]
            def g(row, key, d=0.0):
                try: return float(row.get(key,d) or d)
                except: return d
            clean_open = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
            qpos_vals = [g(r, 'gripper_qpos_before') for r in window_rows]
            eef_vals = [g(r, 'eef_z') for r in window_rows]
            post_grasp = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1 and (ph['fc'] is None or int(r['step']) >= ph['fc']))
            all_candidates.append({
                'task':task,'state_id':sid,'ws':ws,'we':we,'phase':phase,
                'fc':ph['fc'],'lift':ph['lift'],'pp':ph['pp'],'done':ph['done'],
                'rel_timing':wc/max(ph['done'],1),
                'clean_open_count':clean_open,'clean_open_frac':clean_open/WINDOW_LEN,
                'post_grasp_open_count':post_grasp,
                'qpos_mean':float(np.mean(qpos_vals)) if qpos_vals else 0,
                'eef_disp':max(eef_vals)-min(eef_vals) if len(eef_vals)>=2 else 0,
                'source':'s20k' if 's20k' in d else ('s20i_exp' if 's20i_clean' in d else 's20d_orig'),
            })

print('Clean states: %d' % len(clean_summary))
print('Candidates: %d' % len(all_candidates))
tasks = Counter(c['task'] for c in all_candidates)
print('Tasks: %d — %s' % (len(tasks), dict(tasks)))
print('Phases: %s' % dict(Counter(c['phase'] for c in all_candidates)))

# Screen candidates
screened = []
for c in all_candidates:
    phase = c['phase']; open_frac = c['clean_open_frac']
    if phase in ('transport','preplace') and open_frac <= 0.1:
        tier = 'eligible_strict'
    elif phase in ('transport','preplace','grasp_transition','early_transport') and open_frac <= 0.2:
        tier = 'eligible_usable'
    elif phase in ('grasp_transition','early_transport') and open_frac <= 0.3:
        tier = 'eligible_usable'
    else:
        tier = 'predicted_random_sensitive'
    c['tier'] = tier
    screened.append(c)

tier_counts = Counter(c['tier'] for c in screened)
print('Tiers: %s' % dict(tier_counts))

# Exclude already-run
existing = set()
for d in ['/data/liuyu/outputs/stageb_s20j_randhead_screening_20260613',
          '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
          '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612']:
    for f in glob.glob(d + '/summary_*.json'):
        s = json.load(open(f))
        existing.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))

# Select
preferred = ['bbq_sauce','butter','chocolate_pudding','cream_cheese','salad_dressing','orange_juice','alphabet_soup']
limited = ['ketchup','milk','tomato_sauce']

selected = []; task_n = Counter(); phase_n = Counter()
tier_limits = {'eligible_strict': 18, 'eligible_usable': 10, 'predicted_random_sensitive': 8}

def sort_key(c):
    tier_order = {'eligible_strict':0,'eligible_usable':1,'predicted_random_sensitive':2}
    return (tier_order.get(c['tier'],9), c['task'] not in preferred, c['task'] in limited)

pool = sorted([c for c in screened if (c['task'],c['state_id'],c['ws'],c['we']) not in existing], key=sort_key)

for c in pool:
    if len(selected) >= 36: break
    if task_n[c['task']] >= 10: continue
    if phase_n[c['phase']] >= 36 * 0.35: continue
    if len([s for s in selected if s['tier']==c['tier']]) >= tier_limits.get(c['tier'],99): continue
    selected.append(c)
    task_n[c['task']] += 1; phase_n[c['phase']] += 1

print()
print('S20L selected: %d' % len(selected))
print('Tasks: %s' % dict(task_n))
print('Phases: %s' % dict(phase_n))
print('Tiers: %s' % dict(Counter(c['tier'] for c in selected)))

# Build jobs
jobs = []; jid = 260000
for c in selected:
    cid = '%s_s%s_w%d_%d' % (c['task'],c['state_id'],c['ws'],c['we'])
    jid += 1; jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['state_id'],'window_start':str(c['ws']),'window_end':str(c['we']),'condition':'random_linf','attack_seed':'88','random_control_seed':'88','seed':'0','candidate_id':cid,'tier':'L_'+c['tier'],'track':'S20L','status':'pending'})
    jid += 1; jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['state_id'],'window_start':str(c['ws']),'window_end':str(c['we']),'condition':'vis_pgd','attack_seed':'88','random_control_seed':'','seed':'0','candidate_id':cid,'tier':'L_'+c['tier'],'track':'S20L','status':'pending'})

# Split
queues = {'gpu10':[],'gpu26':[],'gpu45':[]}
gpus = ['gpu10','gpu26','gpu45']
for i, j in enumerate(jobs):
    queues[gpus[i%3]].append(j)

for gpu, gpu_jobs in queues.items():
    qpath = OUT + '/queues/s20l_%s.csv' % gpu
    with open(qpath,'w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(jobs[0].keys())); w.writeheader(); w.writerows(gpu_jobs)
    print('%s: %d jobs' % (gpu, len(gpu_jobs)))

print('Total S20L: %d jobs, %d candidates (seed88)' % (len(jobs), len(selected)))
print('Output: %s' % OUT)
