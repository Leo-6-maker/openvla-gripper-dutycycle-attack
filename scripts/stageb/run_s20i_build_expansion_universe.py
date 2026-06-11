#!/usr/bin/env python3
"""Build candidate universe from S20I clean expansion traces, generate refill queues."""
import csv, os, glob, json
import numpy as np
from collections import Counter

OUT = '/data/liuyu/outputs/stageb_s20i_clean_expansion_20260612'
QUEUES_DIR = '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612/queues'
TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

WINDOW_LEN = 10
STRIDE = 5
MIN_WS = 5
TASKS = ['milk', 'cream_cheese', 'salad_dressing', 'bbq_sauce', 'orange_juice', 'alphabet_soup']

# Phase detection (same as S20F universe builder)
def detect_phases(rows):
    def g(row, key, d=0.0):
        try: return float(row.get(key, d) or d)
        except: return d
    is_open = [g(r, 'decoded_open_bool') for r in rows]
    eef_z_vals = [g(r, 'eef_z') for r in rows]
    obj_z_vals = [g(r, 'obj_z') for r in rows]
    first_close = None; streak = 0
    for i, o in enumerate(is_open):
        if o == 0:
            if streak == 0: first_close = i
            streak += 1
            if streak >= 3: break
        else: streak = 0; first_close = None
    if streak < 3: first_close = None
    base_obj = float(np.median([g(r, 'obj_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    base_eef = float(np.median([g(r, 'eef_z') for r in rows[:5]])) if len(rows) >= 5 else 0.0
    lift_step = None
    for i in range(first_close or 0, len(rows)):
        if g(rows[i], 'obj_z') - base_obj >= 0.015 or g(rows[i], 'eef_z') - base_eef >= 0.03:
            lift_step = i; break
    preplace_step = None; peak_step = None; peak_z = None
    if lift_step is not None:
        for i in range(lift_step, len(rows)):
            z = g(rows[i], 'eef_z')
            if peak_z is None or z > peak_z: peak_z = z; peak_step = i
            if peak_step is not None and i > peak_step + 3 and peak_z - z >= 0.005:
                preplace_step = i; break
    done_step = None
    for i, r in enumerate(rows):
        if int(r.get('success_primary', '0') or '0') == 1 or int(r.get('success_done', '0') or '0') == 1:
            done_step = i; break
    if done_step is None: done_step = len(rows) - 1
    return {'first_close_step': first_close, 'lift_step': lift_step, 'preplace_step': preplace_step, 'done_step': done_step, 'base_obj_z': base_obj, 'base_eef_z': base_eef, 'max_steps': len(rows)}

def classify_phase(ws, we, phases):
    fc = phases['first_close_step']; lift = phases['lift_step']
    pp = phases['preplace_step']; done = phases['done_step']; wc = (ws + we) / 2.0
    if wc >= done - 5: return 'place_or_done'
    if pp is not None and wc >= pp: return 'preplace'
    if lift is not None and wc >= lift + 5: return 'transport'
    if lift is not None and wc >= lift: return 'early_transport'
    if fc is not None and wc >= fc: return 'grasp_transition'
    return 'approach'

# Load existing reserved parents (all queues + prior summaries)
reserved_parents = set()
for d in ['/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
          '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
          '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
          '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612']:
    for f in glob.glob(d + '/summary_*.json'):
        s = json.load(open(f))
        reserved_parents.add((s['task'], str(s['state_id']), s['window_start'], s['window_end']))
for q in glob.glob(QUEUES_DIR + '/s20i_gpu*.csv'):
    with open(q) as f:
        for r in csv.DictReader(f):
            reserved_parents.add((r['task'], r['state_id'], int(r['window_start']), int(r['window_end'])))

held_out = {('tomato_sauce', '0', 70, 80), ('ketchup', '0', 150, 160)}

clean_summary = []
all_candidates = []

for task in TASKS:
    trace_path = OUT + '/trace_%s_s1_w0_10_s20d_clean_seed0_job23*.csv' % task
    trace_files = sorted(glob.glob(trace_path))
    if not trace_files:
        print('[%s] No trace found' % task)
        continue

    with open(trace_files[0]) as f:
        rows = list(csv.DictReader(f))
    phases = detect_phases(rows)
    success = any(r.get('success_primary') == '1' for r in rows)
    done_step = phases['done_step']
    max_step = phases['max_steps']

    clean_summary.append({
        'task': task, 'state_id': '1', 'success': success,
        'done_step': done_step, 'max_steps': max_step,
        'first_close': phases['first_close_step'], 'lift': phases['lift_step'],
        'preplace': phases['preplace_step'],
    })
    status = 'SUCCESS' if success else 'FAIL'
    print('[%s] %s done=%d max=%d fc=%s lift=%s pp=%s' % (
        task, status, done_step, max_step, phases['first_close_step'], phases['lift_step'], phases['preplace_step']))

    if not success: continue

    for ws in range(MIN_WS, max_step - WINDOW_LEN, STRIDE):
        we = ws + WINDOW_LEN
        if we > done_step + 5: continue
        phase = classify_phase(ws, we, phases)
        if (task, '1', ws, we) in held_out: continue
        if (task, '1', ws, we) in reserved_parents: continue
        all_candidates.append({
            'task': task, 'state_id': '1', 'window_start': ws, 'window_end': we,
            'window_len': WINDOW_LEN, 'phase': phase,
            'first_close_step': phases['first_close_step'],
            'lift_step': phases['lift_step'],
            'preplace_step': phases['preplace_step'],
            'done_step': done_step,
        })

# Write clean summary
with open(TABLES + '/s20i_clean_expansion_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','success','done_step','max_steps','first_close','lift','preplace'])
    w.writeheader(); w.writerows(clean_summary)

# Write candidate universe
with open(TABLES + '/s20i_clean_expansion_candidate_universe.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task','state_id','window_start','window_end','window_len','phase','first_close_step','lift_step','preplace_step','done_step'])
    w.writeheader(); w.writerows(all_candidates)

# Phase coverage
phase_counts = Counter(c['phase'] for c in all_candidates)
task_counts = Counter(c['task'] for c in all_candidates)
with open(TABLES + '/s20i_clean_expansion_phase_coverage.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['task','phase','n'])
    for t in sorted(task_counts):
        for p in ['approach','grasp_transition','early_transport','transport','preplace','place_or_done']:
            n = sum(1 for c in all_candidates if c['task'] == t and c['phase'] == p)
            if n > 0: w.writerow([t, p, n])

print('\nTotal new candidates: %d (across %d tasks)' % (len(all_candidates), len(task_counts)))
print('Phases: %s' % dict(phase_counts))
print('Tasks: %s' % dict(task_counts))

# Build refill Track C queues: 1 per phase per task (seed85)
refill_jobs = []
jid = 230000
for phase in ['grasp_transition', 'early_transport', 'transport', 'preplace']:
    for task in TASKS:
        phase_cands = [c for c in all_candidates if c['task'] == task and c['phase'] == phase]
        if not phase_cands: continue
        c = phase_cands[0]  # pick first per phase per task
        cid = '%s_s%s_w%d_%d' % (c['task'], c['state_id'], c['window_start'], c['window_end'])
        jid += 1; refill_jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['state_id'],'window_start':str(c['window_start']),'window_end':str(c['window_end']),'condition':'random_linf','attack_seed':'85','random_control_seed':'85','seed':'0','candidate_id':cid,'tier':'C_refill_'+phase,'track':'C_refill_expansion','status':'pending'})
        jid += 1; refill_jobs.append({'job_id':str(jid),'task':c['task'],'state_id':c['state_id'],'window_start':str(c['window_start']),'window_end':str(c['window_end']),'condition':'vis_pgd','attack_seed':'85','random_control_seed':'','seed':'0','candidate_id':cid,'tier':'C_refill_'+phase,'track':'C_refill_expansion','status':'pending'})

# Split across 3 GPUs
refill_qs = {'gpu10': [], 'gpu26': [], 'gpu45': []}
gpus = ['gpu10', 'gpu26', 'gpu45']
pairs = [(refill_jobs[i], refill_jobs[i+1]) for i in range(0, len(refill_jobs), 2)]
for i, (rj, vj) in enumerate(pairs):
    refill_qs[gpus[i % 3]].append(rj)
    refill_qs[gpus[i % 3]].append(vj)

for gpu, jobs in refill_qs.items():
    if not jobs: continue
    qpath = QUEUES_DIR + '/s20i_%s_expansion_refill.csv' % gpu
    with open(qpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader(); w.writerows(jobs)
    print('%s expansion refill: %d jobs (%d candidates)' % (gpu, len(jobs), len(jobs)//2))

with open(TABLES + '/s20i_refill_from_clean_expansion_queue.csv', 'w', newline='') as f:
    fields = list(refill_jobs[0].keys()) if refill_jobs else []
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(refill_jobs)

print('Refill total: %d jobs, %d unique candidates' % (len(refill_jobs), len(refill_jobs)//2))
print('Done.')
