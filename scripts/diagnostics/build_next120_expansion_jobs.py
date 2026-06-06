#!/usr/bin/env python3
"""Build additional 60 paired VIS/random windows for Stage-B expansion."""
import csv, os, json, random
import numpy as np

random.seed(123); np.random.seed(123)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'
OUT_CSV = os.path.join(REPO, 'tables', 'object100_next120_vis_label_jobs.csv')

TASK_SHORT_TO_FULL = {
    'alphabet_soup': 'pick_up_the_alphabet_soup_and_place_it_in_the_basket',
    'bbq_sauce': 'pick_up_the_bbq_sauce_and_place_it_in_the_basket',
    'butter': 'pick_up_the_butter_and_place_it_in_the_basket',
    'cream_cheese': 'pick_up_the_cream_cheese_and_place_it_in_the_basket',
    'ketchup': 'pick_up_the_ketchup_and_place_it_in_the_basket',
    'milk': 'pick_up_the_milk_and_place_it_in_the_basket',
    'orange_juice': 'pick_up_the_orange_juice_and_place_it_in_the_basket',
    'salad_dressing': 'pick_up_the_salad_dressing_and_place_it_in_the_basket',
    'tomato_sauce': 'pick_up_the_tomato_sauce_and_place_it_in_the_basket',
}

def load_step_records(task_short, sid):
    task_full = TASK_SHORT_TO_FULL.get(task_short, '')
    if not task_full: return None
    task_dir = task_full.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
    path = os.path.join(OBJ100_DIR, 'runs', 'libero_object',
                        'pick_up_the_%s_and_place_it_in_the_basket_state%s' % (task_dir, sid),
                        'step_records.jsonl')
    if not os.path.exists(path): return None
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: records.append(json.loads(line))
                except: pass
    return records

def safe_f(v, default=0.0):
    try: return float(v)
    except: return default

# ── Load existing jobs to avoid overlap ──────────────────────────
existing = set()
with open(os.path.join(REPO, 'tables', 'object100_next80_vis_label_jobs.csv')) as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], int(r['window_start']), int(r['window_end']))
        existing.add(key)

def overlaps(task, sid, ws, we, margin=5):
    for et, es, ews, ewe in existing:
        if et == task and es == sid:
            if not (we + margin < ews or ws - margin > ewe):
                return True
    return False

# ── Load teacher sanity for usable episodes ──────────────────────
with open(os.path.join(REPO, 'tables', 'object100_teacher_window_sanity.csv')) as f:
    sanity = list(csv.DictReader(f))

usable_eps = {}
for r in sanity:
    if r['recommended_use'] in ('use_teacher_window', 'use_reanchored_pre_open_window') \
       and r['mechanism_eligible'].strip().lower() == 'true' \
       and r['clean_success'].strip().lower() == 'true':
        key = (r['task_key'], r['state_id'])
        if key not in usable_eps:
            usable_eps[key] = []
        usable_eps[key].append(r)

print('Usable episodes: %d' % len(usable_eps))

# ── Build expansion jobs ─────────────────────────────────────────
jobs = []; job_id = 1000  # start at 1000 to avoid collision
MAX_TEACHER = 30  # 15 high + 15 medium
MAX_ADJACENT = 20
MAX_CONTROL = 10
teacher_count = 0; adjacent_count = 0; control_count = 0

# Shuffle episodes for diversity
ep_list = list(usable_eps.items())
random.shuffle(ep_list)

# Task balance tracking
task_counts = {t: 0 for t in TASK_SHORT_TO_FULL}

for (task, sid), teacher_rows in ep_list:
    if teacher_count >= MAX_TEACHER and adjacent_count >= MAX_ADJACENT and control_count >= MAX_CONTROL:
        break

    records = load_step_records(task, sid)
    if records is None: continue
    n_total = len(records)

    # Pick a representative teacher window
    tr = random.choice(teacher_rows)
    tws = int(tr['teacher_window_start']); twe = int(tr['teacher_window_end'])
    rec = tr.get('recommended_use', '')
    if 'reanchor' in rec:
        tws = int(tr.get('reanchored_window_start', tws))
        twe = int(tr.get('reanchored_window_end', twe))
    window_len = twe - tws + 1
    mechanism = tr.get('mechanism_type', '')

    # Phase boundaries
    qpos_all = [safe_f(rr.get('gripper_qpos', 0)) for rr in records]
    final_release_step = -1
    for i in range(1, len(records)):
        if qpos_all[i-1] < 0.03 and qpos_all[i] > 0.035:
            final_release_step = records[i].get('step_idx', -1)

    # A) Teacher expansion — different adjacent windows near the teacher
    if teacher_count < MAX_TEACHER and task_counts[task] < 6:
        # Find unused offset positions near teacher window
        offsets = [-25, -15, 5, 12, 20]
        for off in offsets:
            ews = tws + off; ewe = ews + window_len
            if ews < 5 or ewe >= n_total - 5: continue
            if overlaps(task, sid, ews, ewe, margin=8): continue
            stratum = 'teacher_high' if teacher_count < 15 else 'teacher_medium'
            # VIS
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(ews), 'window_end': str(ewe),
                'stratum': stratum, 'condition': 'vis_pgd', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': '', 'worker_id': 0,
            })
            vis_jid = job_id; job_id += 1; teacher_count += 1; task_counts[task] += 1
            # Random
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(ews), 'window_end': str(ewe),
                'stratum': stratum, 'condition': 'random_linf', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': str(vis_jid), 'worker_id': 0,
            })
            job_id += 1
            existing.add((task, sid, ews, ewe))
            break  # one per episode

    # B) Adjacent hard controls
    if adjacent_count < MAX_ADJACENT and task_counts[task] < 5:
        adj_positions = [tws - 20, tws - 10, twe + 5, twe + 15]
        for apos in adj_positions:
            aws = apos; awe = apos + window_len
            if aws < 5 or awe >= n_total - 5: continue
            if overlaps(task, sid, aws, awe, margin=8): continue
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(aws), 'window_end': str(awe),
                'stratum': 'adjacent_hard_control', 'condition': 'vis_pgd', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': '', 'worker_id': 0,
            })
            avis_jid = job_id; job_id += 1; adjacent_count += 1; task_counts[task] += 1
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(aws), 'window_end': str(awe),
                'stratum': 'adjacent_hard_control', 'condition': 'random_linf', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': str(avis_jid), 'worker_id': 0,
            })
            job_id += 1
            existing.add((task, sid, aws, awe))
            break

    # C) Control windows
    if control_count < MAX_CONTROL and task_counts[task] < 4:
        cws = None; cwe = None; cstratum = ''
        if final_release_step > 0 and final_release_step + 15 < n_total:
            cws = final_release_step + 8; cwe = cws + window_len
            cstratum = 'post_release_control'
        if (cws is None or cwe >= n_total) and len(qpos_all) > 30:
            cws = random.randint(25, 60); cwe = cws + window_len
            cstratum = 'early_control'
        if cws and cwe and cwe < n_total and not overlaps(task, sid, cws, cwe, margin=8):
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(cws), 'window_end': str(cwe),
                'stratum': cstratum, 'condition': 'vis_pgd', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': '', 'worker_id': 0,
            })
            cvis_jid = job_id; job_id += 1; control_count += 1; task_counts[task] += 1
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(cws), 'window_end': str(cwe),
                'stratum': cstratum, 'condition': 'random_linf', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': str(cvis_jid), 'worker_id': 0,
            })
            job_id += 1
            existing.add((task, sid, cws, cwe))

print('Expansion: %d jobs (%d VIS + %d random)' % (len(jobs),
    sum(1 for j in jobs if j['condition']=='vis_pgd'),
    sum(1 for j in jobs if j['condition']=='random_linf')))

# Shard across 3 workers
vis_jobs = [j for j in jobs if j['condition']=='vis_pgd']
job_id_to_worker = {}
for i, j in enumerate(vis_jobs):
    j['worker_id'] = i % 3
    job_id_to_worker[j['job_id']] = j['worker_id']
for j in jobs:
    if j['condition'] == 'random_linf':
        pid = j.get('paired_vis_job_id', '')
        j['worker_id'] = job_id_to_worker.get(pid, 0)

# Write
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
    w.writeheader(); w.writerows(jobs)
print('Wrote %d jobs to %s' % (len(jobs), OUT_CSV))

# Per-worker summary
from collections import Counter
WORKER_GPU = {0: 'worker_26 (GPU 2,6)', 1: 'worker_45 (GPU 4,5)', 2: 'worker_10 (GPU 1,0)'}
for w in range(3):
    wj = [j for j in jobs if j['worker_id'] == w]
    strata = Counter(j['stratum'] for j in wj)
    conds = Counter(j['condition'] for j in wj)
    print('Worker %d (%s): %d jobs' % (w, WORKER_GPU[w], len(wj)))
    for s, c in strata.most_common():
        print('  %s: %d' % (s, c))
cts = Counter(j['task_key'] for j in jobs)
print('Task balance: %s' % dict(cts))
