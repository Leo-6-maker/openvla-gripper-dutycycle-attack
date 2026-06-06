#!/usr/bin/env python3
"""Build Next80 VIS label jobs with hard controls for Stage-B labeling."""
import csv, os, sys, json, random
import numpy as np

random.seed(42); np.random.seed(42)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'
OUT_DIR = '/data/liuyu/outputs/overnight_stageb_labels_20260607'
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Load teacher sanity ──────────────────────────────────────────
with open(os.path.join(REPO, 'tables', 'object100_teacher_window_sanity.csv')) as f:
    sanity = list(csv.DictReader(f))

# Filter usable
usable = [r for r in sanity if r['recommended_use'] in ('use_teacher_window', 'use_reanchored_pre_open_window')
          and r['mechanism_eligible'].strip().lower() == 'true'
          and r['clean_success'].strip().lower() == 'true']
print('Usable teacher windows: %d from %d episodes' % (len(usable), len(set((r['task_key'],r['state_id']) for r in usable))))

# ── Select windows ───────────────────────────────────────────────
jobs = []; job_id = 0

# Shuffle usable for random selection
random.shuffle(usable)

# Stratum limits
MAX_TEACHER = 40  # 20 high + 20 medium
MAX_ADJACENT = 20
MAX_CONTROL = 20

teacher_count = 0; adjacent_count = 0; control_count = 0
episodes_used = set()

for r in usable:
    task = r['task_key']; sid = r['state_id']
    ep_key = (task, sid)
    tws = int(r['teacher_window_start']); twe = int(r['teacher_window_end'])
    rec = r.get('recommended_use', '')
    if 'reanchor' in rec:
        tws = int(r.get('reanchored_window_start', tws))
        twe = int(r.get('reanchored_window_end', twe))
    window_len = twe - tws + 1
    mechanism = r.get('mechanism_type', '')

    records = load_step_records(task, sid)
    if records is None: continue
    n_total = len(records)

    # Find phase boundaries
    qpos_all = [safe_f(rr.get('gripper_qpos', 0)) for rr in records]
    final_release_step = -1
    for i in range(1, len(records)):
        if qpos_all[i-1] < 0.03 and qpos_all[i] > 0.035:
            final_release_step = records[i].get('step_idx', -1)
    first_approach_open = -1
    for i in range(len(records)):
        if qpos_all[i] > 0.03 and safe_f(records[i].get('gripper_command', 0)) > 0:
            first_approach_open = records[i].get('step_idx', -1)
            break

    # A) Teacher window (high score) — up to 20
    if teacher_count < MAX_TEACHER:
        jobs.append({
            'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
            'window_start': str(tws), 'window_end': str(twe),
            'stratum': 'teacher_high' if teacher_count < 20 else 'teacher_medium',
            'condition': 'vis_pgd', 'matched_random': 'yes',
            'window_len': str(window_len), 'mechanism_type': mechanism,
            'parent_teacher_window': '%d-%d' % (tws, twe), 'paired_vis_job_id': '', 'worker_id': 0,
        })
        vis_jid = job_id; job_id += 1; teacher_count += 1

        # Also add matched random job
        jobs.append({
            'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
            'window_start': str(tws), 'window_end': str(twe),
            'stratum': 'teacher_high' if teacher_count <= 20 else 'teacher_medium',
            'condition': 'random_linf', 'matched_random': 'yes',
            'window_len': str(window_len), 'mechanism_type': mechanism,
            'parent_teacher_window': '%d-%d' % (tws, twe),
            'paired_vis_job_id': str(vis_jid),
        })
        job_id += 1
        episodes_used.add(ep_key)

    # B) Adjacent hard control — same length, just before or after teacher window
    if adjacent_count < MAX_ADJACENT and teacher_count <= MAX_TEACHER:
        # Before: [tws - window_len - 5, tws - 5]
        adj_ws = max(5, tws - window_len - 5)
        adj_we = adj_ws + window_len
        if adj_we < tws - 2 and adj_we < n_total:
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(adj_ws), 'window_end': str(adj_we),
                'stratum': 'adjacent_hard_control',
                'condition': 'vis_pgd', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe), 'paired_vis_job_id': '',
            })
            adj_vis_jid = job_id; job_id += 1; adjacent_count += 1
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(adj_ws), 'window_end': str(adj_we),
                'stratum': 'adjacent_hard_control',
                'condition': 'random_linf', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': str(adj_vis_jid),
            })
            job_id += 1

    # C) Post-release or early pregrasp control
    if control_count < MAX_CONTROL:
        if final_release_step > 0 and final_release_step + window_len + 5 < n_total:
            cws = final_release_step + 5
            cwe = cws + window_len
        elif first_approach_open > 15:
            cws = max(5, first_approach_open - 15)
            cwe = cws + window_len
        else:
            continue
        if cwe < n_total:
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(cws), 'window_end': str(cwe),
                'stratum': 'post_release_or_early_control',
                'condition': 'vis_pgd', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe), 'paired_vis_job_id': '',
            })
            ctrl_vis_jid = job_id; job_id += 1; control_count += 1
            jobs.append({
                'job_id': str(job_id), 'task_key': task, 'state_id': sid, 'seed': '0',
                'window_start': str(cws), 'window_end': str(cwe),
                'stratum': 'post_release_or_early_control',
                'condition': 'random_linf', 'matched_random': 'yes',
                'window_len': str(window_len), 'mechanism_type': mechanism,
                'parent_teacher_window': '%d-%d' % (tws, twe),
                'paired_vis_job_id': str(ctrl_vis_jid),
            })
            job_id += 1

    if teacher_count >= MAX_TEACHER and adjacent_count >= MAX_ADJACENT and control_count >= MAX_CONTROL:
        break

print('Jobs: %d total (VIS=%d, random=%d)' % (len(jobs),
    sum(1 for j in jobs if j['condition'] == 'vis_pgd'),
    sum(1 for j in jobs if j['condition'] == 'random_linf')))

# ── Shard across 3 workers ──────────────────────────────────────
# Group VIS+random pairs together on same worker
vis_jobs = [j for j in jobs if j['condition'] == 'vis_pgd']
random_jobs = [j for j in jobs if j['condition'] == 'random_linf']

# Shard: distribute VIS jobs across workers, pair random jobs to same worker
n_workers = 3
for i, j in enumerate(vis_jobs):
    j['worker_id'] = i % n_workers
for j in random_jobs:
    paired_id = int(j.get('paired_vis_job_id', -1))
    if paired_id >= 0 and paired_id < len(vis_jobs):
        j['worker_id'] = vis_jobs[paired_id]['worker_id']
    else:
        j['worker_id'] = 0

# Worker GPU mapping
WORKER_GPU = {0: 'worker_26 (GPU 2,6)', 1: 'worker_45 (GPU 4,5)', 2: 'worker_10 (GPU 1,0)'}

# ── Write jobs CSV ───────────────────────────────────────────────
JOB_CSV = os.path.join(REPO, 'tables', 'object100_next80_vis_label_jobs.csv')
keys = list(jobs[0].keys())
with open(JOB_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(jobs)
print('Wrote %d jobs to %s' % (len(jobs), JOB_CSV))

# ── Per-worker summary ───────────────────────────────────────────
from collections import Counter
for w in range(n_workers):
    wj = [j for j in jobs if j['worker_id'] == w]
    strata = Counter(j['stratum'] for j in wj)
    conds = Counter(j['condition'] for j in wj)
    print('\nWorker %d (%s): %d jobs' % (w, WORKER_GPU[w], len(wj)))
    for s, c in strata.most_common():
        print('  %s: %d' % (s, c))

# ── Final plan report ────────────────────────────────────────────
with open(os.path.join(REPO, 'reports', 'OBJECT100_NEXT80_VIS_LABEL_PLAN_FINAL.md'), 'w') as f:
    f.write('# Next80 VIS Label Plan — Final\n\n')
    f.write('**Jobs**: %d (VIS PGD20 + matched random Linf pairs)\n\n' % len(vis_jobs))
    f.write('## Strata\n\n')
    for s, c in Counter(j['stratum'] for j in jobs if j['condition']=='vis_pgd').most_common():
        f.write('- **%s**: %d windows\n' % (s, c))
    f.write('\n## Worker Distribution\n\n')
    for w in range(n_workers):
        wj = [j for j in jobs if j['worker_id'] == w]
        f.write('- **Worker %d (%s)**: %d jobs\n' % (w, WORKER_GPU[w], len(wj)))
    f.write('\n## Hard Controls Rationale\n\n')
    f.write('Adjacent hard controls: same window length as teacher window, positioned\n')
    f.write('just before (pre-teacher) or after (post-teacher). Same time band,\n')
    f.write('same length, non-overlapping. These test whether the localizer\n')
    f.write('distinguishes opportunity from nearby non-opportunity.\n')
print('Done')
