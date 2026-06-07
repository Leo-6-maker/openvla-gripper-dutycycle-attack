#!/usr/bin/env python3
"""Top up expansion from 47 to 60 paired windows (+13 pairs = 26 jobs)."""
import csv, os, json, random, sys
random.seed(456)

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
OBJ100_DIR = '/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527'
EXISTING_CSV = os.path.join(REPO, 'tables', 'object100_next120_vis_label_jobs.csv')
OUT_CSV = os.path.join(REPO, 'tables', 'object100_next120_vis_label_jobs.csv')  # overwrite
BACKUP_CSV = os.path.join(REPO, 'tables', 'object100_next120_vis_label_jobs_backup.csv')

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

# Backup existing
import shutil
shutil.copy(EXISTING_CSV, BACKUP_CSV)

# Load existing jobs
existing = set()
with open(EXISTING_CSV) as f:
    jobs = list(csv.DictReader(f))
for j in jobs:
    existing.add((j['task_key'], j['state_id'], int(j['window_start']), int(j['window_end'])))

# Also load original 80 jobs
orig_csv = os.path.join(REPO, 'tables', 'object100_next80_vis_label_jobs.csv')
if os.path.exists(orig_csv):
    with open(orig_csv) as f:
        for r in csv.DictReader(f):
            existing.add((r['task_key'], r['state_id'], int(r['window_start']), int(r['window_end'])))

print('Existing windows: %d' % len(existing))

def overlaps(task, sid, ws, we, margin=5):
    for et, es, ews, ewe in existing:
        if et == task and es == sid:
            if not (we + margin < ews or ws - margin > ewe):
                return True
    return False

# Load teacher sanity
with open(os.path.join(REPO, 'tables', 'object100_teacher_window_sanity.csv')) as f:
    sanity = list(csv.DictReader(f))

usable_eps = {}
for r in sanity:
    if r['recommended_use'] in ('use_teacher_window', 'use_reanchored_pre_open_window') \
       and r['mechanism_eligible'].strip().lower() == 'true' \
       and r['clean_success'].strip().lower() == 'true':
        key = (r['task_key'], r['state_id'])
        if key not in usable_eps: usable_eps[key] = []
        usable_eps[key].append(r)

# Count current task distribution
from collections import Counter
task_counts = Counter(j['task_key'] for j in jobs)
print('Current task distribution: %s' % dict(task_counts.most_common()))

# Target: add 13 pairs (26 jobs) to under-covered tasks and strata
# Current strata from expansion: teacher_high, teacher_medium, adjacent_hard_control, early_control
# Need: 5 adjacent, 4 medium, 4 high from under-covered tasks

new_jobs = []
next_jid = max(int(j['job_id']) for j in jobs) + 1
target_pairs = 13
added = 0

# Prioritize under-covered tasks
under_covered = sorted(task_counts.items(), key=lambda x: x[1])
print('Under-covered tasks: %s' % [t for t, c in under_covered if c < 12])

ep_list = list(usable_eps.items())
random.shuffle(ep_list)

for (task, sid), teacher_rows in ep_list:
    if added >= target_pairs: break

    records = None
    try:
        task_full = TASK_SHORT_TO_FULL.get(task, '')
        if task_full:
            task_dir = task_full.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
            path = os.path.join(OBJ100_DIR, 'runs', 'libero_object',
                                'pick_up_the_%s_and_place_it_in_the_basket_state%s' % (task_dir, sid),
                                'step_records.jsonl')
            if os.path.exists(path):
                records = []
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try: records.append(json.loads(line))
                            except: pass
    except: pass
    if records is None: continue
    n_total = len(records)

    tr = random.choice(teacher_rows)
    tws = int(tr['teacher_window_start']); twe = int(tr['teacher_window_end'])
    rec = tr.get('recommended_use', '')
    if 'reanchor' in rec:
        tws = int(tr.get('reanchored_window_start', tws))
        twe = int(tr.get('reanchored_window_end', twe))
    win_len = twe - tws + 1
    mechanism = tr.get('mechanism_type', '')

    # Try different offset positions
    strata_order = ['adjacent_hard_control', 'teacher_medium', 'teacher_high']
    stratum = strata_order[min(added // 5, 2)]  # rough distribution

    offset_candidates = {
        'adjacent_hard_control': [tws - 18, tws - 8, twe + 3, twe + 12, twe + 22],
        'teacher_medium': [tws - 22, tws + 3, twe + 18, tws - 12],
        'teacher_high': [tws - 5, twe + 8, tws + 7],
    }

    for off in offset_candidates.get(stratum, [tws]):
        aws = off; awe = aws + win_len
        if aws < 5 or awe >= n_total - 5: continue
        if overlaps(task, sid, aws, awe, margin=8): continue
        if task_counts[task] >= 16: continue  # avoid task domination

        # VIS
        new_jobs.append({
            'job_id': str(next_jid), 'task_key': task, 'state_id': sid, 'seed': '0',
            'window_start': str(aws), 'window_end': str(awe),
            'stratum': stratum, 'condition': 'vis_pgd', 'matched_random': 'yes',
            'window_len': str(win_len), 'mechanism_type': mechanism,
            'parent_teacher_window': '%d-%d' % (tws, twe),
            'paired_vis_job_id': '', 'worker_id': 0,
        })
        vis_jid = next_jid; next_jid += 1
        new_jobs.append({
            'job_id': str(next_jid), 'task_key': task, 'state_id': sid, 'seed': '0',
            'window_start': str(aws), 'window_end': str(awe),
            'stratum': stratum, 'condition': 'random_linf', 'matched_random': 'yes',
            'window_len': str(win_len), 'mechanism_type': mechanism,
            'parent_teacher_window': '%d-%d' % (tws, twe),
            'paired_vis_job_id': str(vis_jid), 'worker_id': 0,
        })
        next_jid += 1
        added += 1
        task_counts[task] += 2
        existing.add((task, sid, aws, awe))
        break  # one per episode

print('Added %d pairs (%d jobs)' % (added, len(new_jobs)))

# Merge with existing
all_jobs = jobs + new_jobs

# Reshard evenly
vis_all = [j for j in all_jobs if j['condition'] == 'vis_pgd']
jid_to_w = {}
for i, j in enumerate(vis_all):
    j['worker_id'] = i % 3
    jid_to_w[j['job_id']] = j['worker_id']
for j in all_jobs:
    if j['condition'] == 'random_linf':
        pid = j.get('paired_vis_job_id', '')
        j['worker_id'] = jid_to_w.get(pid, 0)

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_jobs[0].keys()))
    w.writeheader(); w.writerows(all_jobs)
print('Wrote %d total jobs to %s' % (len(all_jobs), OUT_CSV))

cts = Counter(j['task_key'] for j in all_jobs)
print('Final task distribution: %s' % dict(cts.most_common()))
total_pairs = sum(1 for j in all_jobs if j['condition'] == 'vis_pgd')
print('Total pairs: %d' % total_pairs)
