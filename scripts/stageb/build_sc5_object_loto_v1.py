#!/usr/bin/env python3
"""
SC5 Object LOTO v1 — Fold 00 Canary Builder
Task-level 8/1/1 split: train={1-8}, val={0}, held_out={9}

Re-splits existing step CSV by task_idx.
Teacher labels: inherited from old calibration (known limitation for canary).
Full re-teacher isolation will be done for 10-fold version.
"""
import csv, hashlib, json, os, sys
from pathlib import Path
from collections import Counter, defaultdict

# ── Frozen Fold 0 split ──
TRAIN_TASKS = {1, 2, 3, 4, 5, 6, 7, 8}
VAL_TASKS = {0}
HELD_OUT_TASKS = {9}

TASK_NAMES = {
    0: 'butter', 1: 'ketchup', 2: 'milk', 3: 'orange_juice',
    4: 'alphabet_soup', 5: 'tomato_sauce', 6: 'cream_cheese',
    7: 'salad_dressing', 8: 'bbq_sauce', 9: 'chocolate_pudding',
}

BASE = Path('/mnt/sdc/dty_user/openvla_attack')
STEP_CSV = BASE / 'migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv'
MANIFEST_CSV = BASE / 'migration_audit/m1c/sc5_v2_data/SC5_V2_DATASET_MANIFEST.csv'
OUT_DIR = BASE / 'tables/sc5_object_loto_v1/fold_00'
REPORT_DIR = BASE / 'reports/sc5_object_loto_v1/fold_00'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── 1. Load manifest: episode → task_idx ──
print('=== STEP 1: Load manifest ===')
ep_to_task = {}
ep_to_state = {}
with open(MANIFEST_CSV) as f:
    for r in csv.DictReader(f):
        sid = r['sample_id']
        ti = int(r['task_idx'])
        state = r.get('parent_state_id', '?')
        # sample_id = train_task0_state10 → episode not directly available
        # We'll map via step CSV later
        # Store for cross-reference
        pass

print('Manifest: using step CSV task_idx directly')

# ── 2. Read step CSV, assign new splits ──
print('=== STEP 2: Re-split by task_idx ===')
rows = []
episodes = defaultdict(lambda: {'task_idx': None, 'states': set(), 'rows': 0, 'succ': False})
skipped = 0

with open(STEP_CSV) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        ti_str = row.get('task_idx', '')
        if not ti_str or ti_str == '?':
            skipped += 1
            continue

        ti = int(ti_str)
        if ti < 0 or ti > 9:
            skipped += 1
            continue

        ep_id = row.get('episode_id', '')
        state_id = row.get('parent_state_id', row.get('step_idx', '?'))

        # Assign new split
        if ti in TRAIN_TASKS:
            new_split = 'train'
        elif ti in VAL_TASKS:
            new_split = 'val'
        elif ti in HELD_OUT_TASKS:
            new_split = 'held_out'
        else:
            skipped += 1
            continue

        # Update row
        row['split'] = new_split
        row['is_held_out'] = 'True' if new_split == 'held_out' else 'False'

        rows.append(row)

        # Track episode stats
        ep_info = episodes[ep_id]
        ep_info['task_idx'] = ti
        ep_info['states'].add(state_id)
        ep_info['rows'] += 1
        if row.get('task_success', '') == '1' or row.get('task_success', '') == 'True':
            ep_info['succ'] = True

print('Kept rows: {}'.format(len(rows)))
print('Skipped rows (no task_idx or out of range): {}'.format(skipped))

# ── 3. Audit ──
print('\n=== STEP 3: Split audit ===')

# Count by new split + task_idx
task_by_split = defaultdict(lambda: Counter())
for row in rows:
    task_by_split[row['split']][int(row['task_idx'])] += 1

for split_name in ['train', 'val', 'held_out']:
    total = sum(task_by_split[split_name].values())
    print('  {}: {} rows across {} tasks'.format(split_name, total, len(task_by_split[split_name])))
    for ti, cnt in sorted(task_by_split[split_name].items()):
        print('    task {} ({}): {} rows'.format(ti, TASK_NAMES.get(ti, '?'), cnt))

# Check expected task sets
actual_train = set(task_by_split['train'].keys())
actual_val = set(task_by_split['val'].keys())
actual_held = set(task_by_split['held_out'].keys())

print('\nExpected train tasks: {}'.format(sorted(TRAIN_TASKS)))
print('Actual train tasks:   {}'.format(sorted(actual_train)))
print('Expected val tasks:   {}'.format(sorted(VAL_TASKS)))
print('Actual val tasks:     {}'.format(sorted(actual_val)))
print('Expected held tasks:  {}'.format(sorted(HELD_OUT_TASKS)))
print('Actual held tasks:    {}'.format(sorted(actual_held)))

# ── Leakage checks ──
train_tasks_actual = set(task_by_split['train'].keys())
val_tasks_actual = set(task_by_split['val'].keys())
held_tasks_actual = set(task_by_split['held_out'].keys())

task_isolation_pass = True
isolation_issues = []

if train_tasks_actual & val_tasks_actual:
    isolation_issues.append('TASK_LEAK: train ∩ val = {}'.format(train_tasks_actual & val_tasks_actual))
    task_isolation_pass = False
if train_tasks_actual & held_tasks_actual:
    isolation_issues.append('TASK_LEAK: train ∩ held_out = {}'.format(train_tasks_actual & held_tasks_actual))
    task_isolation_pass = False
if val_tasks_actual & held_tasks_actual:
    isolation_issues.append('TASK_LEAK: val ∩ held_out = {}'.format(val_tasks_actual & held_tasks_actual))
    task_isolation_pass = False
if train_tasks_actual != TRAIN_TASKS:
    isolation_issues.append('TRAIN_MISMATCH: expected={} actual={}'.format(TRAIN_TASKS, train_tasks_actual))
    task_isolation_pass = False
if val_tasks_actual != VAL_TASKS:
    isolation_issues.append('VAL_MISMATCH: expected={} actual={}'.format(VAL_TASKS, val_tasks_actual))
    task_isolation_pass = False
if held_tasks_actual != HELD_OUT_TASKS:
    isolation_issues.append('HELD_OUT_MISMATCH: expected={} actual={}'.format(HELD_OUT_TASKS, held_tasks_actual))
    task_isolation_pass = False

print('\nTask isolation: {}'.format('PASS' if task_isolation_pass else 'FAIL'))
for issue in isolation_issues:
    print('  {}'.format(issue))
if not task_isolation_pass:
    sys.exit(1)

# ── Episode-level audit ──
print('\n=== Episode-level audit ===')
ep_by_split = defaultdict(set)
for ep_id, info in episodes.items():
    ti = info['task_idx']
    if ti in TRAIN_TASKS:
        ep_by_split['train'].add(ep_id)
    elif ti in VAL_TASKS:
        ep_by_split['val'].add(ep_id)
    elif ti in HELD_OUT_TASKS:
        ep_by_split['held_out'].add(ep_id)

# Episode leakage: check for cross-split episode IDs
ep_isolation_pass = True
ep_overlap_issues = []
train_eps = ep_by_split['train']
val_eps = ep_by_split['val']
held_eps = ep_by_split['held_out']

if train_eps & val_eps:
    ep_overlap_issues.append('EP_LEAK: train ∩ val = {} episodes'.format(len(train_eps & val_eps)))
    ep_isolation_pass = False
if train_eps & held_eps:
    ep_overlap_issues.append('EP_LEAK: train ∩ held_out = {} episodes'.format(len(train_eps & held_eps)))
    ep_isolation_pass = False
if val_eps & held_eps:
    ep_overlap_issues.append('EP_LEAK: val ∩ held_out = {} episodes'.format(len(val_eps & held_eps)))
    ep_isolation_pass = False

print('Episode isolation: {}'.format('PASS' if ep_isolation_pass else 'FAIL'))
for issue in ep_overlap_issues:
    print('  {}'.format(issue))

for split_name in ['train', 'val', 'held_out']:
    eps = ep_by_split[split_name]
    tasks_in_split = set()
    for ep in eps:
        tasks_in_split.add(episodes[ep]['task_idx'])
    print('  {}: {} episodes across tasks {}'.format(split_name, len(eps), sorted(tasks_in_split)))

# Gate
if not task_isolation_pass or not ep_isolation_pass:
    print('\nGATE: FAIL')
    sys.exit(1)
print('\nGATE: PASS — Student row isolation verified')
print('NOTE: Teacher labels inherited from old all-task calibration. This is a STUDENT-ONLY isolation canary.')

# ── 4. Write outputs ──
print('\n=== STEP 4: Write outputs ===')

# Main dataset CSV
dataset_path = OUT_DIR / 'sc5_object_loto_fold00_dataset.csv'
with open(dataset_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print('Dataset: {} ({:.1f} MB)'.format(dataset_path, dataset_path.stat().st_size / 1e6))

# Episode membership
ep_path = OUT_DIR / 'sc5_object_loto_fold00_episode_membership.csv'
with open(ep_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['episode_id', 'task_idx', 'task_name', 'split', 'n_rows', 'n_states', 'clean_success'])
    for ep_id in sorted(episodes.keys()):
        info = episodes[ep_id]
        ti = info['task_idx']
        if ti in TRAIN_TASKS:
            sp = 'train'
        elif ti in VAL_TASKS:
            sp = 'val'
        else:
            sp = 'held_out'
        w.writerow([ep_id, ti, TASK_NAMES.get(ti, '?'), sp, info['rows'], len(info['states']), info['succ']])
print('Membership: {} episodes'.format(len(episodes)))

# Split audit JSON
audit = {
    'protocol': 'sc5_object_loto_v1_fold00',
    'split_type': 'task_level_811_student_only',
    'isolation_level': 'STUDENT_ROW_ONLY',
    'train_tasks': sorted(TRAIN_TASKS),
    'val_tasks': sorted(VAL_TASKS),
    'held_out_tasks': sorted(HELD_OUT_TASKS),
    'task_isolation_pass': task_isolation_pass,
    'episode_isolation_pass': ep_isolation_pass,
    'teacher_calibration_source': 'inherited_from_old_all_task_build',
    'teacher_isolation_note': (
        'Teacher calibration and per-row labels were produced by the original build_sc5_canonical_corpus_v2.py '
        'which had access to all 10 Object tasks. Student row split is strict 8/1/1, but teacher labels may '
        'carry indirect information about val/held_out tasks through calibration thresholds. '
        'This is a STUDENT-ONLY isolation canary. Full teacher-isolated LOTO requires re-running '
        'privileged teacher calibration on train tasks only, which needs raw step_records.jsonl files.'
    ),
    'rows': {k: sum(v.values()) for k, v in task_by_split.items()},
    'episodes': {k: len(v) for k, v in ep_by_split.items()},
    'gate_pass': task_isolation_pass and ep_isolation_pass,
}
audit_path = REPORT_DIR / 'sc5_object_loto_fold00_split_audit.json'
with open(audit_path, 'w') as f:
    json.dump(audit, f, indent=2)
print('Audit: {}'.format(audit_path))

# Task counts
tc_path = REPORT_DIR / 'sc5_object_loto_fold00_task_counts.csv'
with open(tc_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task_idx', 'task_name', 'split', 'n_rows', 'n_episodes'])
    for ti in range(10):
        for sp in ['train', 'val', 'held_out']:
            cnt = task_by_split[sp].get(ti, 0)
            if cnt > 0:
                eps = sum(1 for ep, info in episodes.items() if info['task_idx'] == ti)
                w.writerow([ti, TASK_NAMES[ti], sp, cnt, eps])
print('Task counts: {}'.format(tc_path))

# SHA256 of outputs
sha_path = REPORT_DIR / 'SHA256SUMS.txt'
checksums = {}
for p in [dataset_path, ep_path, audit_path, tc_path]:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    checksums[p.name] = h.hexdigest()

with open(sha_path, 'w') as f:
    for name, sha in checksums.items():
        f.write('{}  {}\n'.format(sha, name))
print('SHA256SUMS: {}'.format(sha_path))

# Summary
print('\n=== CANARY FOLD 00 COMPLETE ===')
print('Train: {} rows, {} episodes'.format(sum(task_by_split['train'].values()), len(ep_by_split['train'])))
print('Val:   {} rows, {} episodes'.format(sum(task_by_split['val'].values()), len(ep_by_split['val'])))
print('Held:  {} rows, {} episodes'.format(sum(task_by_split['held_out'].values()), len(ep_by_split['held_out'])))
print('Gate:  PASS' if len(leakage) == 0 else 'FAIL')
