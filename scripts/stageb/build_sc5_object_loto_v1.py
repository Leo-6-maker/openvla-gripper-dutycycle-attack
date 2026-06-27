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

# Leakage checks
train_set = set(task_by_split['train'].keys())
val_set = set(task_by_split['val'].keys())
held_set = set(task_by_split['held_out'].keys())

leakage = []
if train_set & val_set:
    leakage.append('TRAIN ∩ VAL: {}'.format(train_set & val_set))
if train_set & held_set:
    leakage.append('TRAIN ∩ HELD_OUT: {}'.format(train_set & held_set))
if val_set & held_set:
    leakage.append('VAL ∩ HELD_OUT: {}'.format(val_set & held_set))
if actual_train != TRAIN_TASKS:
    leakage.append('TRAIN MISMATCH: expected={} actual={}'.format(TRAIN_TASKS, actual_train))
if actual_val != VAL_TASKS:
    leakage.append('VAL MISMATCH: expected={} actual={}'.format(VAL_TASKS, actual_val))
if actual_held != HELD_OUT_TASKS:
    leakage.append('HELD_OUT MISMATCH: expected={} actual={}'.format(HELD_OUT_TASKS, actual_held))

if leakage:
    print('\n*** LEAKAGE DETECTED ***')
    for l in leakage:
        print('  {}'.format(l))
    print('GATE: FAIL')
    sys.exit(1)
else:
    print('\nGATE: PASS — No task leakage')

# Episode-level audit
print('\n=== Episode audit ===')
ep_by_split = defaultdict(set)
for ep_id, info in episodes.items():
    ti = info['task_idx']
    if ti in TRAIN_TASKS:
        ep_by_split['train'].add(ep_id)
    elif ti in VAL_TASKS:
        ep_by_split['val'].add(ep_id)
    elif ti in HELD_OUT_TASKS:
        ep_by_split['held_out'].add(ep_id)

for split_name in ['train', 'val', 'held_out']:
    eps = ep_by_split[split_name]
    tasks_in_split = set()
    for ep in eps:
        tasks_in_split.add(episodes[ep]['task_idx'])
    print('  {}: {} episodes across tasks {}'.format(split_name, len(eps), sorted(tasks_in_split)))

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
    'split_type': 'task_level_811',
    'train_tasks': sorted(TRAIN_TASKS),
    'val_tasks': sorted(VAL_TASKS),
    'held_out_tasks': sorted(HELD_OUT_TASKS),
    'task_leakage': len(leakage) == 0,
    'episode_leakage': len(leakage) == 0,
    'rows': {k: sum(v.values()) for k, v in task_by_split.items()},
    'episodes': {k: len(v) for k, v in ep_by_split.items()},
    'teacher_calibration_source': 'inherited_from_old_build',
    'teacher_isolation_note': 'CANARY ONLY — old teacher saw all tasks. Full re-teacher isolation will be done for 10-fold.',
    'gate_pass': len(leakage) == 0,
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
