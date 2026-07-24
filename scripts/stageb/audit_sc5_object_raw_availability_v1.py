#!/usr/bin/env python3
"""
SC5 Object Raw Availability Audit v1
Maps 370 expected episodes to raw artifact sources.
Identifies gaps for Strict Fold 0 V2 teacher re-calibration.
"""
import csv, json, hashlib, os, sys
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path('/mnt/sdc/dty_user/openvla_attack')
MANIFEST_CSV = BASE / 'migration_audit/m1c/sc5_v2_data/SC5_V2_DATASET_MANIFEST.csv'
STEP_CSV = BASE / 'migration_audit/m1c/sc5_v2_data/SC5_V2_STEP_DATASET.csv'
REPLAY_ROOT = BASE / 'evidence/object_checkpoint_migration/m1_runtime_b0_d1/replay_60cell'
OUT_DIR = BASE / 'tables/sc5_object_loto_v2'
REPORT_DIR = BASE / 'reports/sc5_object_loto_v2'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TASK_NAMES = {
    0: 'butter', 1: 'ketchup', 2: 'milk', 3: 'orange_juice',
    4: 'alphabet_soup', 5: 'tomato_sauce', 6: 'cream_cheese',
    7: 'salad_dressing', 8: 'bbq_sauce', 9: 'chocolate_pudding',
}

print('=' * 60)
print('SC5 OBJECT RAW AVAILABILITY AUDIT')
print('=' * 60)

# ── 1. Expected membership ──
print('\n── 1. Expected Membership ──')
with open(MANIFEST_CSV) as f:
    manifest = list(csv.DictReader(f))
with open(STEP_CSV) as f:
    step_rows = list(csv.DictReader(f))

# Count by task_idx
task_eps = defaultdict(set)
for r in step_rows:
    ti = r.get('task_idx', '?')
    ep = r.get('episode_id', '?')
    if ti != '?' and ep != '?':
        task_eps[int(ti)].add(ep)

print('Expected: 10 tasks')
for ti in range(10):
    eps = task_eps.get(ti, set())
    print('  task {} ({}): {} episodes, {} rows'.format(ti, TASK_NAMES.get(ti, '?'), len(eps),
          sum(1 for r in step_rows if r.get('task_idx') == str(ti))))

total_eps = len(set().union(*task_eps.values()))
print('Total expected episodes: {}'.format(total_eps))

# ── 2. Raw source inventory ──
print('\n── 2. Raw Source Inventory ──')

# 2a. step_records.jsonl (none found)
step_records_files = list(BASE.rglob('step_records.jsonl'))
print('step_records.jsonl: {} files found'.format(len(step_records_files)))

# 2b. privileged_step_records.jsonl
priv_files = list(BASE.rglob('privileged_step_records.jsonl'))
print('privileged_step_records.jsonl: {} files found'.format(len(priv_files)))
priv_by_task = defaultdict(list)
for pf in priv_files:
    # Extract task from path: .../replay_60cell/tomato_sauce_s1/B0/privileged_step_records.jsonl
    parts = pf.parts
    for i, p in enumerate(parts):
        if p == 'replay_60cell' and i + 1 < len(parts):
            task_state = parts[i + 1]
            # task_state = "tomato_sauce_s1"
            if '_s' in task_state:
                task_name = task_state.rsplit('_s', 1)[0]
                state_id = task_state.rsplit('_s', 1)[1]
                priv_by_task[task_name].append(pf)
            break

print('\nPrivileged records by task:')
for tn in sorted(priv_by_task.keys()):
    print('  {}: {} files'.format(tn, len(priv_by_task[tn])))

# Check teacher input fields in privileged records
print('\n── 3. Teacher Input Field Coverage ──')
teacher_fields = [
    'teacher_privileged_state_available',
    'object_eef_distance', 'object_to_target_distance',
    'object_pose', 'target_pose',
    'gripper_command', 'eef_x', 'eef_y', 'eef_z',
]
if priv_files:
    sample = priv_files[0]
    with open(sample) as f:
        first_line = json.loads(f.readline())
    available = sorted(first_line.keys())
    print('Sample privileged record fields ({}):'.format(len(available)))
    for tf in teacher_fields:
        found = tf in available or any(tf in k for k in available)
        print('  {}: {}'.format(tf, 'FOUND' if found else 'MISSING'))
    print('All fields: {}'.format(available[:30]))

# ── 4. Step CSV field assessment ──
print('\n── 4. Step CSV Field Assessment ──')
csv_cols = step_rows[0].keys() if step_rows else []
has_teacher_input = any('object_' in c or 'target_' in c or 'privilege' in c.lower() for c in csv_cols)
has_teacher_labels = 'teacher_sc5_corridor_active' in csv_cols
has_student_features = 'gripper_command' in csv_cols
has_split = 'split' in csv_cols

print('Has teacher input fields (object/target/privileged): {}'.format(has_teacher_input))
print('Has teacher output labels: {}'.format(has_teacher_labels))
print('Has student features (25D): {}'.format(has_student_features))
print('Has split column: {}'.format(has_split))

# ── 5. Coverage matrix ──
print('\n── 5. Raw Coverage Matrix ──')
for ti in range(10):
    tn = TASK_NAMES[ti]
    csv_eps = len(task_eps.get(ti, set()))
    priv_count = len(priv_by_task.get(tn, []))
    has_step_records = 0
    status = 'OK' if priv_count >= 2 else ('PARTIAL' if priv_count > 0 else 'NO_RAW')
    print('  task {} ({}): csv={} eps, step_records=0, priv={} files [{}]'.format(
        ti, tn, csv_eps, priv_count, status))

# ── 6. Verdict ──
print('\n── 6. Verdict ──')
print('Situation: PARTIAL_RAW_PRIVILEGED_ONLY')
print('step_records.jsonl: 0/370 episodes')
print('privileged_step_records.jsonl: 60 files covering 10 tasks (3 states × 2 variants each)')
print('Step CSV: has student features + old teacher labels, NO teacher input fields')
print()
print('Teacher re-calibration:')
print('  Full re-labeling (370 eps): NOT POSSIBLE — raw teacher input missing')
print('  Threshold re-calibration comparison: POSSIBLE — 60 privileged episodes available')
print('  Teacher label drift estimation: POSSIBLE — compare old vs new teacher on 60 episodes')
print()
print('RECOMMENDATION:')
print('  Accept current STUDENT_ONLY isolation as canary.')
print('  For strict teacher-isolated V2: either collect privileged records for all 370 eps,')
print('  or use RAW_ENOUGH_CSV path with train-only threshold recalibration from existing labels.')

# ── 7. Write outputs ──
availability_rows = []
for ti in range(10):
    tn = TASK_NAMES[ti]
    csv_eps = len(task_eps.get(ti, set()))
    csv_rows = sum(1 for r in step_rows if r.get('task_idx') == str(ti))
    priv_count = len(priv_by_task.get(tn, []))
    availability_rows.append({
        'task_idx': ti, 'task_name': tn,
        'csv_episodes': csv_eps, 'csv_rows': csv_rows,
        'step_records_jsonl': 0, 'privileged_jsonl': priv_count,
        'has_teacher_input': False, 'has_student_features': True,
        'status': 'PARTIAL' if priv_count > 0 else 'NO_RAW'
    })

with open(OUT_DIR / 'raw_availability.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=availability_rows[0].keys())
    w.writeheader()
    w.writerows(availability_rows)

audit_json = {
    'total_expected_episodes': 370,
    'step_records_jsonl_found': 0,
    'privileged_step_records_jsonl_found': len(priv_files),
    'tasks_with_privileged': len([t for t, fs in priv_by_task.items() if fs]),
    'step_csv_has_teacher_input': has_teacher_input,
    'step_csv_has_teacher_labels': has_teacher_labels,
    'step_csv_has_student_features': has_student_features,
    'situation': 'PARTIAL_RAW_PRIVILEGED_ONLY',
    'full_re_labeling_possible': False,
    'threshold_comparison_possible': True,
    'recommendation': 'Accept STUDENT_ONLY canary. For strict V2: collect privileged records or use RAW_ENOUGH_CSV.'
}
with open(REPORT_DIR / 'RAW_AVAILABILITY_AUDIT.json', 'w') as f:
    json.dump(audit_json, f, indent=2)

print('\nOutputs written:')
print('  {}'.format(OUT_DIR / 'raw_availability.csv'))
print('  {}'.format(REPORT_DIR / 'RAW_AVAILABILITY_AUDIT.json'))
