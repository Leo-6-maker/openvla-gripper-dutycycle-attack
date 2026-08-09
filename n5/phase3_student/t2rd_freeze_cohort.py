"""T2R-C0: Freeze confirmation cohort for T2R-D final evaluation.

Requirements:
  - Only train identities (640)
  - NOT overlapping with original T2-64 (seed=20103)
  - NOT val/cal identities
  - NOT G10 identities
  - Cover In, On, Stack relations
  - Cover microwave, stove, desk-caddy tasks
  - Include failures, pregrasp negatives, unsupported/articulated unknowns
  - At least 60 eligible successful placements, 40 negative/control
"""
import json, os, sys, time, hashlib
from collections import defaultdict
import numpy as np

DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(DIR), 'phase2_labels'))
from v22_production_v2 import get_object_slices_for_task

CS200 = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
G6_SEAL = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
G10_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G10_TEST_MANIFEST.json'
T2RD_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/t2rd_confirmation_cohort'
os.makedirs(T2RD_OUT, exist_ok=True)

SEED_COHORT = 20260726  # today's date as seed, deterministic

# Load existing sets
with open(G6_SEAL) as f:
    seal = json.load(f)
train_ids = set(seal['split']['train_identities'])
val_ids = set(seal['split']['val_identities'])
cal_ids = set(seal['split']['cal_identities'])

with open(G10_MANIFEST) as f:
    g10 = json.load(f)
g10_ids = set(g10['identities'])

# Rebuild original T2-64 (seed=20103) to exclude them
rng_old = np.random.RandomState(20103)
old_t2 = set()
for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
    suite_eps = sorted([i for i in (train_ids | val_ids | cal_ids) if i.startswith(suite)])
    successful = []; other = []
    for ident in suite_eps:
        s, t, st = ident.split('/')
        sp = os.path.join(CS200, s, t, st, 'episode_summary.json')
        if os.path.isfile(sp):
            with open(sp) as f:
                if json.load(f).get('success', False):
                    successful.append(ident)
                else:
                    other.append(ident)
    rng_old.shuffle(successful); rng_old.shuffle(other)
    old_t2.update(successful[:10] + other[:6])

print(f'Train: {len(train_ids)}, Val: {len(val_ids)}, Cal: {len(cal_ids)}')
print(f'G10: {len(g10_ids)}, Old T2: {len(old_t2)}')

# Candidate pool: train only, excluding old-T2, val, cal, G10
excluded = old_t2 | val_ids | cal_ids | g10_ids
candidate_pool = sorted(train_ids - excluded)
print(f'Candidate pool: {len(candidate_pool)}')

# Classify candidates by relation type and task
from collections import defaultdict

def task_key(suite, task):
    return f'{suite}/{task}'

by_suite = defaultdict(list)
by_relation = defaultdict(list)
fixture_tasks = set()
task_07_eps = []
successful_pool = []
failed_pool = []
unknown_pool = []

for ident in candidate_pool:
    suite, task, state = ident.split('/')
    by_suite[suite].append(ident)

    # Check success
    sp = os.path.join(CS200, suite, task, state, 'episode_summary.json')
    is_success = False
    if os.path.isfile(sp):
        with open(sp) as f:
            is_success = json.load(f).get('success', False)

    if is_success:
        successful_pool.append(ident)
    else:
        failed_pool.append(ident)

    # Check BDDL relations
    task_idx = int(task.replace('task_', ''))
    bddl = get_object_slices_for_task(suite, task_idx)
    if bddl:
        g_rels = bddl['task_role'].get('goal_relations', [])
        gs_names = bddl['task_role'].get('goal_support_names', [])
        for pred, obj, tgt in g_rels:
            by_relation[pred].append(ident)
        # Detect fixture tasks (target not in object_slices)
        slices = bddl['object_slices']
        for pred, obj, tgt in g_rels:
            tgt_base = tgt
            for suffix in ['_contain_region', '_init_region', '_cook_region',
                           '_heating_region', '_top_region', '_front_region',
                           '_back_contain_region']:
                tgt_base = tgt_base.replace(suffix, '')
            if tgt_base not in slices:
                fixture_tasks.add(task_key(suite, task))
        # task_07
        if task == 'task_07':
            task_07_eps.append(ident)

    # Check for articulated/unsupported: use BDDL goal predicates
    if bddl:
        g_rels = bddl['task_role'].get('goal_relations', [])
        if not g_rels and task == 'task_07':
            unknown_pool.append(ident)


print(f'By suite: { {s: len(v) for s, v in by_suite.items()} }')
print(f'By relation: { {r: len(v) for r, v in by_relation.items()} }')
print(f'Successful pool: {len(successful_pool)}')
print(f'Failed pool: {len(failed_pool)}')

# Build confirmation cohort
rng = np.random.RandomState(SEED_COHORT)
cohort = []
cohort_meta = []

# Target: 60 successful placements + 40 negative/control

# 1. Successful placement candidates per suite per relation
target_successful = 80
per_suite_success = target_successful // 4  # 20 per suite

for suite in ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']:
    suite_success = [i for i in successful_pool if i.startswith(suite)]
    rng.shuffle(suite_success)
    # Stratify by task to ensure coverage
    by_task = defaultdict(list)
    for ident in suite_success:
        by_task[ident.split('/')[1]].append(ident)
    n_per_task = max(2, per_suite_success // max(1, len(by_task)) + 1)
    picked = 0
    for task_name in sorted(by_task.keys()):
        eps = by_task[task_name]
        rng.shuffle(eps)
        take = min(n_per_task, len(eps))
        for ident in eps[:take]:
            if picked >= per_suite_success:
                break
            if ident in cohort: continue
            cohort.append(ident)
            cohort_meta.append({'identity': ident, 'type': 'successful_placement'})
            picked += 1
        if picked >= per_suite_success:
            break
    # If still short, fill from any task
    if picked < per_suite_success:
        remaining = [i for i in suite_success if i not in cohort]
        rng.shuffle(remaining)
        for ident in remaining:
            if picked >= per_suite_success: break
            cohort.append(ident)
            cohort_meta.append({'identity': ident, 'type': 'successful_placement'})
            picked += 1

# 2. Negative/control: failures, pregrasp, articulated unknown
target_negative = 50
# Failures
rng.shuffle(failed_pool)
n_fail = min(30, len(failed_pool))
for ident in failed_pool[:n_fail]:
    cohort.append(ident)
    cohort_meta.append({'identity': ident, 'type': 'failure_negative'})

# Articulated unknown
rng.shuffle(unknown_pool)
n_unk = min(10, len(unknown_pool))
for ident in unknown_pool[:n_unk]:
    if ident not in cohort:
        cohort.append(ident)
        cohort_meta.append({'identity': ident, 'type': 'articulated_unknown'})

# Pregrasp: just use failed episodes from large-object tasks
pregrasp_eps = [i for i in failed_pool if 'libero_10' in i or 'libero_spatial' in i]
rng.shuffle(pregrasp_eps)
for ident in pregrasp_eps[:10]:
    if ident not in cohort and len(cohort) < 100:
        cohort.append(ident)
        cohort_meta.append({'identity': ident, 'type': 'short_pregrasp'})

# Fill remaining with failed episodes
for ident in failed_pool:
    if ident not in cohort:
        cohort.append(ident)
        cohort_meta.append({'identity': ident, 'type': 'extra_failure'})
        if len([m for m in cohort_meta if 'failure' in m['type'] or 'negative' in m['type']]) >= 40:
            break

print(f'\nCohort size: {len(cohort)}')
type_counts = defaultdict(int)
for m in cohort_meta:
    type_counts[m['type']] += 1
print(f'  {dict(type_counts)}')

# Check overlap with excluded sets
cohort_set = set(cohort)
assert len(cohort_set & old_t2) == 0, 'Overlap with old T2!'
assert len(cohort_set & val_ids) == 0, 'Overlap with val!'
assert len(cohort_set & cal_ids) == 0, 'Overlap with cal!'
assert len(cohort_set & g10_ids) == 0, 'Overlap with G10!'

# Per-suite breakdown
suite_breakdown = defaultdict(int)
for ident in cohort:
    suite_breakdown[ident.split('/')[0]] += 1
print(f'Per suite: {dict(suite_breakdown)}')

# Count successful placements in cohort
n_success_cohort = sum(1 for ident in cohort if ident in successful_pool)
print(f'Successful placements: {n_success_cohort}')

# Write manifest
manifest = {
    'manifest': 'T2RD_CONFIRM_MANIFEST_V1',
    'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'seed': SEED_COHORT,
    'n_episodes': len(cohort),
    'n_successful_placements': n_success_cohort,
    'n_negative_control': len(cohort) - n_success_cohort,
    'per_suite': dict(suite_breakdown),
    'per_type': dict(type_counts),
    'exclusions': {
        'old_t2_64': 'excluded (development set)',
        'val_80': 'excluded',
        'cal_80': 'excluded',
        'g10_1200': 'excluded',
    },
    'identities': sorted(cohort),
    'identity_metadata': sorted(cohort_meta, key=lambda m: m['identity']),
    'rules': [
        'NO identity from val/cal/G10',
        'NO identity from original T2-64 (development set)',
        'ONE-TIME unblinding for T2R-D final evaluation only',
        'Recall < 90% OR precision < 90% => exit 5, HOLD',
    ],
}

manifest_path = os.path.join(T2RD_OUT, 'T2RD_CONFIRM_MANIFEST_V1.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, default=str)
manifest_sha = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
manifest['self_sha256'] = manifest_sha
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2, default=str)

print(f'\nManifest: {manifest_path}')
print(f'SHA: {manifest_sha[:16]}...')
print('T2R-C0: DONE')
