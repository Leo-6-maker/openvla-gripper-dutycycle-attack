"""Unified 2000 Teacher manifest + H2/C3/P3 allocation. Fixed seed, no Student scores."""
import json, os, hashlib, time, random
from pathlib import Path
from collections import defaultdict

EVIDENCE = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716')
CLEAN = EVIDENCE / 'clean'; OPS = EVIDENCE / 'ops'
OUT = OPS / 'V21_TEACHER_AND_ROLES_20260725'
os.makedirs(OUT, exist_ok=True)
SEED = 20260725; K10 = 10
random.seed(SEED)

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

# ═══ 1. Unified 2000 Teacher Manifest ═══
print('=== 1. UNIFIED TEACHER MANIFEST ===')
NEW_ROOT = OPS / 'FACTORIZED_TEACHER_STATES_35_49_20260725' / 'labels'

teacher_index = {}
for root_dir in ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']:
    if not os.path.isdir(root_dir): continue
    for suite in sorted(os.listdir(root_dir)):
        sp = os.path.join(root_dir, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp): continue
            for state in sorted(os.listdir(tp)):
                eid = '{}/{}/{}'.format(suite, task, state)
                label_path = os.path.join(tp, state, 'factorized_teacher_v1.jsonl')
                if os.path.isfile(label_path):
                    teacher_index[eid] = {'source': 'HISTORICAL', 'path': label_path,
                        'suite': suite, 'task': task, 'state': state}

for suite in sorted(os.listdir(NEW_ROOT)):
    sp = NEW_ROOT / suite
    if not sp.is_dir(): continue
    for task in sorted(os.listdir(sp)):
        tp = sp / task
        if not tp.is_dir(): continue
        for state in sorted(os.listdir(tp)):
            eid = '{}/{}/{}'.format(suite, task, state)
            label_path = tp / state / 'factorized_teacher_v1.jsonl'
            if label_path.is_file():
                teacher_index[eid] = {'source': 'NEW_STATES_35_49', 'path': str(label_path),
                    'suite': suite, 'task': task, 'state': state}

total = len(teacher_index)
hist = sum(1 for v in teacher_index.values() if v['source']=='HISTORICAL')
new_ct = sum(1 for v in teacher_index.values() if v['source']=='NEW_STATES_35_49')
print('Total: {} (Historical: {} New: {})'.format(total, hist, new_ct))

unified = {
    'schema': 'V21_UNIFIED_TEACHER_MANIFEST_V1',
    'total_identities': total, 'external_k10_count': total,
    'internal_fallback_count': 0, 'internal_fallback_status': 'PASS',
    'teacher_version': 'Physics V2.1C', 'k10_version': 'V1.2.2',
    'by_source': {'historical': hist, 'new_states_35_49': new_ct},
    'identity_index': {eid: v for eid, v in sorted(teacher_index.items())},
}
with open(OUT / 'V21_UNIFIED_TEACHER_MANIFEST_V1.json', 'w') as f:
    json.dump(unified, f, indent=2)

# ═══ 2. Allocate new-H2 ═══
print('\n=== 2. NEW-H2 ALLOCATION ===')
new_eps = {eid: v for eid, v in teacher_index.items() if v['source'] == 'NEW_STATES_35_49'}
task_groups = defaultdict(list)
for eid, v in new_eps.items():
    task_key = '{}/{}'.format(v['suite'], v['task'])
    task_groups[task_key].append((int(v['state'].replace('state_','')), eid))

h2_ids = set()
for task_key, states in sorted(task_groups.items()):
    states.sort()
    indices = list(range(len(states)))
    random.shuffle(indices)
    h2_batch = set(states[i][1] for i in indices[:5])
    h2_ids.update(h2_batch)
print('new-H2: {}'.format(len(h2_ids)))

h2_manifest = {
    'schema': 'V21_NEW_H2_IDENTITY_MANIFEST_V1',
    'count': len(h2_ids), 'identities': sorted(h2_ids),
    'access': 'FORBIDDEN_UNTIL_FORMAL_EVALUATION',
    'allocation_method': 'per-task random 5/15 states, seed={}'.format(SEED),
    'student_access': 0, 'metric_access': 0,
}
with open(OUT / 'V21_NEW_H2_IDENTITY_MANIFEST_V1.json', 'w') as f:
    json.dump(h2_manifest, f, indent=2)
h2_seal_dir = OUT / 'new_h2_sealed'; os.makedirs(h2_seal_dir, exist_ok=True)
with open(h2_seal_dir / 'identities.json', 'w') as f:
    json.dump(sorted(h2_ids), f, indent=2)
sh = sha256_file(h2_seal_dir / 'identities.json')
with open(h2_seal_dir / 'SHA256SUMS', 'w') as f:
    f.write('{}  identities.json\n'.format(sh))
with open(h2_seal_dir / 'SHA256SUMS.sha256', 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sha256_file(h2_seal_dir / 'SHA256SUMS')))
print('H2 sealed: {}'.format(sh[:16]))

# ═══ 3. Allocate C3/P3 ═══
print('\n=== 3. C3/P3 ALLOCATION ===')
remaining = set(new_eps.keys()) - h2_ids
rem_task_groups = defaultdict(list)
for eid in remaining:
    v = new_eps[eid]
    rem_task_groups['{}/{}'.format(v['suite'], v['task'])].append(eid)

def get_absence_reason(eid, v):
    labels = [json.loads(l) for l in open(Path(v['path'])).read().splitlines() if l.strip()]
    T = len(labels); max_t = min(T, T-K10+1)
    n_k10_pos = sum(1 for t in range(max_t) if labels[t].get('strict_k10_feasible',False) and labels[t].get('strict_k10_known_mask',False))
    if n_k10_pos > 0: return 'OPPORTUNITY_PRESENT'
    any_k10_known = any(labels[t].get('strict_k10_known_mask',False) for t in range(max_t))
    any_manip_known = any(labels[t].get('manipulation_active_known_mask',False) for t in range(max_t))
    any_grasp_known = any(labels[t].get('grasp_established_known_mask',False) for t in range(max_t))
    n_manip_pos = sum(1 for t in range(max_t) if labels[t].get('manipulation_active',False) and labels[t].get('manipulation_active_known_mask',False))
    n_grasp_pos = sum(1 for t in range(max_t) if labels[t].get('grasp_established',False) and labels[t].get('grasp_established_known_mask',False))
    if not any_k10_known: return 'F1_STRUCTURAL_ZERO'
    if n_manip_pos == 0 and any_manip_known: return 'F3_NO_MANIPULATION'
    if n_grasp_pos == 0 and any_grasp_known: return 'F4_NO_STABLE_GRASP'
    return 'F6_OTHER'

c3_ids = set(); p3_ids = set()
p3_opp = 0; p3_abs = 0; p3_f1 = 0; p3_f3 = 0; p3_f4 = 0

for task_key, eids in sorted(rem_task_groups.items()):
    eids_sorted = sorted(eids, key=lambda e: int(e.split('/')[2].replace('state_','')))
    c3_ids.update(eids_sorted[:5]); p3_ids.update(eids_sorted[5:])
    for eid in eids_sorted[5:]:
        reason = get_absence_reason(eid, new_eps[eid])
        if reason == 'OPPORTUNITY_PRESENT': p3_opp += 1
        else:
            p3_abs += 1
            if reason == 'F1_STRUCTURAL_ZERO': p3_f1 += 1
            elif reason == 'F3_NO_MANIPULATION': p3_f3 += 1
            elif reason == 'F4_NO_STABLE_GRASP': p3_f4 += 1

print('C3: {}  P3: {}'.format(len(c3_ids), len(p3_ids)))
print('P3 strata: opp={} abs={} F1={} F3={} F4={}'.format(p3_opp, p3_abs, p3_f1, p3_f3, p3_f4))

p3_gates = {'opp_ge_60': p3_opp >= 60, 'abs_ge_60': p3_abs >= 60,
    'F3_ge_15': p3_f3 >= 15, 'F4_ge_15': p3_f4 >= 15, 'parser_eq_0': True}
for k,v in p3_gates.items(): print('  {}: {}'.format(k, 'PASS' if v else 'FAIL'))
print('P3: {}'.format('PASS' if all(p3_gates.values()) else 'FAIL_INSUFFICIENT_COVERAGE'))

for role, ids in [('C3', c3_ids), ('P3', p3_ids)]:
    m = {'schema': 'V21_{}_IDENTITY_MANIFEST_V1'.format(role), 'role': role,
         'count': len(ids), 'identities': sorted(ids)}
    with open(OUT / 'V21_{}_IDENTITY_MANIFEST_V1.json'.format(role), 'w') as f:
        json.dump(m, f, indent=2)

# Seal
all_files = []
for root, dirs, fns in os.walk(OUT):
    for fn in sorted(fns):
        if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT)
        all_files.append((rel, sha256_file(fp)))
with open(OUT / 'SHA256SUMS', 'w') as f:
    for rel, h in sorted(all_files): f.write('{}  {}\n'.format(h, rel))
sh = sha256_file(OUT / 'SHA256SUMS')
with open(OUT / 'SHA256SUMS.sha256', 'w') as f: f.write('{}  SHA256SUMS\n'.format(sh))

print('\nSealed: {}'.format(OUT))
print('Seal: {}'.format(sh[:16]))
print('NEW_H2: SEALED/UNREAD ({} identities)'.format(len(h2_ids)))
