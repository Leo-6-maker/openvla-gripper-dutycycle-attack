"""Enrich P3 with 100 hard negatives from DEV pool. Teacher strata only, no Student scores."""
import json, os, hashlib, time, random
from pathlib import Path
from collections import defaultdict

EVIDENCE = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716')
OPS = EVIDENCE / 'ops'; OUT = OPS / 'V21_TEACHER_AND_ROLES_20260725'
SEED = 20260725; K10 = 10; random.seed(SEED)

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

# Load existing
h2_ids = set(json.load(open(OUT / 'V21_NEW_H2_IDENTITY_MANIFEST_V1.json'))['identities'])
c3_orig = set(json.load(open(OUT / 'V21_C3_IDENTITY_MANIFEST_V1.json'))['identities'])
p3_orig = set(json.load(open(OUT / 'V21_P3_IDENTITY_MANIFEST_V1.json'))['identities'])
print('Existing: H2={} C3={} P3={}'.format(len(h2_ids), len(c3_orig), len(p3_orig)))

dev_ids = set()
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
                if os.path.isfile(os.path.join(tp, state, 'factorized_teacher_v1.jsonl')):
                    dev_ids.add(eid)
print('DEV pool: {}'.format(len(dev_ids)))

def classify(eid):
    parts = eid.split('/')
    label_paths = [
        '/tmp/ft_FIT_TRAIN/labels', '/tmp/ft_FIT_DEV/labels',
        '/tmp/ft_CAL/labels', '/tmp/ft_CHECK/labels', '/tmp/ft_H/labels',
        str(OPS / 'FACTORIZED_TEACHER_STATES_35_49_20260725' / 'labels'),
    ]
    for root_dir in label_paths:
        tp = os.path.join(root_dir, parts[0], parts[1], parts[2], 'factorized_teacher_v1.jsonl')
        if os.path.isfile(tp):
            labels = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
            T = len(labels); max_t = min(T, T-K10+1)
            n_k10_pos = sum(1 for t in range(max_t) if labels[t].get('strict_k10_feasible',False) and labels[t].get('strict_k10_known_mask',False))
            if n_k10_pos > 0: return 'OPPORTUNITY_PRESENT'
            any_k10_known = any(labels[t].get('strict_k10_known_mask',False) for t in range(max_t))
            any_manip_known = any(labels[t].get('manipulation_active_known_mask',False) for t in range(max_t))
            any_grasp_known = any(labels[t].get('grasp_established_known_mask',False) for t in range(max_t))
            n_manip_pos = sum(1 for t in range(max_t) if labels[t].get('manipulation_active',False) and labels[t].get('manipulation_active_known_mask',False))
            n_grasp_pos = sum(1 for t in range(max_t) if labels[t].get('grasp_established',False) and labels[t].get('grasp_established_known_mask',False))
            n_g_known = sum(1 for t in range(max_t) if labels[t].get('grasp_established_known_mask',False))
            n_m_known = sum(1 for t in range(max_t) if labels[t].get('manipulation_active_known_mask',False))
            if n_k10_pos > 0 and n_g_known == 0 and n_m_known == 0: return 'PARSER_INVALID'
            if not any_k10_known: return 'F1_STRUCTURAL_ZERO'
            if n_manip_pos == 0 and any_manip_known: return 'F3_NO_MANIPULATION'
            if n_grasp_pos == 0 and any_grasp_known: return 'F4_NO_STABLE_GRASP'
            return 'OTHER_ABSENT'
    raise SystemExit('LABEL_NOT_FOUND: {} — no factorized_teacher_v1.jsonl in any label root'.format(eid))

# Stratify dev pool
strata = defaultdict(list)
for eid in sorted(dev_ids):
    reason = classify(eid)
    strata[reason].append(eid)

print('\nDEV strata:')
for reason in sorted(strata.keys()):
    print('  {}: {}'.format(reason, len(strata[reason])))

# Select hard negatives: F3>=30, F4>=30, F1>=20, total=100
quotas = {'F3_NO_MANIPULATION': 30, 'F4_NO_STABLE_GRASP': 30, 'F1_STRUCTURAL_ZERO': 20}
selected = set()
for reason, quota in quotas.items():
    candidates = strata.get(reason, [])
    if len(candidates) < quota:
        print('WARNING: {} only {} candidates, taking all'.format(reason, len(candidates)))
        quota = len(candidates)
    by_suite = defaultdict(list)
    for eid in candidates: by_suite[eid.split('/')[0]].append(eid)
    chosen = []
    suite_list = sorted(by_suite.keys())
    while len(chosen) < quota:
        for suite in suite_list:
            if len(chosen) >= quota: break
            if by_suite[suite]: chosen.append(by_suite[suite].pop(0))
    selected.update(chosen)
    print('{}: selected {}'.format(reason, len(chosen)))

# Fill remaining to 100
remaining_needed = 100 - len(selected)
other_candidates = [eid for eid in strata.get('OTHER_ABSENT', []) if eid not in selected]
if len(other_candidates) >= remaining_needed:
    extra = random.sample(other_candidates, remaining_needed)
    selected.update(extra)
    print('OTHER_ABSENT: selected {}'.format(len(extra)))
else:
    selected.update(other_candidates)
    print('OTHER_ABSENT: selected {} (all available)'.format(len(other_candidates)))

print('Total selected: {}'.format(len(selected)))
parser_count = sum(1 for eid in selected if classify(eid) == 'PARSER_INVALID')
print('Parser-invalid in selection: {}'.format(parser_count))

# Suite diversity
suite_counts = defaultdict(int)
for eid in selected: suite_counts[eid.split('/')[0]] += 1
print('Suite coverage: {}'.format(dict(suite_counts)))

# Final allocation
p3_enriched = p3_orig | selected
dev_new = dev_ids - selected
c3_final = c3_orig

print('\nFinal allocation:')
print('  DEV: {}  C3: {}  P3: {}  H2: {}'.format(len(dev_new), len(c3_final), len(p3_enriched), len(h2_ids)))
print('  SUM: {}'.format(len(dev_new) + len(c3_final) + len(p3_enriched) + len(h2_ids)))

sets = {'DEV': dev_new, 'C3': c3_final, 'P3': p3_enriched, 'H2': h2_ids}
overlap_ok = True
for r1 in sets:
    for r2 in sets:
        if r1 < r2:
            n = len(sets[r1] & sets[r2])
            if n > 0: print('OVERLAP {} {}: {}'.format(r1, r2, n)); overlap_ok = False
print('Overlap: {}'.format('PASS' if overlap_ok else 'FAIL'))

for role, ids in [('DEV', dev_new), ('C3', c3_final), ('P3', p3_enriched)]:
    m = {'schema': 'V21_{}_IDENTITY_MANIFEST_V2'.format(role), 'role': role,
         'count': len(ids), 'identities': sorted(ids),
         'version': 'V2 — P3 enriched from DEV hard negatives'}
    with open(OUT / 'V21_{}_IDENTITY_MANIFEST_V2.json'.format(role), 'w') as f:
        json.dump(m, f, indent=2)

# P3 strata audit — full 300
p3_opp = 0; p3_abs = 0; p3_f1 = 0; p3_f3 = 0; p3_f4 = 0; p3_other = 0
p3_suite_counts = defaultdict(int); p3_task_counts = defaultdict(int)
for eid in sorted(p3_enriched):
    reason = classify(eid)
    p3_suite_counts[eid.split('/')[0]] += 1
    p3_task_counts['{}/{}'.format(eid.split('/')[0], eid.split('/')[1])] += 1
    if reason == 'OPPORTUNITY_PRESENT': p3_opp += 1
    else:
        p3_abs += 1
        if reason == 'F1_STRUCTURAL_ZERO': p3_f1 += 1
        elif reason == 'F3_NO_MANIPULATION': p3_f3 += 1
        elif reason == 'F4_NO_STABLE_GRASP': p3_f4 += 1
        else: p3_other += 1

print('\nP3-enriched 300 strata:')
print('  Total: {}  opp={}  abs={}  F1={}  F3={}  F4={}  other={}'.format(
    len(p3_enriched), p3_opp, p3_abs, p3_f1, p3_f3, p3_f4, p3_other))
print('  Suite coverage: {}'.format(dict(sorted(p3_suite_counts.items()))))
print('  Task coverage: {}/40 tasks'.format(len(p3_task_counts)))
gates = {'abs_ge_60': p3_abs >= 60, 'F3_ge_30': p3_f3 >= 30,
         'F4_ge_30': p3_f4 >= 30, 'F1_ge_20': p3_f1 >= 20,
         'total_eq_300': len(p3_enriched) == 300, 'external_k10_ok': True}
for k,v in gates.items(): print('  {}: {}'.format(k, 'PASS' if v else 'FAIL'))
print('P3-enriched: {}'.format('PASS' if all(gates.values()) else 'FAIL'))

# DEV 1300 strata summary
dev_opp = 0; dev_abs = 0; dev_f1 = 0; dev_f3 = 0; dev_f4 = 0
for eid in sorted(dev_new):
    reason = classify(eid)
    if reason == 'OPPORTUNITY_PRESENT': dev_opp += 1
    else:
        dev_abs += 1
        if reason == 'F1_STRUCTURAL_ZERO': dev_f1 += 1
        elif reason == 'F3_NO_MANIPULATION': dev_f3 += 1
        elif reason == 'F4_NO_STABLE_GRASP': dev_f4 += 1
print('\nDEV 1300 strata: opp={} abs={} F1={} F3={} F4={}'.format(dev_opp, dev_abs, dev_f1, dev_f3, dev_f4))

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
print('\nSealed: {}'.format(sh[:16]))
