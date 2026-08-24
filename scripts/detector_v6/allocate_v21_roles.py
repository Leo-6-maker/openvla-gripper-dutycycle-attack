"""Allocate 600 unused identities → C3 / P3 / new-H2.

Per-task interleaved: 5 states each, no contiguous state blocks.
Verifies P3 hard-negative coverage before freezing manifests.
Seed frozen. No Student scores used.
"""
import json, os, hashlib, time, random
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
OUT_DIR = '/mnt/sdc/dty_user/openvla_attack_evidence/v21_role_allocation'
SEED = 20260725; K10 = 10
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)

# ── 1. Identify consumed identities ──
print('=== V2.1 ROLE ALLOCATION ===')
consumed = set()
for root in ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']:
    if not os.path.isdir(root): continue
    for suite in sorted(os.listdir(root)):
        sp = os.path.join(root, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp): continue
            for state in sorted(os.listdir(tp)):
                consumed.add(f'{suite}/{task}/{state}')

# All clean identities
all_ids = set()
for suite in sorted(os.listdir(FEAT_ROOT)):
    sp = os.path.join(FEAT_ROOT, suite)
    if not os.path.isdir(sp): continue
    for task in sorted(os.listdir(sp)):
        tp = os.path.join(sp, task)
        if not os.path.isdir(tp): continue
        for state in sorted(os.listdir(tp)):
            all_ids.add(f'{suite}/{task}/{state}')

unused = sorted(all_ids - consumed)
print(f'Total: {len(all_ids)}  Consumed: {len(consumed)}  Unused: {len(unused)}')

# ── 2. Per-task allocation ──
# Group unused by task, sort states numerically within each task
task_states = defaultdict(list)
for eid in unused:
    parts = eid.split('/')
    task_key = f'{parts[0]}/{parts[1]}'
    state_num = int(parts[2].replace('state_',''))
    task_states[task_key].append((state_num, eid))

print(f'Tasks with unused states: {len(task_states)}')

# Verify: each task should have exactly 15 unused states
state_counts = {k: len(v) for k, v in task_states.items()}
bad_tasks = {k: v for k, v in state_counts.items() if v != 15}
if bad_tasks:
    print(f'WARNING: Tasks with !=15 unused states: {bad_tasks}')

# Allocate: sort states within task, take every 3rd state for each role
c3_ids = set(); p3_ids = set(); h2_new_ids = set()
allocation_detail = {}

for task_key, states in sorted(task_states.items()):
    states.sort()  # sort by state number
    # Interleave: states[0::3] → C3, states[1::3] → P3, states[2::3] → H2
    c3_batch = set(eid for _, eid in states[0::3])
    p3_batch = set(eid for _, eid in states[1::3])
    h2_batch = set(eid for _, eid in states[2::3])

    c3_ids.update(c3_batch); p3_ids.update(p3_batch); h2_new_ids.update(h2_batch)
    allocation_detail[task_key] = {
        'C3': sorted([int(eid.split('/')[2].replace('state_','')) for eid in c3_batch]),
        'P3': sorted([int(eid.split('/')[2].replace('state_','')) for eid in p3_batch]),
        'H2': sorted([int(eid.split('/')[2].replace('state_','')) for eid in h2_batch]),
    }

print(f'Allocated: C3={len(c3_ids)} P3={len(p3_ids)} H2={len(h2_new_ids)}')
print(f'Overlap C3∩P3={len(c3_ids & p3_ids)} C3∩H2={len(c3_ids & h2_new_ids)} P3∩H2={len(p3_ids & h2_new_ids)}')
print(f'Overlap with consumed: {len(c3_ids & consumed)} {len(p3_ids & consumed)} {len(h2_new_ids & consumed)}')

# ── 3. Verify P3 hard-negative coverage ──
print('\n=== P3 COVERAGE VERIFICATION ===')
p3_opp = 0; p3_abs = 0; p3_f1 = 0; p3_f3 = 0; p3_f4 = 0; p3_parser = 0

for eid in sorted(p3_ids):
    parts = eid.split('/')
    # Find teacher labels (check both CAL and CHECK roots since states are from later range)
    found = False
    for root in ['/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']:
        if found: break
        tp = os.path.join(root, parts[0], parts[1], parts[2], 'factorized_teacher_v1.jsonl')
        if not os.path.isfile(tp): continue
        tr = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
        tr.sort(key=lambda r: r['step']); T = len(tr); max_t = min(T, T-K10+1)
        found = True

        any_k10_known = any(tr[t].get('strict_k10_known_mask',False) for t in range(max_t))
        any_manip_known = any(tr[t].get('manipulation_active_known_mask',False) for t in range(max_t))
        any_grasp_known = any(tr[t].get('grasp_established_known_mask',False) for t in range(max_t))
        n_k10_pos = sum(1 for t in range(max_t) if tr[t].get('strict_k10_feasible',False) and tr[t].get('strict_k10_known_mask',False))
        n_manip_pos = sum(1 for t in range(max_t) if tr[t].get('manipulation_active',False) and tr[t].get('manipulation_active_known_mask',False))
        n_grasp_pos = sum(1 for t in range(max_t) if tr[t].get('grasp_established',False) and tr[t].get('grasp_established_known_mask',False))

        # Parser check
        n_g_known = sum(1 for t in range(max_t) if tr[t].get('grasp_established_known_mask',False))
        n_m_known = sum(1 for t in range(max_t) if tr[t].get('manipulation_active_known_mask',False))
        is_parser = (n_k10_pos > 0 and n_g_known == 0 and n_m_known == 0)

        if n_k10_pos > 0: p3_opp += 1
        else:
            p3_abs += 1
            if not any_k10_known: p3_f1 += 1
            elif n_manip_pos == 0 and any_manip_known: p3_f3 += 1
            elif n_grasp_pos == 0 and any_grasp_known: p3_f4 += 1
            elif is_parser: p3_parser += 1

print(f'P3: opp={p3_opp} abs={p3_abs} F1={p3_f1} F3={p3_f3} F4={p3_f4} parser={p3_parser}')
gates = {
    'total_absent_ge_60': p3_abs >= 60,
    'F3_ge_15': p3_f3 >= 15,
    'F4_ge_15': p3_f4 >= 15,
    'opp_ge_60': p3_opp >= 60,
    'parser_eq_0': p3_parser == 0,
}
all_pass = all(gates.values())
for k, v in gates.items():
    print(f'  {k}: {"PASS" if v else "FAIL"}')
print(f'P3 coverage: {"PASS" if all_pass else "FAIL"}')

# Also verify H2
h2_opp = 0; h2_abs = 0
for eid in sorted(h2_new_ids):
    parts = eid.split('/')
    for root in ['/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']:
        tp = os.path.join(root, parts[0], parts[1], parts[2], 'factorized_teacher_v1.jsonl')
        if not os.path.isfile(tp): continue
        tr = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
        T = len(tr); max_t = min(T, T-K10+1)
        n_k10_pos = sum(1 for t in range(max_t) if tr[t].get('strict_k10_feasible',False) and tr[t].get('strict_k10_known_mask',False))
        if n_k10_pos > 0: h2_opp += 1
        else: h2_abs += 1
        break  # found labels

print(f'\nNew H2: opp={h2_opp} abs={h2_abs} total={len(h2_new_ids)}')

# ── 4. Generate manifests ──
manifest_c3 = {'schema': 'V21_C3_IDENTITY_MANIFEST_V1', 'role': 'C3', 'identities': sorted(c3_ids), 'count': len(c3_ids)}
manifest_p3 = {'schema': 'V21_P3_IDENTITY_MANIFEST_V1', 'role': 'P3', 'identities': sorted(p3_ids), 'count': len(p3_ids),
               'coverage': {'opp': p3_opp, 'abs': p3_abs, 'F1': p3_f1, 'F3': p3_f3, 'F4': p3_f4, 'parser': p3_parser},
               'coverage_pass': all_pass}
manifest_h2 = {'schema': 'V21_H2_IDENTITY_MANIFEST_V1', 'role': 'H2', 'identities': sorted(h2_new_ids), 'count': len(h2_new_ids),
               'access': 'FORBIDDEN_UNTIL_FORMAL_EVALUATION', 'note': 'Original H2 consumed by H1. This is NEW H2.'}

with open(os.path.join(OUT_DIR, 'C3_identity_manifest.json'), 'w') as f: json.dump(manifest_c3, f, indent=2)
with open(os.path.join(OUT_DIR, 'P3_identity_manifest.json'), 'w') as f: json.dump(manifest_p3, f, indent=2)
with open(os.path.join(OUT_DIR, 'H2_identity_manifest.json'), 'w') as f: json.dump(manifest_h2, f, indent=2)

# Allocation receipt
receipt = {
    'schema': 'V21_ROLE_ALLOCATION_RECEIPT_V1',
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'seed': SEED, 'method': 'per-task state interleaving (every 3rd state)',
    'allocation': {'C3': len(c3_ids), 'P3': len(p3_ids), 'H2': len(h2_new_ids)},
    'P3_coverage_gates': gates,
    'P3_coverage_pass': all_pass,
    'H2_access': 'FORBIDDEN',
    'consumed_pool': len(consumed),
    'unused_allocated': len(c3_ids) + len(p3_ids) + len(h2_new_ids),
    'note': 'Original heldout_l3 (200 identities) invalidated by H1 consumption. These 600 are from states never seen by any V1/V2 pipeline.',
}
with open(os.path.join(OUT_DIR, 'V21_ROLE_ALLOCATION_RECEIPT_V1.json'), 'w') as f: json.dump(receipt, f, indent=2)

# Seal
def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

all_files = []
for root, dirs, fns in os.walk(OUT_DIR):
    for fn in sorted(fns):
        if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_DIR)
        all_files.append((rel, sha256_file(fp)))
with open(os.path.join(OUT_DIR, 'SHA256SUMS'), 'w') as f:
    for rel, h in sorted(all_files): f.write(f'{h}  {rel}\n')
sh = sha256_file(os.path.join(OUT_DIR, 'SHA256SUMS'))
with open(os.path.join(OUT_DIR, 'SHA256SUMS.sha256'), 'w') as f:
    f.write(f'{sh}  SHA256SUMS\n')

print(f'\nAllocation sealed: {OUT_DIR}')
print(f'Seal: {sh[:16]}')
print(f'H2: FORBIDDEN ({len(h2_new_ids)} identities)')
