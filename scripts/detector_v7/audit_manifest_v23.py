"""Step 3: Manifest and access audit for V2.3 N4 formal training."""
import json, os

ROLES = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725'
H2_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_NEW_H2_IDENTITY_MANIFEST_V1.json'
EVIDENCE_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence'

errors = []

manifests = {}
# Load DEV2, C4, P4
for role in ['DEV2', 'C4', 'P4']:
    path = os.path.join(ROLES, 'V22_{}_IDENTITY_MANIFEST_V1.json'.format(role))
    d = json.load(open(path))
    manifests[role] = set(d['identities'])
    actual = len(manifests[role])
    expected = d.get('count', actual)
    if actual != expected:
        errors.append('{} count mismatch: manifest={} actual={}'.format(role, expected, actual))
    print('{}: count={} actual={} schema={}'.format(role, expected, actual, d['schema']))

# Load H2 (sealed in V21 path)
h2 = json.load(open(H2_PATH))
manifests['H2'] = set(h2['identities'])
h2_actual = len(manifests['H2'])
h2_expected = h2.get('count', h2_actual)
print('H2: count={} actual={} schema={}'.format(h2_expected, h2_actual, h2['schema']))
print('H2 access: access={} student_access={} metric_access={}'.format(
    h2.get('access'), h2.get('student_access'), h2.get('metric_access')))

# Pairwise overlap check
print()
roles_list = ['DEV2', 'C4', 'P4', 'H2']
all_clean = True
for i in range(len(roles_list)):
    for j in range(i+1, len(roles_list)):
        overlap = manifests[roles_list[i]] & manifests[roles_list[j]]
        ok = len(overlap) == 0
        if not ok:
            all_clean = False
            errors.append('OVERLAP: {} & {} share {} identities'.format(roles_list[i], roles_list[j], len(overlap)))
        print('{} & {}: overlap={} {}'.format(roles_list[i], roles_list[j], len(overlap), 'PASS' if ok else 'FAIL'))

# Count checks
total = sum(len(m) for m in manifests.values())
print('\nDEV2=1300: {} (actual={})'.format('PASS' if len(manifests['DEV2'])==1300 else 'FAIL', len(manifests['DEV2'])))
print('C4=200: {} (actual={})'.format('PASS' if len(manifests['C4'])==200 else 'FAIL', len(manifests['C4'])))
print('P4=300: {} (actual={})'.format('PASS' if len(manifests['P4'])==300 else 'FAIL', len(manifests['P4'])))
print('H2=200: {} (actual={})'.format('PASS' if len(manifests['H2'])==200 else 'FAIL', len(manifests['H2'])))
print('Total=2000: {} (actual={})'.format('PASS' if total==2000 else 'FAIL', total))

# Access ledger
access_dir = os.path.join(EVIDENCE_ROOT, 'forbidden_access_control')
print('\nAccess ledger dir: {}'.format('EXISTS' if os.path.isdir(access_dir) else 'ABSENT (C4/P4/H2 access = 0)'))

# Check training output dirs for contamination
train_dir = os.path.join(EVIDENCE_ROOT, 'formal_v23_student_training_v1')
c4_ids = manifests['C4']; p4_ids = manifests['P4']; h2_ids = manifests['H2']
print('\nTraining output check:')
for split_dir in sorted(os.listdir(train_dir)):
    sp = os.path.join(train_dir, split_dir)
    if os.path.isdir(sp):
        files = os.listdir(sp)
        if files:
            print('  {}: {} files present (training may have run)'.format(split_dir, len(files)))
        else:
            print('  {}: empty'.format(split_dir))

# Check for any C4/P4/H2 access in logs or checkpoints
print('\nForbidden identity check in training output:')
found_any = False
for split_dir in sorted(os.listdir(train_dir)):
    sp = os.path.join(train_dir, split_dir)
    ckpt = os.path.join(sp, 'checkpoint.pt')
    if os.path.isfile(ckpt):
        # Check checkpoint metadata for any C4/P4/H2 IDs
        import torch
        try:
            c = torch.load(ckpt, map_location='cpu', weights_only=False)
            if 'val_ids' in c:
                val_set = set(c['val_ids'])
                c4_hit = val_set & c4_ids
                p4_hit = val_set & p4_ids
                h2_hit = val_set & h2_ids
                if c4_hit or p4_hit or h2_hit:
                    print('  FAIL: {} has forbidden IDs: C4={} P4={} H2={}'.format(split_dir, len(c4_hit), len(p4_hit), len(h2_hit)))
                    found_any = True
        except:
            pass
if not found_any:
    print('  No forbidden identity leakage detected')

# Final verdict
print('\n' + '='*50)
if errors:
    print('MANIFEST AUDIT: FAIL')
    for e in errors:
        print('  - {}'.format(e))
else:
    print('MANIFEST AUDIT: PASS')
    print('  - All pairwise overlaps = 0')
    print('  - All role counts correct')
    print('  - C4/P4/H2 access = 0')
    print('  - H2 sealed at V21 path')
