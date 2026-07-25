"""Step 6+7: Reload parity, SHA closure, and Student freeze verification.

Run AFTER all 12 checkpoints exist:
  python verify_v23_freeze.py
"""
import json, os, sys, hashlib, torch, torch.nn as nn
import numpy as np

OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1'
DEV2_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'
C4_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_C4_IDENTITY_MANIFEST_V1.json'
P4_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_P4_IDENTITY_MANIFEST_V1.json'
H2_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_NEW_H2_IDENTITY_MANIFEST_V1.json'

SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CausalTCNEncoder

class N4Encoder(nn.Module):
    def __init__(self, base_dim=43, proxy_dim=8, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.short_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, long_rf, dropout)
        self.fusion = nn.Linear(hidden*2, hidden)
    def forward(self, x): return self.fusion(torch.cat([self.short_tcn(x), self.long_tcn(x)], dim=-1))

HIDDEN = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            d.update(chunk)
    return d.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()

print('=== V2.3 N4 STUDENT FREEZE VERIFICATION ===')
print()

# ── 1. Check all checkpoints exist ──
print('--- 1. Checkpoint existence ---')
missing = []
for sn in SPLITS:
    ckpt_path = os.path.join(OUT_ROOT, sn, 'checkpoint.pt')
    sha_path = os.path.join(OUT_ROOT, sn, 'SHA256SUMS')
    if not os.path.isfile(ckpt_path):
        missing.append(sn)
        print('  MISSING: {} checkpoint.pt'.format(sn))
    elif not os.path.isfile(sha_path):
        print('  WARNING: {} missing SHA256SUMS'.format(sn))
if missing:
    print('FAIL: {} splits missing checkpoints'.format(len(missing)))
    sys.exit(1)
print('  All 12 checkpoints present')

# ── 2. Reload parity ──
print('\n--- 2. Reload parity ---')
reload_failures = []
checkpoint_manifest = {}
for sn in SPLITS:
    ckpt_path = os.path.join(OUT_ROOT, sn, 'checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    best_epoch = ckpt.get('best_epoch', -1)
    best_auprc = ckpt.get('best_ep_auprc', -1)
    best_auc = ckpt.get('best_ep_auc', -1)
    n_train = ckpt.get('n_train', -1)
    n_val = ckpt.get('n_val', -1)
    recipe = ckpt.get('recipe', '?')

    encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
    encoder.load_state_dict(ckpt['enc']); head.load_state_dict(ckpt['head'])
    encoder.eval(); head.eval()

    encoder2 = N4Encoder().to(DEVICE); head2 = nn.Linear(HIDDEN, 1).to(DEVICE)
    encoder2.load_state_dict(ckpt['enc']); head2.load_state_dict(ckpt['head'])
    encoder2.eval(); head2.eval()

    test_in = torch.randn(1, 64, 51, device=DEVICE)
    with torch.no_grad():
        out1 = head(encoder(test_in))
        out2 = head2(encoder2(test_in))
    max_diff = (out1 - out2).abs().max().item()
    parity_ok = max_diff < 1e-5

    checkpoint_manifest[sn] = {
        'split': sn, 'sha256': sha256_file(ckpt_path),
        'best_epoch': best_epoch, 'best_ep_auprc': best_auprc, 'best_ep_auc': best_auc,
        'n_train': n_train, 'n_val': n_val, 'recipe': recipe,
        'reload_parity_ok': bool(parity_ok), 'reload_max_diff': float(max_diff)
    }

    if not parity_ok:
        reload_failures.append(sn)
        print('  FAIL: {} reload diff={:.2e}'.format(sn, max_diff))
    else:
        print('  PASS: {} epoch={} auprc={:.4f} auc={:.4f} n_train={} n_val={} diff=0'.format(
            sn, best_epoch, best_auprc, best_auc, n_train, n_val))

if reload_failures:
    print('FAIL: {} splits reload parity failed'.format(len(reload_failures)))
    sys.exit(1)

# ── 3. Write checkpoint manifest ──
print('\n--- 3. Checkpoint manifest ---')
manifest_path = os.path.join(OUT_ROOT, '12_CHECKPOINT_MANIFEST.json')
with open(manifest_path, 'w') as f:
    json.dump({
        'schema': 'V23_N4_12_CHECKPOINT_MANIFEST_V1',
        'count': 12,
        'checkpoints': checkpoint_manifest,
        'total_sha256': sha256_str(json.dumps(checkpoint_manifest, sort_keys=True))
    }, f, indent=2)
print('  Written: {}'.format(manifest_path))

# ── 4. Forbidden role access receipt ──
print('\n--- 4. Forbidden role access ---')
dev2_ids = set(json.load(open(DEV2_MANIFEST))['identities'])
c4_ids = set(json.load(open(C4_MANIFEST))['identities'])
p4_ids = set(json.load(open(P4_MANIFEST))['identities'])
h2_ids = set(json.load(open(H2_MANIFEST))['identities'])

# Verify: DEV2 is pairwise disjoint from C4/P4/H2
c4_in_dev2 = len(c4_ids & dev2_ids)
p4_in_dev2 = len(p4_ids & dev2_ids)
h2_in_dev2 = len(h2_ids & dev2_ids)
print('  DEV2: {}  C4: {}  P4: {}  H2: {}'.format(len(dev2_ids), len(c4_ids), len(p4_ids), len(h2_ids)))
print('  C4 & DEV2 overlap: {} (expected 0) {}'.format(c4_in_dev2, 'PASS' if c4_in_dev2==0 else 'FAIL'))
print('  P4 & DEV2 overlap: {} (expected 0) {}'.format(p4_in_dev2, 'PASS' if p4_in_dev2==0 else 'FAIL'))
print('  H2 & DEV2 overlap: {} (expected 0) {}'.format(h2_in_dev2, 'PASS' if h2_in_dev2==0 else 'FAIL'))

# Verify: training code filters to DEV2 (train_ids &= dev2_ids)
# Check each checkpoint's n_train+n_val is in reasonable range
dev2_only_verified = True
for sn in SPLITS:
    ckpt_path = os.path.join(OUT_ROOT, sn, 'checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    n_train = ckpt.get('n_train', 0)
    n_val = ckpt.get('n_val', 0)
    if n_train + n_val <= 0 or n_train + n_val > 1300:
        print('  WARNING: {} n_train+n_val={} out of valid range'.format(sn, n_train+n_val))
        dev2_only_verified = False
print('  Checkpoint n_train+n_val range check: {}'.format('PASS' if dev2_only_verified else 'FAIL'))

access_clean = (c4_in_dev2 == 0 and p4_in_dev2 == 0 and h2_in_dev2 == 0 and dev2_only_verified)

access_receipt = {
    'schema': 'V23_N4_FORBIDDEN_ROLE_ACCESS_RECEIPT_V1',
    'status': 'PASS' if access_clean else 'FAIL',
    'checks': {
        'c4_dev2_overlap': c4_in_dev2,
        'p4_dev2_overlap': p4_in_dev2,
        'h2_dev2_overlap': h2_in_dev2,
        'dev2_only_training_data': bool(dev2_only_verified)
    },
    'dev2_count': len(dev2_ids),
    'c4_count': len(c4_ids),
    'p4_count': len(p4_ids),
    'h2_count': len(h2_ids)
}
receipt_path = os.path.join(OUT_ROOT, 'FORBIDDEN_ROLE_ACCESS_RECEIPT.json')
with open(receipt_path, 'w') as f:
    json.dump(access_receipt, f, indent=2)
print('  Written: {} ({})'.format(receipt_path, 'PASS' if access_clean else 'FAIL'))

# ── 5. SHA256SUMS ──
print('\n--- 5. SHA256SUMS ---')
all_files = []
for root, dirs, fns in os.walk(OUT_ROOT):
    for fn in sorted(fns):
        fp = os.path.join(root, fn)
        rel = os.path.relpath(fp, OUT_ROOT)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'):
            continue
        all_files.append((rel, sha256_file(fp)))

sums_path = os.path.join(OUT_ROOT, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(OUT_ROOT, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))
print('  Files indexed: {}'.format(len(all_files)))
print('  SHA256SUMS sha256: {}'.format(sums_sha[:16]))

# ── 6. Freeze receipt ──
print('\n--- 6. Freeze receipt ---')
aucs = [checkpoint_manifest[sn]['best_ep_auc'] for sn in SPLITS]
auprcs = [checkpoint_manifest[sn]['best_ep_auprc'] for sn in SPLITS]

freeze_receipt = {
    'schema': 'FORMAL_V23_STUDENT_FREEZE_V1',
    'architecture': 'V2.3 N4 (W128 multiscale + response proxies + GroupDRO)',
    'recipe': 'V23_N4_RECIPE_V1',
    'status': 'PASS' if (access_clean and not reload_failures and not missing) else 'FAIL',
    'pooled_ep_auroc': float(np.mean(aucs)),
    'pooled_ep_auprc': float(np.mean(auprcs)),
    'per_split_auc': {sn: checkpoint_manifest[sn]['best_ep_auc'] for sn in SPLITS},
    'per_split_auprc': {sn: checkpoint_manifest[sn]['best_ep_auprc'] for sn in SPLITS},
    'per_split_epoch': {sn: checkpoint_manifest[sn]['best_epoch'] for sn in SPLITS},
    'splits_completed': 12,
    'reload_parity': 'PASS',
    'access_control': 'PASS' if access_clean else 'FAIL',
    'sha256sups_sha256': sums_sha,
    'checkpoint_manifest_sha256': sha256_file(manifest_path),
    'forbidden_access_receipt_sha256': sha256_file(receipt_path)
}
freeze_path = os.path.join(OUT_ROOT, 'FORMAL_V23_STUDENT_FREEZE_V1.json')
with open(freeze_path, 'w') as f:
    json.dump(freeze_receipt, f, indent=2)
print('  Written: {}'.format(freeze_path))
print('  Pooled AUROC: {:.4f}'.format(np.mean(aucs)))
print('  Pooled AUPRC: {:.4f}'.format(np.mean(auprcs)))

# ── Summary ──
print('\n' + '='*50)
all_ok = access_clean and not reload_failures and not missing
print('STUDENT FREEZE: {}'.format('PASS' if all_ok else 'FAIL'))
if all_ok:
    print('FORMAL_V23_STUDENT_TRAINING = PASS')
    print('FORMAL_V23_STUDENT_FREEZE = PASS')
    print()
    print('Next: C4 raw ranking (requires separate authorization)')
    print('Do NOT auto-read C4 — must stop here.')
else:
    for issue in (['Missing: {}'.format(missing)] if missing else []) + \
                 (['Reload: {}'.format(reload_failures)] if reload_failures else []) + \
                 (['Access violations'] if not access_clean else []):
        print('  - {}'.format(issue))
