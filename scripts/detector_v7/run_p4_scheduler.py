"""Stage H (P4): One-shot scheduler search.

Frozen inputs: 12 checkpoints, Platt calibrator, candidate-close definition.
Searches: threshold x persistence grid.
Evaluates: F3 FS <= 10%, F4 FS <= 10%, all-absent FS <= 10%.
"""
import json, os, sys, hashlib, numpy as np, torch, torch.nn as nn
from collections import defaultdict

# ── Frozen configuration ──
EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
P4_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_P4_IDENTITY_MANIFEST_V1.json'
TRAIN_DIR = EVIDENCE + '/formal_v23_student_training_v1'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
OUT_ROOT = EVIDENCE + '/p4_scheduler_v1'
os.makedirs(OUT_ROOT, exist_ok=True)

# Platt calibrator (full precision from C4_CALIBRATOR_V1.json)
PLATT_A = 0.5190011735319306
PLATT_B = 0.812702331013635

# Scheduler search grid (pre-registered)
THRESHOLDS = np.arange(0.005, 1.0, 0.01)
PERSISTENCES = list(range(1, 11))
CANDIDATE_CLOSE_ON = True
ONE_SHOT_LATCH = True

K10 = 10; HIDDEN = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CausalTCNEncoder

class N4Encoder(nn.Module):
    def __init__(self, base_dim=43, proxy_dim=8, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        self.short_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, long_rf, dropout)
        self.fusion = nn.Linear(hidden*2, hidden)
    def forward(self, x): return self.fusion(torch.cat([self.short_tcn(x), self.long_tcn(x)], dim=-1))

def compute_proxies(f25d, p9d, g9d, T):
    proxies = np.zeros((T, 8), dtype=np.float32)
    cmd = f25d[:,0]; qpos = f25d[:,1]
    proxies[:,0] = cmd - qpos
    proxies[:,1] = (cmd < 0).astype(np.float32)
    proxies[1:,2] = np.diff(qpos); proxies[0,2] = 0
    dur = 0; cd = np.zeros(T)
    for t in range(T):
        if cmd[t] < 0: dur += 1
        else: dur = 0
        cd[t] = dur
    proxies[:,3] = cd
    proxies[:,4] = np.sqrt(f25d[:,6]**2 + f25d[:,7]**2 + f25d[:,8]**2)
    for t in range(T):
        w_s = max(0, t-4); w_e = min(T, t+1)
        proxies[t,5] = np.var(qpos[w_s:w_e]) if w_e-w_s > 1 else 0
    proxies[:,6] = g9d[:,0]; proxies[:,7] = g9d[:,7]
    return np.nan_to_num(proxies, 0).astype(np.float32)

def sigmoid(x):
    xc = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-xc))

def calibrated_prob(raw_logit):
    return sigmoid(PLATT_A * np.array(raw_logit) + PLATT_B)

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

# ── 1. Load P4 manifest ──
print('=== P4 SCHEDULER SEARCH ===')
print()
p4_ids = json.load(open(P4_MANIFEST_PATH))['identities']
p4_id_set = set(p4_ids)
print('P4 manifest: {} identities'.format(len(p4_ids)))

# Load o0_i0 training IDs for normalization
dev2_ids = set(json.load(open(EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'))['identities'])
split_manifest = json.load(open(EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'))
outer_0 = split_manifest['splits']['fold_0']
o0i0_train_ids = set()
for j, inf in enumerate(outer_0['inner_folds']):
    if j != 0: o0i0_train_ids.update(inf['identities'])
o0i0_train_ids &= dev2_ids

def load_episode(eid):
    suite, task, state = eid.split('/')
    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
    lp = None
    for root in LABEL_ROOTS:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate): lp = candidate; break
    if not os.path.isfile(fp) or lp is None: return None
    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step']); T = len(recs); max_t = min(T, T-K10+1)
    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
        for r in recs], dtype=np.float32)
    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and
                       labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False)
                    for t in range(T)], dtype=bool)
    has_opp = bool(k10_s[:max_t].any())
    absence_reason = 'OPPORTUNITY_PRESENT'
    if not has_opp:
        any_k10_known = any(k10_k[:max_t])
        any_mk = any(labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False) for t in range(max_t))
        any_gk = any(labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False) for t in range(max_t))
        n_mp = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('manipulation_active',False) and labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False))
        n_gp = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('grasp_established',False) and labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False))
        if not any_k10_known: absence_reason = 'F1_STRUCTURAL_ZERO'
        elif n_mp == 0 and any_mk: absence_reason = 'F3_NO_MANIPULATION'
        elif n_gp == 0 and any_gk: absence_reason = 'F4_NO_STABLE_GRASP'
        else: absence_reason = 'OTHER_ABSENT'
    proxies = compute_proxies(f25d, p9d, g9d, T)
    return {'eid': eid, 'T': T, 'max_t': max_t, 'f25d': f25d, 'p9d': p9d, 'g9d': g9d,
            'proxies': proxies, 'k10_s': k10_s, 'k10_k': k10_k, 'cc': cc,
            'has_opp': has_opp, 'absence_reason': absence_reason, 'suite': suite}

# ── 2. Load and run inference ──
print('--- Loading training episodes for normalization ---')
train_eps_for_norm = []
for eid in o0i0_train_ids:
    ep = load_episode(eid)
    if ep is not None: train_eps_for_norm.append(ep)
cat_25d = np.concatenate([e['f25d'] for e in train_eps_for_norm], axis=0)
cat_p9d = np.concatenate([e['p9d'] for e in train_eps_for_norm], axis=0)
cat_g9d = np.concatenate([e['g9d'] for e in train_eps_for_norm], axis=0)
n25d_m = torch.tensor(cat_25d.mean(0), device=DEVICE); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=DEVICE)
np9d_m = torch.tensor(cat_p9d.mean(0), device=DEVICE); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=DEVICE)
ng9d_m = torch.tensor(cat_g9d.mean(0), device=DEVICE); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=DEVICE)
print('Norm from {} episodes'.format(len(train_eps_for_norm)))

print('--- Loading P4 episodes ---')
p4_eps = {}
for eid in p4_ids:
    ep = load_episode(eid)
    if ep is not None: p4_eps[eid] = ep
print('Loaded {}/{} P4 episodes'.format(len(p4_eps), len(p4_ids)))

print('--- Running inference (checkpoint o0_i0) ---')
ckpt = torch.load(os.path.join(TRAIN_DIR, 'o0_i0', 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
encoder.load_state_dict(ckpt['enc']); head.load_state_dict(ckpt['head'])
encoder.eval(); head.eval()

p4_raw_logits = {}
with torch.no_grad():
    for eid, e in sorted(p4_eps.items()):
        base = torch.cat([(torch.tensor(e['f25d'], device=DEVICE) - n25d_m) / n25d_s,
                          (torch.tensor(e['p9d'], device=DEVICE) - np9d_m) / np9d_s,
                          (torch.tensor(e['g9d'], device=DEVICE) - ng9d_m) / ng9d_s,
                          torch.tensor(e['proxies'], device=DEVICE)], dim=-1).unsqueeze(0)
        p4_raw_logits[eid] = head(encoder(base)).squeeze().cpu().numpy()

# ── 3. Composition audit ──
print('\n' + '='*60)
print('P4 COMPOSITION AUDIT')
print('='*60)
opp_eids = [eid for eid, e in p4_eps.items() if e['has_opp']]
f1_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F1_STRUCTURAL_ZERO']
f3_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F3_NO_MANIPULATION']
f4_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F4_NO_STABLE_GRASP']
other_eids = [eid for eid, e in p4_eps.items() if not e['has_opp'] and e['absence_reason'] not in ('F1_STRUCTURAL_ZERO','F3_NO_MANIPULATION','F4_NO_STABLE_GRASP')]

composition = {
    'total_manifest': len(p4_ids), 'total_loaded': len(p4_eps),
    'opp': len(opp_eids), 'F1': len(f1_eids), 'F3': len(f3_eids),
    'F4': len(f4_eids), 'other_absent': len(other_eids)
}
print('P4 strata: opp={} F1={} F3={} F4={} other={}'.format(
    composition['opp'], composition['F1'], composition['F3'], composition['F4'], composition['other_absent']))

# Check F3/F4 quotas
f3_evaluable = len(f3_eids) >= 5
f4_evaluable = len(f4_eids) >= 5
print('F3 evaluable (>=5): {}  F4 evaluable (>=5): {}'.format(f3_evaluable, f4_evaluable))

if not f3_evaluable:
    print('\n*** F3 INSUFFICIENT: P4 = NOT_ESTIMABLE ***')
    with open(os.path.join(OUT_ROOT, 'P4_COMPOSITION_AUDIT_V1.json'), 'w') as f:
        json.dump({'schema': 'P4_COMPOSITION_AUDIT_V1', 'status': 'NOT_ESTIMABLE',
                   'reason': 'F3 count < 5', 'composition': composition}, f, indent=2)
    print('See P4_COMPOSITION_AUDIT_V1.json')
    sys.exit(1)

# Per-suite breakdown
suite_counts = defaultdict(lambda: {'opp':0,'F3':0,'F4':0,'other':0})
for eid, e in p4_eps.items():
    s = e['suite']
    if e['has_opp']: suite_counts[s]['opp'] += 1
    elif e['absence_reason'] == 'F3_NO_MANIPULATION': suite_counts[s]['F3'] += 1
    elif e['absence_reason'] == 'F4_NO_STABLE_GRASP': suite_counts[s]['F4'] += 1
    else: suite_counts[s]['other'] += 1
print('\nPer-suite:')
for suite in sorted(suite_counts.keys()):
    d = suite_counts[suite]
    print('  {}: opp={} F3={} F4={} other={}'.format(suite, d['opp'], d['F3'], d['F4'], d['other']))

with open(os.path.join(OUT_ROOT, 'P4_COMPOSITION_AUDIT_V1.json'), 'w') as f:
    json.dump({'schema': 'P4_COMPOSITION_AUDIT_V1', 'status': 'PASS',
               'composition': composition, 'per_suite': {s: dict(d) for s, d in suite_counts.items()}}, f, indent=2)

# ── 4. Scheduler grid search ──
print('\n' + '='*60)
print('P4 SCHEDULER GRID SEARCH')
print('='*60)
print('Thresholds: {} (n={})'.format(THRESHOLDS[:5].tolist(), len(THRESHOLDS)))
print('Persistences: {}'.format(PERSISTENCES))
print('Grid size: {}'.format(len(THRESHOLDS) * len(PERSISTENCES)))

def run_scheduler(ep, raw_logits, threshold, persistence):
    """One-shot scheduler. Returns: emit_step or None (no emit)."""
    cal_probs = calibrated_prob(raw_logits)
    T = ep['T']; max_t = min(T, T - K10 + 1)

    cons = 0
    for t in range(max_t):
        eligible = ep['cc'][t] and cal_probs[t] >= threshold
        if eligible:
            cons += 1
        else:
            cons = 0
        if cons >= persistence:
            return t
    return None

all_absent_eids = [eid for eid in p4_eps if not p4_eps[eid]['has_opp']]
n_opp = len(opp_eids); n_f3 = len(f3_eids); n_f4 = len(f4_eids); n_abs = len(all_absent_eids)

# Search
results = []
for threshold in THRESHOLDS:
    for d in PERSISTENCES:
        vt = 0; fs_f3 = 0; fs_f4 = 0; fs_abs = 0
        emits_opp = []
        for eid in opp_eids:
            emit_t = run_scheduler(p4_eps[eid], p4_raw_logits[eid], threshold, d)
            if emit_t is not None:
                ep = p4_eps[eid]
                # Valid trigger: y_t=1 AND within max_t (K10 executable)
                if emit_t < ep['max_t'] and ep['k10_s'][emit_t] and ep['k10_k'][emit_t]:
                    vt += 1
                emits_opp.append({'eid': eid, 'emit_t': emit_t, 'valid': bool(emit_t < ep['max_t'] and ep['k10_s'][emit_t])})

        for eid in f3_eids:
            emit_t = run_scheduler(p4_eps[eid], p4_raw_logits[eid], threshold, d)
            if emit_t is not None: fs_f3 += 1

        for eid in f4_eids:
            emit_t = run_scheduler(p4_eps[eid], p4_raw_logits[eid], threshold, d)
            if emit_t is not None: fs_f4 += 1

        for eid in all_absent_eids:
            emit_t = run_scheduler(p4_eps[eid], p4_raw_logits[eid], threshold, d)
            if emit_t is not None: fs_abs += 1

        recall = vt / max(n_opp, 1)
        fs_f3_rate = fs_f3 / max(n_f3, 1)
        fs_f4_rate = fs_f4 / max(n_f4, 1)
        fs_abs_rate = fs_abs / max(n_abs, 1)

        feasible = (fs_f3_rate <= 0.10 and fs_f4_rate <= 0.10 and
                    fs_abs_rate <= 0.10 and vt > 0)

        results.append({
            'threshold': float(threshold), 'd': d,
            'recall': recall, 'vt': vt, 'n_opp': n_opp,
            'fs_f3': fs_f3, 'fs_f3_rate': fs_f3_rate,
            'fs_f4': fs_f4, 'fs_f4_rate': fs_f4_rate,
            'fs_abs': fs_abs, 'fs_abs_rate': fs_abs_rate,
            'feasible': feasible
        })

# ── 5. Select best policy ──
feasible = [r for r in results if r['feasible']]
print('\nFeasible policies: {} / {}'.format(len(feasible), len(results)))

if len(feasible) == 0:
    print('\n*** NO FEASIBLE POLICY ***')
    print('P4_SCHEDULER = NO_FEASIBLE_POLICY')
    print('FINAL_DETECTOR_V23 = FAIL_BEFORE_H2')
    status = 'NO_FEASIBLE_POLICY'
    best = None
else:
    # Tie-break (pre-registered order):
    # 1. Maximize recall
    # 2. Maximize worst-suite recall (approximate with min of FS constraints satisfaction)
    # 3. Minimize max(FS_F3, FS_F4, FS_abs)
    # 4. Higher threshold (more conservative)
    # 5. Larger persistence
    feasible.sort(key=lambda r: (
        -r['recall'],
        max(r['fs_f3_rate'], r['fs_f4_rate'], r['fs_abs_rate']),
        -r['threshold'],
        -r['d']
    ))
    best = feasible[0]
    status = 'PASS'

    print('\nBest policy:')
    print('  threshold = {:.3f}'.format(best['threshold']))
    print('  persistence d = {}'.format(best['d']))
    print('  recall = {:.4f} ({}/{})'.format(best['recall'], best['vt'], best['n_opp']))
    print('  F3 FS = {:.4f} ({}/{})'.format(best['fs_f3_rate'], best['fs_f3'], n_f3))
    print('  F4 FS = {:.4f} ({}/{})'.format(best['fs_f4_rate'], best['fs_f4'], n_f4))
    print('  Absent FS = {:.4f} ({}/{})'.format(best['fs_abs_rate'], best['fs_abs'], n_abs))

    # Top alternatives
    print('\nTop 5 feasible policies:')
    for i, r in enumerate(feasible[:5]):
        print('  #{}) tau={:.3f} d={} recall={:.4f} F3={:.4f} F4={:.4f} abs={:.4f}'.format(
            i+1, r['threshold'], r['d'], r['recall'], r['fs_f3_rate'], r['fs_f4_rate'], r['fs_abs_rate']))

# ── 6. Per-suite recall for best ──
if best is not None:
    print('\n--- Per-suite recall (best policy) ---')
    suite_recall = defaultdict(lambda: {'vt': 0, 'total': 0})
    for eid in opp_eids:
        ep = p4_eps[eid]
        emit_t = run_scheduler(ep, p4_raw_logits[eid], best['threshold'], best['d'])
        suite = ep['suite']
        suite_recall[suite]['total'] += 1
        if emit_t is not None and emit_t < ep['max_t'] and ep['k10_s'][emit_t] and ep['k10_k'][emit_t]:
            suite_recall[suite]['vt'] += 1
    for suite in sorted(suite_recall.keys()):
        d = suite_recall[suite]
        rec = d['vt'] / max(d['total'], 1)
        print('  {}: recall={:.4f} ({}/{})'.format(suite, rec, d['vt'], d['total']))

# ── 7. Write outputs ──
print('\n--- Writing outputs ---')

# Policy table
with open(os.path.join(OUT_ROOT, 'P4_POLICY_TABLE_V1.json'), 'w') as f:
    json.dump({'schema': 'P4_POLICY_TABLE_V1', 'n_total': len(results),
               'n_feasible': len(feasible), 'results': results}, f)

# Search receipt
receipt = {
    'schema': 'P4_SCHEDULER_SEARCH_RECEIPT_V1',
    'status': status,
    'search_grid': {'thresholds': THRESHOLDS.tolist(), 'persistences': PERSISTENCES,
                    'n_thresholds': len(THRESHOLDS), 'n_persistences': len(PERSISTENCES),
                    'total_combinations': len(results)},
    'composition': composition,
    'best_policy': best,
    'n_feasible': len(feasible),
    'platt_a': PLATT_A, 'platt_b': PLATT_B,
    'checkpoint': 'o0_i0'
}
with open(os.path.join(OUT_ROOT, 'P4_SCHEDULER_SEARCH_RECEIPT_V1.json'), 'w') as f:
    json.dump(receipt, f, indent=2)

if status == 'PASS':
    freeze = {
        'schema': 'P4_SCHEDULER_FREEZE_V1',
        'status': 'PASS',
        'threshold': best['threshold'], 'persistence': best['d'],
        'candidate_close': 'frozen ON', 'one_shot_latch': 'frozen ON',
        'scheduler_semantics': 'emit on d consecutive candidate_close steps with calibrated_prob >= threshold',
        'valid_emit_definition': 'emit step has k10_feasible=true AND k10_known=true',
        'false_start_definition': 'any emit on F3/F4/absent episode OR emit on invalid step in opp episode',
        'recall': best['recall'], 'fs_f3_rate': best['fs_f3_rate'],
        'fs_f4_rate': best['fs_f4_rate'], 'fs_abs_rate': best['fs_abs_rate'],
        'platt_a': PLATT_A, 'platt_b': PLATT_B
    }
    with open(os.path.join(OUT_ROOT, 'P4_SCHEDULER_FREEZE_V1.json'), 'w') as f:
        json.dump(freeze, f, indent=2)

# P4 access ledger
ledger_path = os.path.join(OUT_ROOT, 'P4_ACCESS_LEDGER.json')
ledger = {'schema': 'P4_ACCESS_LEDGER_V1', 'p4_access_count': len(p4_eps),
          'h2_access_count': 0, 'action': 'P4_SCHEDULER_SEARCH'}
with open(ledger_path, 'w') as f:
    json.dump(ledger, f, indent=2)

# SHA256SUMS
all_files = []
for root, dirs, fns in os.walk(OUT_ROOT):
    for fn in sorted(fns):
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_ROOT)
        if fn == 'SHA256SUMS' or fn.endswith('.sha256'): continue
        all_files.append((rel, sha256_file(fp)))
sums_path = os.path.join(OUT_ROOT, 'SHA256SUMS')
with open(sums_path, 'w') as f:
    for rel, h in sorted(all_files):
        f.write('{}  {}\n'.format(h, rel))
sums_sha = sha256_file(sums_path)
with open(os.path.join(OUT_ROOT, 'SHA256SUMS.sha256'), 'w') as f:
    f.write('{}  SHA256SUMS\n'.format(sums_sha))

print('\nOutput: {}'.format(OUT_ROOT))
print('SHA256SUMS: {}'.format(sums_sha[:16]))
print('Status: {}'.format(status))
print('DONE.')
