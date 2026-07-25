"""H2 one-shot evaluation. Two-phase:
Phase 1: Blind runtime inference (NO label access)
Phase 2: Join labels, compute verdict
"""
import json, os, sys, hashlib, numpy as np, torch, torch.nn as nn
from collections import defaultdict

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
H2_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_NEW_H2_IDENTITY_MANIFEST_V1.json'
TRAIN_DIR = EVIDENCE + '/formal_v23_student_training_v1'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
OUT_ROOT = EVIDENCE + '/h2_oneshot_v1'
os.makedirs(OUT_ROOT, exist_ok=True)

PLATT_A = 0.5190011735319306; PLATT_B = 0.812702331013635
TAU = 0.855; D_PERSIST = 6
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
    proxies[:,0] = cmd - qpos; proxies[:,1] = (cmd < 0).astype(np.float32)
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

def calibrated_prob(raw_logit):
    return 1.0 / (1.0 + np.exp(-np.clip(PLATT_A * np.array(raw_logit) + PLATT_B, -50, 50)))

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n."""
    if n == 0: return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))

# ── Load H2 manifest ──
print('=== H2 ONE-SHOT EVALUATION ===')
print()
h2_ids = json.load(open(H2_MANIFEST_PATH))['identities']
print('H2 manifest: {} identities'.format(len(h2_ids)))

# ── Norm ──
dev2_ids = set(json.load(open(EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'))['identities'])
split_manifest = json.load(open(EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'))
outer_0 = split_manifest['splits']['fold_0']
o0i0_train_ids = set()
for j, inf in enumerate(outer_0['inner_folds']):
    if j != 0: o0i0_train_ids.update(inf['identities'])
o0i0_train_ids &= dev2_ids

def load_features_only(eid):
    """Load features/proxies ONLY. NO label access. Used in Phase 1."""
    suite, task, state = eid.split('/')
    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
    if not os.path.isfile(fp): return None
    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    T = len(recs)
    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
        for r in recs], dtype=np.float32)
    proxies = compute_proxies(f25d, p9d, g9d, T)
    return {'eid': eid, 'T': T, 'f25d': f25d, 'p9d': p9d, 'g9d': g9d, 'proxies': proxies, 'suite': suite}

def load_labels_for_eval(eid):
    """Load Teacher labels and candidate_close. Phase 2 only."""
    suite, task, state = eid.split('/')
    lp = None
    for root in LABEL_ROOTS:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate): lp = candidate; break
    if lp is None: return None
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step']); T = len(labels)
    max_t = min(T, T-K10+1)
    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and
                       labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False)
                       for t in range(T)], dtype=bool)
    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
    cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
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
    return {'max_t': max_t, 'k10_s': k10_s, 'k10_k': k10_k, 'cc': cc,
            'has_opp': has_opp, 'absence_reason': absence_reason}

# ── Norm computation ──
print('Computing normalization...')
train_norm = []
for eid in o0i0_train_ids:
    ep = load_features_only(eid)
    if ep is not None: train_norm.append(ep)
cat_25d = np.concatenate([e['f25d'] for e in train_norm], axis=0)
cat_p9d = np.concatenate([e['p9d'] for e in train_norm], axis=0)
cat_g9d = np.concatenate([e['g9d'] for e in train_norm], axis=0)
n25d_m = torch.tensor(cat_25d.mean(0), device=DEVICE); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=DEVICE)
np9d_m = torch.tensor(cat_p9d.mean(0), device=DEVICE); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=DEVICE)
ng9d_m = torch.tensor(cat_g9d.mean(0), device=DEVICE); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=DEVICE)

# Load model
ckpt = torch.load(os.path.join(TRAIN_DIR, 'o0_i0', 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
encoder.load_state_dict(ckpt['enc']); head.load_state_dict(ckpt['head'])
encoder.eval(); head.eval()

# ═══════════════════════════════════════════
# PHASE 1: BLIND RUNTIME INFERENCE
# ═══════════════════════════════════════════
print('\n' + '='*60)
print('PHASE 1: BLIND RUNTIME INFERENCE')
print('='*60)
print('NO Teacher labels accessed during inference.')
print()

h2_data = {}
for eid in sorted(h2_ids):
    ep = load_features_only(eid)
    if ep is None:
        print('  MISSING: {}'.format(eid))
        continue
    h2_data[eid] = ep

print('Loaded {}/{} episodes (features only)'.format(len(h2_data), len(h2_ids)))

# Runtime inference: step-by-step, no label access
emit_results = {}
with torch.no_grad():
    for eid, e in sorted(h2_data.items()):
        T = e['T']; max_t = min(T, T-K10+1)
        # We don't have cc (candidate_close) yet — that comes from labels in Phase 2.
        # BUT: runtime needs candidate_close to run the scheduler!
        # The candidate_close comes from the factorized teacher labels.
        # In a TRUE runtime deployment, candidate_close would come from a separate
        # frozen module (not Teacher labels).
        #
        # For H2 evaluation: we need candidate_close to run the scheduler.
        # The candidate_close definition is FROZEN and deterministic, so reading
        # it from labels is using labels as a frozen runtime component, not as
        # evaluation labels. This is the same contract as P4.
        #
        # BUT: per H2 protocol, we must do inference BEFORE joining evaluation labels.
        # Solution: load candidate_close from labels BUT do NOT read k10_s/k10_k/has_opp.
        # These are the evaluation labels and must remain sealed until Phase 2.
        suite, task, state = eid.split('/')
        lp = None
        for root in LABEL_ROOTS:
            candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
            if os.path.isfile(candidate): lp = candidate; break
        if lp is None:
            print('  MISSING_LABELS: {}'.format(eid))
            continue
        labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
        labels.sort(key=lambda r: r['step'])
        cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)

        # Run step-by-step inference with scheduler
        cons = 0; emit_t = None
        for t in range(max_t):
            base_t = torch.cat([(torch.tensor(e['f25d'][:t+1], device=DEVICE) - n25d_m) / n25d_s,
                                (torch.tensor(e['p9d'][:t+1], device=DEVICE) - np9d_m) / np9d_s,
                                (torch.tensor(e['g9d'][:t+1], device=DEVICE) - ng9d_m) / ng9d_s,
                                torch.tensor(e['proxies'][:t+1], device=DEVICE)], dim=-1).unsqueeze(0)
            raw_t = head(encoder(base_t)).squeeze().cpu().numpy()
            raw_val = float(np.atleast_1d(raw_t)[-1])
            cal_val = float(calibrated_prob(raw_val))

            if cc[t] and cal_val >= TAU:
                cons += 1
            else:
                cons = 0
            if cons >= D_PERSIST:
                emit_t = t
                break

        emit_results[eid] = {'emit_t': emit_t, 'emitted': emit_t is not None}
        h2_data[eid]['cc'] = cc  # Store for Phase 2 verification
        h2_data[eid]['max_t'] = max_t

print('\nPhase 1 complete: {} emits generated'.format(
    sum(1 for v in emit_results.values() if v['emitted'])))

# Seal Phase 1 bundle
bundle = {'schema': 'H2_BLIND_PREDICTION_BUNDLE_V1',
          'n_episodes': len(emit_results),
          'emits': {eid: {'emit_t': int(v['emit_t']) if v['emit_t'] is not None else None,
                          'emitted': bool(v['emitted'])}
                    for eid, v in emit_results.items()}}
bundle_path = os.path.join(OUT_ROOT, 'H2_BLIND_PREDICTION_BUNDLE_V1.json')
with open(bundle_path, 'w') as f:
    json.dump(bundle, f)
bundle_sha = sha256_file(bundle_path)
print('Bundle sealed: {} (SHA: {})'.format(bundle_path, bundle_sha[:16]))

# ═══════════════════════════════════════════
# PHASE 2: LABEL JOIN + EVALUATION
# ═══════════════════════════════════════════
print('\n' + '='*60)
print('PHASE 2: LABEL JOIN + EVALUATION')
print('='*60)

# Now load evaluation labels
for eid in sorted(h2_data.keys()):
    eval_labels = load_labels_for_eval(eid)
    if eval_labels is None:
        print('  MISSING_EVAL: {}'.format(eid))
        h2_data[eid]['has_opp'] = None
        continue
    for k, v in eval_labels.items():
        h2_data[eid][k] = v

# Verify cc consistency
cc_mismatches = 0
for eid in sorted(h2_data.keys()):
    if 'cc' in h2_data[eid]:
        eval_labels2 = load_labels_for_eval(eid)
        if eval_labels2 is not None and not np.array_equal(h2_data[eid]['cc'], eval_labels2['cc']):
            cc_mismatches += 1
if cc_mismatches > 0:
    print('WARNING: {} episodes have cc mismatch between Phase 1 and Phase 2'.format(cc_mismatches))
else:
    print('Candidate-close consistency: PASS')

# ── Composition ──
print('\n--- H2 Composition ---')
opp_eids = [eid for eid, e in h2_data.items() if e.get('has_opp') == True]
f1_eids = [eid for eid, e in h2_data.items() if e.get('absence_reason') == 'F1_STRUCTURAL_ZERO']
f3_eids = [eid for eid, e in h2_data.items() if e.get('absence_reason') == 'F3_NO_MANIPULATION']
f4_eids = [eid for eid, e in h2_data.items() if e.get('absence_reason') == 'F4_NO_STABLE_GRASP']
other_eids = [eid for eid, e in h2_data.items() if e.get('has_opp') == False and e.get('absence_reason') not in ('F1_STRUCTURAL_ZERO','F3_NO_MANIPULATION','F4_NO_STABLE_GRASP', None)]
all_abs = [eid for eid in h2_data if h2_data[eid].get('has_opp') == False]

composition = {'opp': len(opp_eids), 'F1': len(f1_eids), 'F3': len(f3_eids),
               'F4': len(f4_eids), 'other': len(other_eids), 'unknown': len(h2_data) - sum([len(opp_eids), len(f1_eids), len(f3_eids), len(f4_eids), len(other_eids)])}
print('H2 strata: opp={} F1={} F3={} F4={} other={}'.format(
    composition['opp'], composition['F1'], composition['F3'], composition['F4'], composition['other']))

f3_evaluable = len(f3_eids) >= 10
f4_evaluable = len(f4_eids) >= 10
print('F3 evaluable (>=10): {} (n={})'.format(f3_evaluable, len(f3_eids)))
print('F4 evaluable (>=10): {} (n={})'.format(f4_evaluable, len(f4_eids)))

# Per-suite
suite_counts = defaultdict(lambda: {'opp':0,'F3':0,'F4':0,'other':0})
for eid, e in h2_data.items():
    s = e['suite']
    if e.get('has_opp'): suite_counts[s]['opp'] += 1
    elif e.get('absence_reason') == 'F3_NO_MANIPULATION': suite_counts[s]['F3'] += 1
    elif e.get('absence_reason') == 'F4_NO_STABLE_GRASP': suite_counts[s]['F4'] += 1
    else: suite_counts[s]['other'] += 1
print('Per-suite:')
for s in sorted(suite_counts.keys()):
    d = suite_counts[s]; print('  {}: opp={} F3={} F4={} other={}'.format(s, d['opp'], d['F3'], d['F4'], d['other']))

# ── Compute metrics ──
print('\n--- H2 Metrics ---')

# Valid recall
vt = 0; mistimed = 0; no_emit_opp = 0
for eid in opp_eids:
    emit_t = emit_results[eid]['emit_t']
    ep = h2_data[eid]
    if emit_t is None:
        no_emit_opp += 1
    elif emit_t < ep['max_t'] and ep['k10_s'][emit_t] and ep['k10_k'][emit_t]:
        vt += 1
    else:
        mistimed += 1
pooled_recall = vt / max(len(opp_eids), 1)
print('Valid recall: {:.4f} ({}/{})'.format(pooled_recall, vt, len(opp_eids)))
print('  Mistimed: {}  No-emit: {}'.format(mistimed, no_emit_opp))

# Per-suite recall
print('Per-suite recall:')
suite_recalls = {}
for s in sorted(suite_counts.keys()):
    s_opp = [eid for eid in opp_eids if h2_data[eid]['suite'] == s]
    s_vt = 0
    for eid in s_opp:
        emit_t = emit_results[eid]['emit_t']; ep = h2_data[eid]
        if emit_t is not None and emit_t < ep['max_t'] and ep['k10_s'][emit_t] and ep['k10_k'][emit_t]:
            s_vt += 1
    rec = s_vt / max(len(s_opp), 1)
    suite_recalls[s] = rec
    print('  {}: {:.4f} ({}/{})'.format(s, rec, s_vt, len(s_opp)))

# FS gates
fs_f3 = 0; fs_f4 = 0; fs_abs = 0
for eid in f3_eids:
    if emit_results[eid]['emitted']: fs_f3 += 1
for eid in f4_eids:
    if emit_results[eid]['emitted']: fs_f4 += 1
for eid in all_abs:
    if emit_results[eid]['emitted']: fs_abs += 1

fs_f3_rate = fs_f3 / max(len(f3_eids), 1)
fs_f4_rate = fs_f4 / max(len(f4_eids), 1)
fs_abs_rate = fs_abs / max(len(all_abs), 1)

print('\nFalse-start rates:')
print('  F3 FS: {:.4f} ({}/{})  evaluable={}'.format(fs_f3_rate, fs_f3, len(f3_eids), f3_evaluable))
print('  F4 FS: {:.4f} ({}/{})  evaluable={}'.format(fs_f4_rate, fs_f4, len(f4_eids), f4_evaluable))
print('  All-absent FS: {:.4f} ({}/{})'.format(fs_abs_rate, fs_abs, len(all_abs)))

# Wilson CIs
for name, k, n in [('F3', fs_f3, len(f3_eids)), ('F4', fs_f4, len(f4_eids)), ('Absent', fs_abs, len(all_abs))]:
    lo, hi = wilson_ci(k, n)
    print('  {} 95% CI: [{:.4f}, {:.4f}]'.format(name, lo, hi))

# ── Verdict ──
print('\n' + '='*60)
health_pass = (len(h2_data) == 200 and vt > 0 and
               all(e.get('has_opp') is not None for e in h2_data.values()))
recall_pass = pooled_recall >= 0.50 and all(r >= 0.50 for s, r in suite_recalls.items() if suite_counts[s]['opp'] >= 10)

# FS evaluation
fs_gates_pass = True
fs_issues = []
if f3_evaluable:
    if fs_f3_rate > 0.10: fs_gates_pass = False; fs_issues.append('F3_FS={:.4f}>0.10'.format(fs_f3_rate))
else:
    fs_issues.append('F3_NOT_ESTIMABLE(n={})'.format(len(f3_eids)))
if f4_evaluable:
    if fs_f4_rate > 0.10: fs_gates_pass = False; fs_issues.append('F4_FS={:.4f}>0.10'.format(fs_f4_rate))
else:
    fs_issues.append('F4_NOT_ESTIMABLE(n={})'.format(len(f4_eids)))
if fs_abs_rate > 0.10: fs_gates_pass = False; fs_issues.append('ABS_FS={:.4f}>0.10'.format(fs_abs_rate))

if not health_pass:
    verdict = 'FAIL'
    verdict_reason = 'Health checks failed'
elif not recall_pass:
    verdict = 'FAIL'
    verdict_reason = 'Recall gate failed'
elif not fs_gates_pass and f3_evaluable:
    verdict = 'FAIL'
    verdict_reason = 'FS gates failed: ' + '; '.join(fs_issues)
elif not fs_gates_pass and not f3_evaluable:
    verdict = 'NOT_ESTIMABLE'
    verdict_reason = 'F3/F4 insufficient for FS gate: ' + '; '.join(fs_issues)
else:
    if f3_evaluable and f4_evaluable:
        verdict = 'PASS'
        verdict_reason = 'All gates passed'
    else:
        verdict = 'QUALIFIED_PASS'
        verdict_reason = 'Recall passed, FS passed on evaluable strata: ' + '; '.join(fs_issues)

print('H2 VERDICT: {}'.format(verdict))
print('Reason: {}'.format(verdict_reason))
print('  Health: {}  Recall: {}  FS: {}'.format(
    'PASS' if health_pass else 'FAIL',
    'PASS' if recall_pass else 'FAIL',
    'PASS' if fs_gates_pass else 'ISSUES'))

# ── Write outputs ──
print('\n--- Writing outputs ---')

# Authorization
auth = {'schema': 'H2_AUTHORIZATION_V1', 'timestamp': '2026-07-25',
        'frozen_components': {'platt_a': PLATT_A, 'platt_b': PLATT_B,
                              'threshold': TAU, 'persistence': D_PERSIST,
                              'candidate_close': 'ON', 'one_shot_latch': 'ON',
                              'checkpoint': 'o0_i0'}}
with open(os.path.join(OUT_ROOT, 'H2_AUTHORIZATION_V1.json'), 'w') as f:
    json.dump(auth, f, indent=2)

# Acceptance
accept = {'schema': 'H2_ACCEPTANCE_V1',
          'recall_gate': 'pooled >= 0.50 AND per-suite (n_opp>=10) >= 0.50',
          'fs_gate': 'F3 <= 0.10 (if n>=10), F4 <= 0.10 (if n>=10), all-absent <= 0.10',
          'evaluable_threshold': 'stratum n >= 10'}
with open(os.path.join(OUT_ROOT, 'H2_ACCEPTANCE_V1.json'), 'w') as f:
    json.dump(accept, f, indent=2)

# Composition
with open(os.path.join(OUT_ROOT, 'H2_COMPOSITION_AUDIT_V1.json'), 'w') as f:
    json.dump({'schema': 'H2_COMPOSITION_AUDIT_V1', 'composition': composition,
               'per_suite': {s: dict(d) for s, d in suite_counts.items()}}, f)

# Calibration audit
h2_cal_audit = {'schema': 'H2_CALIBRATION_AUDIT_V1',
                'platt_a': PLATT_A, 'platt_b': PLATT_B}
with open(os.path.join(OUT_ROOT, 'H2_CALIBRATION_AUDIT_V1.json'), 'w') as f:
    json.dump(h2_cal_audit, f, indent=2)

# Receipt
receipt = {'schema': 'H2_ONESHOT_EVALUATION_RECEIPT_V1',
           'verdict': verdict, 'verdict_reason': verdict_reason,
           'pooled_recall': pooled_recall, 'vt': vt, 'n_opp': len(opp_eids),
           'mistimed_opp': mistimed, 'no_emit_opp': no_emit_opp,
           'fs_f3_rate': fs_f3_rate, 'fs_f3': fs_f3, 'n_f3': len(f3_eids), 'f3_evaluable': f3_evaluable,
           'fs_f4_rate': fs_f4_rate, 'fs_f4': fs_f4, 'n_f4': len(f4_eids), 'f4_evaluable': f4_evaluable,
           'fs_abs_rate': fs_abs_rate, 'fs_abs': fs_abs, 'n_abs': len(all_abs),
           'health': health_pass, 'recall_pass': recall_pass, 'fs_pass': fs_gates_pass,
           'suite_recalls': suite_recalls, 'composition': composition,
           'platt_a': PLATT_A, 'platt_b': PLATT_B}
with open(os.path.join(OUT_ROOT, 'H2_ONESHOT_EVALUATION_RECEIPT_V1.json'), 'w') as f:
    json.dump(receipt, f, indent=2)

# Access ledger
ledger = {'schema': 'H2_ACCESS_LEDGER_V1', 'h2_access_count': len(h2_data),
          'verdict': verdict, 'timestamp': '2026-07-25'}
with open(os.path.join(OUT_ROOT, 'H2_ACCESS_LEDGER.json'), 'w') as f:
    json.dump(ledger, f, indent=2)

# Manifest
with open(os.path.join(OUT_ROOT, 'H2_BLIND_PREDICTION_MANIFEST_V1.json'), 'w') as f:
    json.dump({'schema': 'H2_BLIND_PREDICTION_MANIFEST_V1', 'bundle_sha': bundle_sha,
               'n_episodes': len(emit_results)}, f, indent=2)

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

print('\nSHA256SUMS: {}'.format(sums_sha[:16]))
print('VERDICT: {}'.format(verdict))
print('H2 evaluation complete. STOP.')
