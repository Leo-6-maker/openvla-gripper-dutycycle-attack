"""Stage I: Runtime parity + pre-parity audits (opportunity disposition, policy neighborhood).

Verifies offline batch == runtime streaming step-by-step for all 300 P4 episodes.
"""
import json, os, sys, hashlib, numpy as np, torch, torch.nn as nn
from collections import defaultdict

EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
P4_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_P4_IDENTITY_MANIFEST_V1.json'
TRAIN_DIR = EVIDENCE + '/formal_v23_student_training_v1'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
OUT_ROOT = EVIDENCE + '/runtime_parity_v1'
os.makedirs(OUT_ROOT, exist_ok=True)

PLATT_A = 0.5190011735319306
PLATT_B = 0.812702331013635
TAU = 0.855
D_PERSIST = 6
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

def calibrated_prob(raw_logit):
    return 1.0 / (1.0 + np.exp(-np.clip(PLATT_A * np.array(raw_logit) + PLATT_B, -50, 50)))

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

# ── Load data and run inference ──
print('=== RUNTIME PARITY ===')
p4_ids = json.load(open(P4_MANIFEST_PATH))['identities']
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
    proxies = compute_proxies(f25d, p9d, g9d, T)
    return {'eid': eid, 'T': T, 'max_t': max_t, 'f25d': f25d, 'p9d': p9d, 'g9d': g9d,
            'proxies': proxies, 'k10_s': k10_s, 'k10_k': k10_k, 'cc': cc,
            'has_opp': has_opp, 'absence_reason': absence_reason, 'suite': suite}

# Norm
print('Loading training norms...')
train_norm = []
for eid in o0i0_train_ids:
    ep = load_episode(eid)
    if ep is not None: train_norm.append(ep)
cat_25d = np.concatenate([e['f25d'] for e in train_norm], axis=0)
cat_p9d = np.concatenate([e['p9d'] for e in train_norm], axis=0)
cat_g9d = np.concatenate([e['g9d'] for e in train_norm], axis=0)
n25d_m = torch.tensor(cat_25d.mean(0), device=DEVICE); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=DEVICE)
np9d_m = torch.tensor(cat_p9d.mean(0), device=DEVICE); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=DEVICE)
ng9d_m = torch.tensor(cat_g9d.mean(0), device=DEVICE); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=DEVICE)

# Load P4
print('Loading P4 episodes...')
p4_eps = {}
for eid in p4_ids:
    ep = load_episode(eid)
    if ep is not None: p4_eps[eid] = ep
print('Loaded {} episodes'.format(len(p4_eps)))

# Load model
ckpt = torch.load(os.path.join(TRAIN_DIR, 'o0_i0', 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
encoder.load_state_dict(ckpt['enc']); head.load_state_dict(ckpt['head'])
encoder.eval(); head.eval()

# ── OFFLINE BATCH inference ──
print('Running offline batch inference...')
offline = {}
with torch.no_grad():
    for eid, e in sorted(p4_eps.items()):
        base = torch.cat([(torch.tensor(e['f25d'], device=DEVICE) - n25d_m) / n25d_s,
                          (torch.tensor(e['p9d'], device=DEVICE) - np9d_m) / np9d_s,
                          (torch.tensor(e['g9d'], device=DEVICE) - ng9d_m) / ng9d_s,
                          torch.tensor(e['proxies'], device=DEVICE)], dim=-1).unsqueeze(0)
        raw = head(encoder(base)).squeeze().cpu().numpy()
        offline[eid] = {'raw_logits': raw, 'cal_probs': calibrated_prob(raw)}

# ── RUNTIME STREAMING inference ──
print('Running runtime streaming inference...')
runtime = {}
diffs = {'raw': [], 'cal': []}
all_parity_ok = True

for eid, e in sorted(p4_eps.items()):
    T = e['T']
    rt_raw = np.zeros(T, dtype=np.float64)
    rt_cal = np.zeros(T, dtype=np.float64)

    for t in range(T):
        # Build input up to step t (causal, exact same as batch for steps <= t)
        base_t = torch.cat([(torch.tensor(e['f25d'][:t+1], device=DEVICE) - n25d_m) / n25d_s,
                            (torch.tensor(e['p9d'][:t+1], device=DEVICE) - np9d_m) / np9d_s,
                            (torch.tensor(e['g9d'][:t+1], device=DEVICE) - ng9d_m) / ng9d_s,
                            torch.tensor(e['proxies'][:t+1], device=DEVICE)], dim=-1).unsqueeze(0)
        with torch.no_grad():
            raw_t = head(encoder(base_t)).squeeze().cpu().numpy()
        raw_val = float(np.atleast_1d(raw_t)[-1])  # Last step = current step
        rt_raw[t] = raw_val
        rt_cal[t] = float(calibrated_prob(raw_val))

    runtime[eid] = {'raw_logits': rt_raw, 'cal_probs': rt_cal}

    # Compare
    off_raw = offline[eid]['raw_logits']; off_cal = offline[eid]['cal_probs']
    max_diff_raw = abs(off_raw - rt_raw).max()
    max_diff_cal = abs(off_cal - rt_cal).max()
    diffs['raw'].append(max_diff_raw)
    diffs['cal'].append(max_diff_cal)
    if max_diff_raw > 1e-5:
        print('  WARNING: {} raw_diff={:.2e}'.format(eid, max_diff_raw))
        all_parity_ok = False

diffs['raw'] = np.array(diffs['raw']); diffs['cal'] = np.array(diffs['cal'])
print('Offline vs runtime:')
print('  Raw logit max diff: {:.2e} (mean={:.2e})'.format(diffs['raw'].max(), diffs['raw'].mean()))
print('  Cal prob max diff:  {:.2e} (mean={:.2e})'.format(diffs['cal'].max(), diffs['cal'].mean()))

# ── SCHEDULER PARITY ──
print('\n--- Scheduler parity ---')

def offline_scheduler(ep, cal_probs):
    cons = 0
    for t in range(ep['max_t']):
        if ep['cc'][t] and cal_probs[t] >= TAU:
            cons += 1
        else:
            cons = 0
        if cons >= D_PERSIST:
            return t, True
    return None, False

def runtime_scheduler(ep, cal_probs):
    """Streaming: process step by step, latch after emit."""
    cons = 0; emitted = False
    for t in range(ep['max_t']):
        if emitted:
            # One-shot latch: ignore subsequent steps
            continue
        if ep['cc'][t] and cal_probs[t] >= TAU:
            cons += 1
        else:
            cons = 0
        if cons >= D_PERSIST:
            return t, True
    return None, False

emit_mismatches = []
for eid, e in sorted(p4_eps.items()):
    off_t, off_emit = offline_scheduler(e, offline[eid]['cal_probs'])
    rt_t, rt_emit = runtime_scheduler(e, runtime[eid]['cal_probs'])
    if off_emit != rt_emit or off_t != rt_t:
        emit_mismatches.append(eid)
        all_parity_ok = False

print('Emit mismatches: {} / {}'.format(len(emit_mismatches), len(p4_eps)))
if emit_mismatches:
    for eid in emit_mismatches[:10]:
        print('  {}'.format(eid))

scheduler_parity = len(emit_mismatches) == 0
print('Scheduler parity: {}'.format('PASS' if scheduler_parity else 'FAIL'))

# ── PRE-AUDIT 1: Opportunity disposition ──
print('\n' + '='*60)
print('PRE-AUDIT 1: OPPORTUNITY FIRST-EMIT DISPOSITION')
print('='*60)

opp_eids = [eid for eid, e in p4_eps.items() if e['has_opp']]
f3_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F3_NO_MANIPULATION']
f4_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F4_NO_STABLE_GRASP']
f1_eids = [eid for eid, e in p4_eps.items() if e['absence_reason'] == 'F1_STRUCTURAL_ZERO']
all_abs = [eid for eid in p4_eps if not p4_eps[eid]['has_opp']]

disposition = {'valid': [], 'mistimed_feasible_but_not_first_corridor': [],
               'mistimed_infeasible': [], 'mistimed_unknown': [],
               'mistimed_terminal': [], 'no_emit': []}

for eid in opp_eids:
    ep = p4_eps[eid]; cal = offline[eid]['cal_probs']
    emit_t, emitted = offline_scheduler(ep, cal)
    if not emitted:
        disposition['no_emit'].append(eid)
        continue
    if emit_t >= ep['max_t']:
        disposition['mistimed_terminal'].append(eid)
    elif ep['k10_s'][emit_t] and ep['k10_k'][emit_t]:
        disposition['valid'].append(eid)
    elif ep['k10_k'][emit_t]:
        disposition['mistimed_infeasible'].append(eid)
    else:
        disposition['mistimed_unknown'].append(eid)

for cat, eids in disposition.items():
    print('  {}: {} ({:.1f}%)'.format(cat, len(eids), 100*len(eids)/max(len(opp_eids),1)))
print('  Total opp: {}  (valid {} + invalid {} + no_emit {})'.format(
    len(opp_eids), len(disposition['valid']),
    sum(len(v) for k,v in disposition.items() if k != 'valid' and k != 'no_emit'),
    len(disposition['no_emit'])))

# ── PRE-AUDIT 2: Absent FP identities ──
print('\n' + '='*60)
print('PRE-AUDIT 2: ABSENT FALSE-START IDENTITIES')
print('='*60)

absent_fps = {'F3': [], 'F4': [], 'F1': [], 'other': []}
for cat, eid_list in [('F3', f3_eids), ('F4', f4_eids), ('F1', f1_eids)]:
    for eid in eid_list:
        ep = p4_eps[eid]; cal = offline[eid]['cal_probs']
        emit_t, emitted = offline_scheduler(ep, cal)
        if emitted:
            absent_fps[cat].append((eid, emit_t))

for cat, fps in absent_fps.items():
    print('  {}: {} false starts'.format(cat, len(fps)))
    for eid, emit_t in fps[:5]:
        ep = p4_eps[eid]
        print('    {} emit_t={} T={} cc_at_emit={} cal_at_emit={:.4f}'.format(
            eid, emit_t, ep['T'], ep['cc'][emit_t] if emit_t < len(ep['cc']) else '?',
            offline[eid]['cal_probs'][emit_t] if emit_t < len(offline[eid]['cal_probs']) else float('nan')))
total_fp = sum(len(v) for v in absent_fps.values())
print('  Total absent FS: {} (expected 6 from P4 search)'.format(total_fp))

# ── PRE-AUDIT 3: Policy neighborhood ──
print('\n' + '='*60)
print('PRE-AUDIT 3: POLICY NEIGHBORHOOD (report only)')
print('='*60)

neighbors = []
for tau_delta in [-0.02, -0.01, 0.0, 0.01, 0.02]:
    for d_delta in [-1, 0, 1]:
        tau_n = round(TAU + tau_delta, 3); d_n = D_PERSIST + d_delta
        if tau_n <= 0 or d_n < 1: continue
        vt = 0; fs3 = 0; fs4 = 0; fs_abs = 0
        for eid in opp_eids:
            cal = offline[eid]['cal_probs']; ep = p4_eps[eid]
            cons = 0
            for t in range(ep['max_t']):
                if ep['cc'][t] and cal[t] >= tau_n: cons += 1
                else: cons = 0
                if cons >= d_n:
                    if t < ep['max_t'] and ep['k10_s'][t] and ep['k10_k'][t]: vt += 1
                    break
        for eid in f3_eids:
            cal = offline[eid]['cal_probs']; ep = p4_eps[eid]
            cons = 0
            for t in range(ep['max_t']):
                if ep['cc'][t] and cal[t] >= tau_n: cons += 1
                else: cons = 0
                if cons >= d_n: fs3 += 1; break
        for eid in f4_eids:
            cal = offline[eid]['cal_probs']; ep = p4_eps[eid]
            cons = 0
            for t in range(ep['max_t']):
                if ep['cc'][t] and cal[t] >= tau_n: cons += 1
                else: cons = 0
                if cons >= d_n: fs4 += 1; break
        for eid in all_abs:
            cal = offline[eid]['cal_probs']; ep = p4_eps[eid]
            cons = 0
            for t in range(ep['max_t']):
                if ep['cc'][t] and cal[t] >= tau_n: cons += 1
                else: cons = 0
                if cons >= d_n: fs_abs += 1; break
        rec = vt / max(len(opp_eids), 1)
        fs3_r = fs3 / max(len(f3_eids), 1); fs4_r = fs4 / max(len(f4_eids), 1)
        fs_abs_r = fs_abs / max(len(all_abs), 1)
        feasible_n = fs3_r <= 0.10 and fs4_r <= 0.10 and fs_abs_r <= 0.10 and vt > 0
        is_frozen = abs(tau_n - TAU) < 0.001 and d_n == D_PERSIST
        neighbors.append({'tau': tau_n, 'd': d_n, 'recall': rec, 'vt': vt,
                          'fs_f3': fs3, 'fs_f3_rate': fs3_r, 'fs_f4': fs4, 'fs_f4_rate': fs4_r,
                          'fs_abs': fs_abs, 'fs_abs_rate': fs_abs_r, 'feasible': feasible_n,
                          'is_frozen_policy': is_frozen})

for n in neighbors:
    marker = ' *** FROZEN ***' if n['is_frozen_policy'] else ''
    print('  tau={:.3f} d={} recall={:.4f} F3={}/{}={:.3f} F4={}/{}={:.3f} abs={}/{}={:.3f} {}{}'.format(
        n['tau'], n['d'], n['recall'], n['fs_f3'], len(f3_eids), n['fs_f3_rate'],
        n['fs_f4'], len(f4_eids), n['fs_f4_rate'], n['fs_abs'], len(all_abs), n['fs_abs_rate'],
        'FEASIBLE' if n['feasible'] else 'FAIL', marker))

# ── Cold-start boundary tests ──
print('\n--- Cold-start / edge case tests ---')
# Test all episodes at various T cutoffs
cold_start_ok = True
test_eids = list(sorted(p4_eps.keys()))[:20]  # First 20 for speed
for eid in test_eids:
    ep = p4_eps[eid]; T = ep['T']
    for cutoff in [1, 2, 32, 64, 128, min(T, 200)]:
        if cutoff > T: continue
        base_full = torch.cat([(torch.tensor(ep['f25d'], device=DEVICE) - n25d_m) / n25d_s,
                               (torch.tensor(ep['p9d'], device=DEVICE) - np9d_m) / np9d_s,
                               (torch.tensor(ep['g9d'], device=DEVICE) - ng9d_m) / ng9d_s,
                               torch.tensor(ep['proxies'], device=DEVICE)], dim=-1).unsqueeze(0)
        base_cut = torch.cat([(torch.tensor(ep['f25d'][:cutoff], device=DEVICE) - n25d_m) / n25d_s,
                              (torch.tensor(ep['p9d'][:cutoff], device=DEVICE) - np9d_m) / np9d_s,
                              (torch.tensor(ep['g9d'][:cutoff], device=DEVICE) - ng9d_m) / ng9d_s,
                              torch.tensor(ep['proxies'][:cutoff], device=DEVICE)], dim=-1).unsqueeze(0)
        with torch.no_grad():
            full_out = head(encoder(base_full)).squeeze().cpu().numpy()
            cut_out = head(encoder(base_cut)).squeeze().cpu().numpy()
        full_out = np.atleast_1d(full_out); cut_out = np.atleast_1d(cut_out)
        for t in range(cutoff):
            diff = abs(float(full_out[t]) - float(cut_out[t]))
            if diff > 1e-5:
                print('  FAIL: {} cutoff={} step={} diff={:.2e}'.format(eid, cutoff, t, diff))
                cold_start_ok = False
                break
        if not cold_start_ok: break
    if not cold_start_ok: break

print('Cold-start parity: {}'.format('PASS' if cold_start_ok else 'FAIL'))
all_parity_ok = all_parity_ok and scheduler_parity and cold_start_ok

# ── Write outputs ──
print('\n--- Writing outputs ---')

# Disposition
disp_out = {'schema': 'P4_FIRST_EMIT_DISPOSITION_AUDIT_V1',
            'n_opp': len(opp_eids), 'n_valid': len(disposition['valid']),
            'n_mistimed_infeasible': len(disposition['mistimed_infeasible']),
            'n_mistimed_unknown': len(disposition['mistimed_unknown']),
            'n_mistimed_terminal': len(disposition['mistimed_terminal']),
            'n_no_emit': len(disposition['no_emit']),
            'disposition': {cat: eids for cat, eids in disposition.items()}}
with open(os.path.join(OUT_ROOT, 'P4_FIRST_EMIT_DISPOSITION_AUDIT_V1.json'), 'w') as f:
    json.dump(disp_out, f)

# Absent FP
fp_out = {'schema': 'P4_ABSENT_FP_IDENTITIES_V1', 'total_fp': total_fp,
          'f3_fp': [{'eid': eid, 'emit_t': t} for eid, t in absent_fps['F3']],
          'f4_fp': [{'eid': eid, 'emit_t': t} for eid, t in absent_fps['F4']]}
with open(os.path.join(OUT_ROOT, 'P4_ABSENT_FP_IDENTITIES_V1.json'), 'w') as f:
    json.dump(fp_out, f)

# Neighborhood
with open(os.path.join(OUT_ROOT, 'P4_POLICY_NEIGHBORHOOD_AUDIT_V1.json'), 'w') as f:
    json.dump({'schema': 'P4_POLICY_NEIGHBORHOOD_AUDIT_V1', 'frozen_tau': TAU, 'frozen_d': D_PERSIST,
               'neighbors': neighbors}, f)

# Runtime contract
contract = {'schema': 'V23_RUNTIME_CONTRACT_V1',
            'feature_order': '43D + policy_9d + gripper_9d + response_proxies_8d = 51D',
            'normalization': 'z-score from o0_i0 training data',
            'platt_a': PLATT_A, 'platt_b': PLATT_B,
            'threshold': TAU, 'persistence': D_PERSIST,
            'candidate_close': 'ON (frozen from factorized teacher)',
            'one_shot_latch': 'ON',
            'comparison': '>=', 'dtype': 'float32 inference, float64 comparison',
            'sigmoid': 'numerically stable with clip(-50, 50)'}
with open(os.path.join(OUT_ROOT, 'V23_RUNTIME_CONTRACT_V1.json'), 'w') as f:
    json.dump(contract, f, indent=2)

# Parity receipt
parity = {'schema': 'OFFLINE_RUNTIME_PARITY_RECEIPT_V1',
          'status': 'PASS' if all_parity_ok else 'FAIL',
          'identity_coverage': '{}/{}'.format(len(p4_eps), len(p4_ids)),
          'raw_logit_max_diff': float(diffs['raw'].max()),
          'cal_prob_max_diff': float(diffs['cal'].max()),
          'scheduler_parity': bool(scheduler_parity),
          'emit_mismatches': len(emit_mismatches),
          'cold_start_parity': bool(cold_start_ok),
          'h2_access': 0}
with open(os.path.join(OUT_ROOT, 'OFFLINE_RUNTIME_PARITY_RECEIPT_V1.json'), 'w') as f:
    json.dump(parity, f, indent=2)

# Trace manifest
with open(os.path.join(OUT_ROOT, 'RUNTIME_PARITY_TRACE_MANIFEST_V1.json'), 'w') as f:
    json.dump({'schema': 'RUNTIME_PARITY_TRACE_MANIFEST_V1',
               'n_episodes': len(p4_eps), 'parity_passed': bool(all_parity_ok),
               'platt_a': PLATT_A, 'platt_b': PLATT_B}, f)

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

print('\n' + '='*60)
print('RUNTIME PARITY: {}'.format('PASS' if all_parity_ok else 'FAIL'))
print('Raw max diff: {:.2e}'.format(diffs['raw'].max()))
print('Cal max diff:  {:.2e}'.format(diffs['cal'].max()))
print('Scheduler: {}  Cold-start: {}'.format(
    'PASS' if scheduler_parity else 'FAIL', 'PASS' if cold_start_ok else 'FAIL'))
print('SHA256SUMS: {}'.format(sums_sha[:16]))
print('DONE.')
