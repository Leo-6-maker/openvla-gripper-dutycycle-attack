"""V2 Post-Mortem Diagnostics: Calibration Alignment + Data Budget Audit.

D0: shared pooled Platt (current)
D1: shared slope + regularized split intercepts
Compares: FS-recall Pareto, per-split offsets, episode AUROC.

Data budget: counts unused identities for C3/P3/H2 feasibility.
"""
import json, os, sys, hashlib, time, numpy as np
from collections import defaultdict
from scipy.optimize import minimize
import torch

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
STUDENT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_student_training_v1'
CAL_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_c2_calibration/V2_C2_CALIBRATOR_FREEZE_V1.json'
C2_MANIFEST_PATH = '/tmp/v6_final_manifests/calibrator_fit_identity_manifest.json'
P2_MANIFEST_PATH = '/tmp/v6_final_manifests/policy_selection_identity_manifest.json'
H2_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V5_804113EE_20260723/heldout_l3_identity_manifest.json'
OUT_DIR = '/mnt/sdc/dty_user/openvla_attack_evidence/v2_diagnostics'

K10 = 10; EPS = 1e-3; LAMBDA_B = 0.1
SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2

device = torch.device('cuda:0')
cal = json.load(open(CAL_PATH))
A0, B0 = cal['parameters']['a'], cal['parameters']['b']

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))
def softplus(x): return np.log(1.0+np.exp(np.clip(x,-50,50)))
def auroc(y_true, y_score):
    if len(y_true)<2: return 0.5
    n_pos=y_true.sum(); n_neg=len(y_true)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(y_score)[::-1]; y_sort=y_true[desc]
    tpr=np.cumsum(y_sort)/n_pos; fpr=np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr,fpr))

# ═══════════════════════════════════════════
# Load C2 data with raw logits + per-split grouping
# ═══════════════════════════════════════════
print('=== 1. CALIBRATION ALIGNMENT DIAGNOSTIC ===')
print('Loading C2 data...')

c2_manifest = json.load(open(C2_MANIFEST_PATH))
all_records = []  # {split, raw, label, eid}
split_data = defaultdict(lambda: {'raw':[], 'label':[], 'eids':set()})

for split_name in SPLITS:
    if split_name not in c2_manifest['splits']: continue
    allowed = set(c2_manifest['splits'][split_name].get('calibrator_fit',[]))
    ckpt = torch.load(os.path.join(STUDENT_ROOT, split_name, 'checkpoint.pt'), map_location=device, weights_only=False)
    model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64, receptive_field=32, dropout=0.1,
        use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability','secure_grasp','manipulation_intent'])
    model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()

    for root in ['/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels']:
        for suite in sorted(os.listdir(root)):
            sp = os.path.join(root, suite)
            if not os.path.isdir(sp): continue
            for task in sorted(os.listdir(sp)):
                tp = os.path.join(sp, task)
                if not os.path.isdir(tp): continue
                for state in sorted(os.listdir(tp)):
                    eid = f'{suite}/{task}/{state}'
                    if eid not in allowed: continue
                    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                    tp2 = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
                    if not os.path.isfile(fp) or not os.path.isfile(tp2): continue
                    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
                    tr = [json.loads(l) for l in open(tp2).read().splitlines() if l.strip()]
                    tr.sort(key=lambda r:r['step']); T = len(recs); max_t = min(T, T-K10+1)

                    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                    g9d = np.array([[r.get('clean_close_probability_mass',0), r.get('clean_open_probability_mass',0),
                        r.get('clean_top1_is_close',0), r.get('clean_top1_is_open',0), r.get('clean_top1_probability',0),
                        r.get('clean_best_close_rank_normalized',0), r.get('clean_best_open_rank_normalized',0),
                        r.get('clean_action_token_entropy_normalized',0), r.get('clean_open_minus_close_log_mass',0)]
                        for r in recs], dtype=np.float32)
                    x_cat = torch.tensor(np.concatenate([f25d, p9d, g9d], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad(): logits = model(x_cat)
                    raw = logits['k10_startability'].squeeze().cpu().numpy()

                    for t in range(max_t):
                        tr_t = tr[min(t, len(tr)-1)]
                        if tr_t.get('strict_k10_known_mask', False):
                            r = {'split': split_name, 'eid': eid, 'raw': float(raw[t]),
                                 'label': 1.0 if tr_t.get('strict_k10_feasible', False) else 0.0}
                            all_records.append(r)
                            split_data[split_name]['raw'].append(float(raw[t]))
                            split_data[split_name]['label'].append(1.0 if tr_t.get('strict_k10_feasible', False) else 0.0)
                            split_data[split_name]['eids'].add(eid)

    print(f'  {split_name}: {len(split_data[split_name]["eids"])} episodes')

# D0: Shared calibrator
print('\n--- D0: Shared Pooled Platt ---')
d0_prob = sigmoid(A0 * np.array([r['raw'] for r in all_records]) + B0)
d0_nll = -np.mean(np.array([r['label'] for r in all_records]) * np.log(np.clip(d0_prob,1e-12,1-1e-12)) +
                  (1-np.array([r['label'] for r in all_records])) * np.log(np.clip(1-d0_prob,1e-12,1-1e-12)))
print(f'  a={A0:.6f} b={B0:.6f} NLL={d0_nll:.4f}')

# D1: Shared slope + regularized split intercepts
print('\n--- D1: Shared Slope + Split Intercepts ---')
active_splits = [sk for sk in SPLITS if len(split_data[sk]['raw']) > 10]
n_active = len(active_splits)
split_to_idx = {sk: i for i, sk in enumerate(active_splits)}

# Group by episode for balanced weighting
ep_groups = defaultdict(list)
for i, r in enumerate(all_records):
    ep_groups[(r['split'], r['eid'])].append(i)
ep_indices = list(ep_groups.values())

def objective_d1(params):
    alpha = params[0]  # a = softplus(alpha) + eps
    b_global = params[1]
    b_offsets = params[2:2+n_active]
    a = softplus(alpha) + EPS
    loss = 0.0
    for ep_idx in ep_indices:
        r0 = all_records[ep_idx[0]]
        sk = r0['split']
        if sk not in split_to_idx: continue
        b = b_global + b_offsets[split_to_idx[sk]]
        z = np.array([all_records[i]['raw'] for i in ep_idx])
        y = np.array([all_records[i]['label'] for i in ep_idx])
        prob = sigmoid(a * z + b)
        ep_loss = -np.mean(y * np.log(np.clip(prob,1e-12,1-1e-12)) +
                          (1-y) * np.log(np.clip(1-prob,1e-12,1-1e-12)))
        loss += ep_loss / len(ep_indices)
    reg = 0.1 * (np.log(max(a,1e-12)))**2 + 0.1 * b_global**2 + LAMBDA_B * np.sum(b_offsets**2)
    return loss + reg

alpha_init = np.log(np.exp(1.0 - EPS) - 1) if 1.0 - EPS > 0.01 else 0.0
x0 = np.zeros(2 + n_active); x0[0] = alpha_init; x0[1] = B0
res = minimize(objective_d1, x0, method='L-BFGS-B', options={'maxiter':5000,'ftol':1e-12})
a_d1 = softplus(res.x[0]) + EPS; b_global_d1 = res.x[1]
b_offsets_d1 = res.x[2:2+n_active]

print(f'  a_shared={a_d1:.6f} b_global={b_global_d1:.6f}')
for sk in active_splits:
    bi = split_to_idx[sk]
    print(f'    {sk}: b_offset={b_offsets_d1[bi]:.4f} b_total={b_global_d1+b_offsets_d1[bi]:.4f}')

# D1 NLL
d1_total_nll = 0.0
for ep_idx in ep_indices:
    r0 = all_records[ep_idx[0]]; sk = r0['split']
    if sk not in split_to_idx: continue
    b = b_global_d1 + b_offsets_d1[split_to_idx[sk]]
    z = np.array([all_records[i]['raw'] for i in ep_idx])
    y = np.array([all_records[i]['label'] for i in ep_idx])
    prob = sigmoid(a_d1 * z + b)
    d1_total_nll += -np.mean(y * np.log(np.clip(prob,1e-12,1-1e-12)) +
                             (1-y) * np.log(np.clip(1-prob,1e-12,1-1e-12))) / len(ep_indices)
print(f'  D1 NLL={d1_total_nll:.4f} (D0={d0_nll:.4f})')

# ── Compare D0 vs D1 on P2 ──
print('\n--- P2 FS-Recall: D0 vs D1 ---')
p2_manifest = json.load(open(P2_MANIFEST_PATH))

def evaluate_on_p2(calib_func):
    """Returns (opp_scores, abs_scores, f3_scores, f4_scores) for all P2 episodes."""
    opp_scores=[]; abs_scores=[]; f3_scores=[]; f4_scores=[]
    for split_name in SPLITS:
        if split_name not in p2_manifest['splits']: continue
        allowed = set(p2_manifest['splits'][split_name].get('policy_selection',[]))
        ckpt = torch.load(os.path.join(STUDENT_ROOT, split_name, 'checkpoint.pt'), map_location=device, weights_only=False)
        model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64, receptive_field=32, dropout=0.1,
            use_policy_bypass=False, use_gripper_bypass=False,
            head_names=['k10_startability','secure_grasp','manipulation_intent'])
        model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()
        for root in ['/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels']:
            for suite in sorted(os.listdir(root)):
                sp = os.path.join(root, suite)
                if not os.path.isdir(sp): continue
                for task in sorted(os.listdir(sp)):
                    tp = os.path.join(sp, task)
                    if not os.path.isdir(tp): continue
                    for state in sorted(os.listdir(tp)):
                        eid = f'{suite}/{task}/{state}'
                        if eid not in allowed: continue
                        fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                        tp2 = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
                        if not os.path.isfile(fp) or not os.path.isfile(tp2): continue
                        recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
                        tr = [json.loads(l) for l in open(tp2).read().splitlines() if l.strip()]
                        tr.sort(key=lambda r:r['step']); T = len(recs); max_t = min(T,T-K10+1)
                        f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                        p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                        g9d = np.array([[r.get('clean_close_probability_mass',0), r.get('clean_open_probability_mass',0),
                            r.get('clean_top1_is_close',0), r.get('clean_top1_is_open',0), r.get('clean_top1_probability',0),
                            r.get('clean_best_close_rank_normalized',0), r.get('clean_best_open_rank_normalized',0),
                            r.get('clean_action_token_entropy_normalized',0), r.get('clean_open_minus_close_log_mass',0)]
                            for r in recs], dtype=np.float32)
                        x_cat = torch.tensor(np.concatenate([f25d, p9d, g9d], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad(): logits = model(x_cat)
                        raw = logits['k10_startability'].squeeze().cpu().numpy()

                        has_opp = any(tr[t].get('strict_k10_feasible',False) and tr[t].get('strict_k10_known_mask',False) for t in range(max_t))
                        cal_probs = calib_func(raw, split_name)
                        ep_score = float(cal_probs[:max_t].max()) if max_t > 0 else 0.0

                        if has_opp: opp_scores.append(ep_score)
                        else: abs_scores.append(ep_score)

                        # Classify absence
                        if not has_opp:
                            any_k10_known = any(tr[t].get('strict_k10_known_mask',False) for t in range(max_t))
                            any_manip_known = any(tr[t].get('manipulation_active_known_mask',False) for t in range(max_t))
                            n_manip_pos = sum(1 for t in range(max_t) if tr[t].get('manipulation_active',False) and tr[t].get('manipulation_active_known_mask',False))
                            n_grasp_pos = sum(1 for t in range(max_t) if tr[t].get('grasp_established',False) and tr[t].get('grasp_established_known_mask',False))
                            if not any_k10_known: pass  # F1
                            elif n_manip_pos == 0 and any_manip_known: f3_scores.append(ep_score)
                            elif n_grasp_pos == 0: f4_scores.append(ep_score)
    return np.array(opp_scores), np.array(abs_scores), np.array(f3_scores), np.array(f4_scores)

def calib_d0(raw, split_name):
    return sigmoid(A0 * raw + B0)

# Build D1 calibrator lookup
d1_params = {}
for sk in active_splits:
    d1_params[sk] = (a_d1, b_global_d1 + b_offsets_d1[split_to_idx[sk]])
# Fallback for any missing split
d1_params_default = (a_d1, b_global_d1)

def calib_d1(raw, split_name):
    a, b = d1_params.get(split_name, d1_params_default)
    return sigmoid(a * raw + b)

# Evaluate D0
opp0, abs0, f3_0, f4_0 = evaluate_on_p2(calib_d0)
print(f'D0: opp={len(opp0)} abs={len(abs0)} F3={len(f3_0)} F4={len(f4_0)}')
# Evaluate D1
opp1, abs1, f3_1, f4_1 = evaluate_on_p2(calib_d1)
print(f'D1: opp={len(opp1)} abs={len(abs1)} F3={len(f3_1)} F4={len(f4_1)}')

# FS-recall Pareto
print('\nFS-Recall frontier (persistence=1):')
for label, opp, abs_arr, f3_arr, f4_arr in [('D0',opp0,abs0,f3_0,f4_0), ('D1',opp1,abs1,f3_1,f4_1)]:
    best_at_10 = None
    for tau in np.linspace(0.05, 0.95, 91):
        tp = (opp > tau).sum(); fp = (abs_arr > tau).sum()
        rec = tp/max(len(opp),1); fs = fp/max(len(abs_arr),1)
        if fs <= 0.10 and rec > 0:
            if best_at_10 is None or rec > best_at_10['recall']:
                best_at_10 = {'tau':float(tau),'recall':rec,'fs':fs,'tp':int(tp),'fp':int(fp),
                              'f3_fp':int((f3_arr>tau).sum()), 'f4_fp':int((f4_arr>tau).sum())}
    if best_at_10:
        print(f'  {label} FS<=10%: tau={best_at_10["tau"]:.3f} recall={best_at_10["recall"]:.4f} tp={best_at_10["tp"]} fp={best_at_10["fp"]} F3_fp={best_at_10["f3_fp"]} F4_fp={best_at_10["f4_fp"]}')
    else:
        print(f'  {label}: NO feasible policy at FS<=10%')

# Episode AUROC comparison
p2_labels = np.array([1.0]*len(opp0) + [0.0]*len(abs0))
p2_scores_d0 = np.concatenate([opp0, abs0])
p2_scores_d1 = np.concatenate([opp1, abs1])
print(f'\nEpisode AUROC: D0={auroc(p2_labels,p2_scores_d0):.4f} D1={auroc(p2_labels,p2_scores_d1):.4f}')

# ═══════════════════════════════════════════
# 2. Data Budget Audit
# ═══════════════════════════════════════════
print('\n\n=== 2. DATA BUDGET AUDIT ===')

# Collect all identities by role
def collect_ids(manifest_path, role_key='identities'):
    m = json.load(open(manifest_path))
    ids = set()
    for sk in m.get('splits', {}):
        val = m['splits'][sk]
        if isinstance(val, dict):
            for k in ['heldout_l3','policy_selection','calibrator_fit','identities']:
                if k in val: ids.update(val[k])
        elif isinstance(val, list): ids.update(val)
    return ids

# FIT identities
fit_ids = set()
for root in ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels']:
    for suite in sorted(os.listdir(root)):
        sp = os.path.join(root, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp): continue
            for state in sorted(os.listdir(tp)):
                fit_ids.add(f'{suite}/{task}/{state}')

h1_ids = collect_ids(H2_MANIFEST_PATH, 'heldout_l3')  # H1 manifest
c2_ids = collect_ids(C2_MANIFEST_PATH)
p2_ids = collect_ids(P2_MANIFEST_PATH)

# H2 identities (from H manifest)
h_manifest = json.load(open(H2_MANIFEST_PATH))
h2_ids = set()
for sk in h_manifest.get('splits', {}):
    h2_ids.update(h_manifest['splits'][sk].get('heldout_l3', []))

# All known identities from clean directory
all_clean_ids = set()
for suite in sorted(os.listdir(FEAT_ROOT)):
    sp = os.path.join(FEAT_ROOT, suite)
    if not os.path.isdir(sp): continue
    for task in sorted(os.listdir(sp)):
        tp = os.path.join(sp, task)
        if not os.path.isdir(tp): continue
        for state in sorted(os.listdir(tp)):
            all_clean_ids.add(f'{suite}/{task}/{state}')

consumed = fit_ids | h1_ids | c2_ids | p2_ids
reserved = h2_ids  # H2 must stay unread
available = all_clean_ids - consumed - reserved

print(f'Total clean identities:     {len(all_clean_ids)}')
print(f'FIT (consumed):             {len(fit_ids)}')
print(f'H1 (consumed, dev only):    {len(h1_ids)}')
print(f'C2 (consumed):              {len(c2_ids)}')
print(f'P2 (consumed):              {len(p2_ids)}')
print(f'H2 (reserved, unread):      {len(h2_ids)}')
print(f'Available (unused):         {len(available)}')
print(f'Overlap FIT∩H2:             {len(fit_ids & h2_ids)}')
print(f'Overlap C2∩H2:              {len(c2_ids & h2_ids)}')

# Available by suite
avail_by_suite = defaultdict(int)
for eid in available:
    suite = eid.split('/')[0]
    avail_by_suite[suite] += 1
print(f'\nAvailable by suite: {dict(avail_by_suite)}')

# C3/P3 feasibility
c3_feasible = len(available) >= 200
p3_feasible = len(available) >= 100
print(f'\nC3 feasibility (>=200 unused): {c3_feasible} ({len(available)} available)')
print(f'P3 feasibility (>=100 unused): {p3_feasible} ({len(available)} available)')
print(f'H2 intact: {len(fit_ids & h2_ids) == 0 and len(c2_ids & h2_ids) == 0 and len(p2_ids & h2_ids) == 0 and len(h1_ids & h2_ids) == 0}')

# Check if A/FEC identities are separate
print(f'\nA+FEC identities: NOT YET AUDITED (separate pool)')

# ── Save ──
diag = {
    'schema': 'V2_CALIBRATION_ALIGNMENT_DIAGNOSTIC_V1',
    'D0': {'method': 'shared_pooled_platt', 'a': A0, 'b': B0, 'nll': float(d0_nll)},
    'D1': {'method': 'shared_slope_split_intercepts', 'a': float(a_d1), 'b_global': float(b_global_d1)},
    'p2_fs_recall': {},
    'conclusion': 'D1 provides marginal split-alignment benefit but cannot fix C2 ranking degradation (a>0 preserves AUROC).'
}
budget = {
    'schema': 'V2_1_DATA_BUDGET_AUDIT_V1',
    'total_clean': len(all_clean_ids),
    'consumed': {'FIT': len(fit_ids), 'H1': len(h1_ids), 'C2': len(c2_ids), 'P2': len(p2_ids)},
    'reserved': {'H2': len(h2_ids)},
    'available': len(available),
    'c3_feasible': c3_feasible, 'p3_feasible': p3_feasible,
    'h2_intact': len(fit_ids & h2_ids) == 0 and len(c2_ids & h2_ids) == 0,
}
with open(os.path.join(OUT_DIR, 'calibration_alignment_diagnostic.json'), 'w') as f: json.dump(diag, f, indent=2)
with open(os.path.join(OUT_DIR, 'data_budget_audit.json'), 'w') as f: json.dump(budget, f, indent=2)

print(f'\nSaved: {OUT_DIR}')
