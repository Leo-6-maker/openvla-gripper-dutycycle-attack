"""Stage F: C4 raw ranking inference.

Re-computes normalization from o0_i0 training data (norms not stored in checkpoint).
Pre-registered mapping: all C4 identities use o0_i0 checkpoint.
"""
import json, os, sys, hashlib, numpy as np, torch, torch.nn as nn
from collections import defaultdict

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
FEAT_ROOT = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/clean'
C4_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_C4_IDENTITY_MANIFEST_V1.json'
DEV2_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'
SPLIT_MANIFEST_PATH = EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'
TRAIN_DIR = EVIDENCE + '/formal_v23_student_training_v1'
LABEL_ROOTS = [
    '/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels',
    '/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    EVIDENCE + '/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
]
OUT_ROOT = EVIDENCE + '/c4_raw_ranking_v1'
os.makedirs(OUT_ROOT, exist_ok=True)
PRED_DIR = os.path.join(OUT_ROOT, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

K10 = 10; HIDDEN = 64
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

def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()

def auroc(yt, ys):
    yt = np.array(yt); ys = np.array(ys)
    if len(yt) < 2 or yt.sum() == 0 or (1-yt).sum() == 0: return float('nan')
    n_pos = yt.sum(); n_neg = len(yt) - n_pos
    desc = np.argsort(ys)[::-1]; ysort = yt[desc]
    tpr = np.cumsum(ysort) / n_pos
    fpr = np.cumsum(1 - ysort) / n_neg
    return float(np.trapz(tpr, fpr))

def auprc(yt, ys):
    yt = np.array(yt); ys = np.array(ys)
    if len(yt) < 2 or yt.sum() == 0: return float('nan')
    n_pos = yt.sum()
    desc = np.argsort(ys)[::-1]; ysort = yt[desc]
    prec = np.cumsum(ysort) / np.arange(1, len(ysort) + 1)
    rec = np.cumsum(ysort) / n_pos
    return float(np.trapz(prec, rec))

def load_episode(eid, feat_root, label_roots):
    suite, task, state = eid.split('/')
    fp = os.path.join(feat_root, suite, task, state, 'step_records.jsonl')
    lp = None
    for root in label_roots:
        candidate = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
        if os.path.isfile(candidate):
            lp = candidate; break
    if not os.path.isfile(fp) or lp is None: return None
    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
    labels.sort(key=lambda r: r['step'])
    T = len(recs); max_t = min(T, T-K10+1)
    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
        for r in recs], dtype=np.float32)
    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
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

# ── 1. Load o0_i0 training IDs to compute normalization ──
print('=== C4 RAW RANKING INFERENCE ===')
print()
print('--- Computing normalization from o0_i0 training data ---')
dev2_ids = set(json.load(open(DEV2_MANIFEST_PATH))['identities'])
split_manifest = json.load(open(SPLIT_MANIFEST_PATH))
outer_0 = split_manifest['splits']['fold_0']
inner_0 = outer_0['inner_folds'][0]  # inner fold 0 = val for o0_i0
o0i0_train_ids = set()
for j, inf in enumerate(outer_0['inner_folds']):
    if j != 0: o0i0_train_ids.update(inf['identities'])
o0i0_train_ids &= dev2_ids

# Load training episodes for norm computation
train_eps_for_norm = []
for eid in o0i0_train_ids:
    ep = load_episode(eid, FEAT_ROOT, LABEL_ROOTS)
    if ep is not None: train_eps_for_norm.append(ep)

cat_25d = np.concatenate([e['f25d'] for e in train_eps_for_norm], axis=0)
cat_p9d = np.concatenate([e['p9d'] for e in train_eps_for_norm], axis=0)
cat_g9d = np.concatenate([e['g9d'] for e in train_eps_for_norm], axis=0)
n25d_m = torch.tensor(cat_25d.mean(0), device=DEVICE)
n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=DEVICE)
np9d_m = torch.tensor(cat_p9d.mean(0), device=DEVICE)
np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=DEVICE)
ng9d_m = torch.tensor(cat_g9d.mean(0), device=DEVICE)
ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=DEVICE)
print('Norm computed from {} training episodes'.format(len(train_eps_for_norm)))

# ── 2. Load C4 episodes ──
print('\n--- Loading C4 episodes ---')
c4_ids = json.load(open(C4_MANIFEST_PATH))['identities']
c4_eps = {}
for eid in c4_ids:
    ep = load_episode(eid, FEAT_ROOT, LABEL_ROOTS)
    if ep is not None: c4_eps[eid] = ep
print('Loaded {}/{} C4 episodes'.format(len(c4_eps), len(c4_ids)))
missing = [eid for eid in c4_ids if eid not in c4_eps]
if missing:
    print('WARNING: {} C4 identities have no data'.format(len(missing)))

# ── 3. Load o0_i0 checkpoint ──
print('\n--- Loading o0_i0 checkpoint ---')
ckpt = torch.load(os.path.join(TRAIN_DIR, 'o0_i0', 'checkpoint.pt'), map_location=DEVICE, weights_only=False)
encoder = N4Encoder().to(DEVICE); head = nn.Linear(HIDDEN, 1).to(DEVICE)
encoder.load_state_dict(ckpt['enc']); head.load_state_dict(ckpt['head'])
encoder.eval(); head.eval()
print('Checkpoint: best_epoch={} best_ep_auprc={:.4f}'.format(ckpt['best_epoch'], ckpt['best_ep_auprc']))

# ── 4. Run inference ──
print('\n--- Running inference on {} C4 episodes ---'.format(len(c4_eps)))
predictions = {}  # eid -> per-step scores
ep_scores = {}    # eid -> episode score
ep_labels = {}    # eid -> 1 if has_opp else 0

with torch.no_grad():
    for eid, e in sorted(c4_eps.items()):
        base = torch.cat([(torch.tensor(e['f25d'], device=DEVICE) - n25d_m) / n25d_s,
                          (torch.tensor(e['p9d'], device=DEVICE) - np9d_m) / np9d_s,
                          (torch.tensor(e['g9d'], device=DEVICE) - ng9d_m) / ng9d_s,
                          torch.tensor(e['proxies'], device=DEVICE)], dim=-1).unsqueeze(0)
        sc = head(encoder(base)).squeeze().cpu().numpy()  # raw logits (not sigmoid)
        predictions[eid] = sc
        # Episode score: max raw logit over candidate_close steps in [0, max_t)
        ep_sc = float(sc[:e['max_t']][e['cc'][:e['max_t']]].max()) if e['cc'][:e['max_t']].any() and e['max_t'] > 0 else float(sc[:e['max_t']].max()) if e['max_t'] > 0 else float(sc.max())
        ep_scores[eid] = ep_sc
        ep_labels[eid] = 1 if e['has_opp'] else 0

# ── 5. Compute metrics ──
print('\n=== C4 RAW RANKING METRICS ===')

# Health checks
scores_arr = np.array(list(ep_scores.values()))
labels_arr = np.array([ep_labels[eid] for eid in sorted(ep_scores.keys())])
health = {}
health['nan_count'] = int(np.isnan(scores_arr).sum())
health['inf_count'] = int(np.isinf(scores_arr).sum())
health['constant'] = bool(scores_arr.std() < 1e-10)
health['identity_coverage'] = '{} / {}'.format(len(c4_eps), len(c4_ids))
health['missing_ids'] = missing
health['h1'] = health['nan_count'] == 0 and health['inf_count'] == 0
health['h2'] = not health['constant']
health['h4'] = len(c4_eps) == len(c4_ids)
print('H1 (NaN/Inf): {}  H2 (not constant): {}  H4 (coverage): {}'.format(
    'PASS' if health['h1'] else 'FAIL', 'PASS' if health['h2'] else 'FAIL',
    'PASS' if health['h4'] else 'FAIL'))

# G1: Pooled episode AUROC
ep_ids_sorted = sorted(ep_scores.keys())
pooled_labels = np.array([ep_labels[eid] for eid in ep_ids_sorted])
pooled_scores = np.array([ep_scores[eid] for eid in ep_ids_sorted])
g1_auroc = auroc(pooled_labels, pooled_scores)
n_pos = int(pooled_labels.sum()); n_neg = len(pooled_labels) - n_pos
print('\nG1: Pooled episode AUROC = {:.4f} (n_pos={} n_neg={})'.format(g1_auroc, n_pos, n_neg))
g1_pass = not np.isnan(g1_auroc) and g1_auroc >= 0.85

# G2: Pooled episode AUPRC
g2_auprc = auprc(pooled_labels, pooled_scores)
prevalence = n_pos / max(len(pooled_labels), 1)
print('G2: Pooled episode AUPRC = {:.4f}  prevalence = {:.3f}  baseline = {:.3f}'.format(
    g2_auprc, prevalence, prevalence))
g2_pass = not np.isnan(g2_auprc) and g2_auprc >= 0.85

# G3: Per-suite
print('\nG3: Per-suite episode AUROC')
suite_scores = defaultdict(lambda: {'labels': [], 'scores': []})
for eid, e in c4_eps.items():
    suite_scores[e['suite']]['labels'].append(ep_labels[eid])
    suite_scores[e['suite']]['scores'].append(ep_scores[eid])

g3_results = {}
g3_all_pass = True
g3_evaluable_count = 0
for suite in sorted(suite_scores.keys()):
    sl = np.array(suite_scores[suite]['labels'])
    ss = np.array(suite_scores[suite]['scores'])
    n_pos_s = int(sl.sum()); n_neg_s = len(sl) - n_pos_s
    evaluable = n_pos_s >= 5 and n_neg_s >= 5
    auc = auroc(sl, ss) if evaluable else float('nan')
    if evaluable:
        g3_evaluable_count += 1
        passes = auc >= 0.80
        if not passes: g3_all_pass = False
    else:
        passes = None
    g3_results[suite] = {'auc': auc, 'n_pos': n_pos_s, 'n_neg': n_neg_s,
                         'evaluable': evaluable, 'pass': passes}
    status = 'PASS' if passes else ('FAIL' if passes is False else 'NOT_ESTIMABLE')
    print('  {}: AUC={:.4f} n_pos={} n_neg={} {}'.format(suite, auc if evaluable else float('nan'), n_pos_s, n_neg_s, status))

# G4/G5: opp vs F3/F4
print('\nG4/G5: Hard negative ranking')
f3_eids = [eid for eid, e in c4_eps.items() if e['absence_reason'] == 'F3_NO_MANIPULATION']
f4_eids = [eid for eid, e in c4_eps.items() if e['absence_reason'] == 'F4_NO_STABLE_GRASP']
f1_eids = [eid for eid, e in c4_eps.items() if e['absence_reason'] == 'F1_STRUCTURAL_ZERO']
opp_eids = [eid for eid, e in c4_eps.items() if e['has_opp']]

print('  C4 strata: opp={} F1={} F3={} F4={} other={}'.format(
    len(opp_eids), len(f1_eids), len(f3_eids), len(f4_eids),
    len(c4_eps) - len(opp_eids) - len(f1_eids) - len(f3_eids) - len(f4_eids)))

# G4: opp vs F3
g4_evaluable = len(f3_eids) >= 5
g4_auc = float('nan')
if g4_evaluable:
    opp_f3_labels = np.array([1]*len(opp_eids) + [0]*len(f3_eids))
    opp_f3_scores = np.array([ep_scores[eid] for eid in opp_eids] + [ep_scores[eid] for eid in f3_eids])
    g4_auc = auroc(opp_f3_labels, opp_f3_scores)
g4_pass = g4_evaluable and not np.isnan(g4_auc) and g4_auc >= 0.70
print('  G4 (opp vs F3): AUC={:.4f} evaluable={} {}'.format(g4_auc, g4_evaluable,
    'PASS' if g4_pass else ('FAIL' if g4_evaluable else 'NOT_ESTIMABLE')))

# G5: opp vs F4
g5_evaluable = len(f4_eids) >= 5
g5_auc = float('nan')
if g5_evaluable:
    opp_f4_labels = np.array([1]*len(opp_eids) + [0]*len(f4_eids))
    opp_f4_scores = np.array([ep_scores[eid] for eid in opp_eids] + [ep_scores[eid] for eid in f4_eids])
    g5_auc = auroc(opp_f4_labels, opp_f4_scores)
g5_pass = g5_evaluable and not np.isnan(g5_auc) and g5_auc >= 0.70
print('  G5 (opp vs F4): AUC={:.4f} evaluable={} {}'.format(g5_auc, g5_evaluable,
    'PASS' if g5_pass else ('FAIL' if g5_evaluable else 'NOT_ESTIMABLE')))

# G6/G7: False peak rate
print('\nG6/G7: False peak rate')
opp_ep_scores = np.array([ep_scores[eid] for eid in opp_eids])
p10_ref = float(np.percentile(opp_ep_scores, 10, interpolation='linear')) if len(opp_ep_scores) > 0 else 0.0
print('  Positive p10 reference: {:.4f} (n_opp={})'.format(p10_ref, len(opp_eids)))

g6_evaluable = len(f3_eids) >= 5
g6_rate = float('nan')
if g6_evaluable and len(opp_ep_scores) > 0:
    f3_scores = np.array([ep_scores[eid] for eid in f3_eids])
    g6_rate = float((f3_scores >= p10_ref).mean())
g6_pass = g6_evaluable and not np.isnan(g6_rate) and g6_rate <= 0.50
print('  G6 (F3 false peak): rate={:.4f} evaluable={} {}'.format(g6_rate, g6_evaluable,
    'PASS' if g6_pass else ('FAIL' if g6_evaluable else 'NOT_ESTIMABLE')))

g7_evaluable = len(f4_eids) >= 5
g7_rate = float('nan')
if g7_evaluable and len(opp_ep_scores) > 0:
    f4_scores = np.array([ep_scores[eid] for eid in f4_eids])
    g7_rate = float((f4_scores >= p10_ref).mean())
g7_pass = g7_evaluable and not np.isnan(g7_rate) and g7_rate <= 0.50
print('  G7 (F4 false peak): rate={:.4f} evaluable={} {}'.format(g7_rate, g7_evaluable,
    'PASS' if g7_pass else ('FAIL' if g7_evaluable else 'NOT_ESTIMABLE')))

# ── 6. Localization metrics (report only) ──
print('\n--- Localization diagnostics (report only) ---')
top1_hits = 0; top3_hits = 0; total_opp = 0
offsets = []
inside_scores = []; outside_scores = []
for eid in opp_eids:
    e = c4_eps[eid]; sc = predictions[eid]
    # Find first feasible start
    feasible_steps = np.where(e['k10_s'][:e['max_t']])[0]
    if len(feasible_steps) == 0: continue
    first = feasible_steps[0]
    corridor_end = min(first + K10, e['max_t'])
    total_opp += 1
    # Top-1
    argmax_t = int(np.argmax(sc[:e['max_t']]))
    if first <= argmax_t < corridor_end: top1_hits += 1
    # Top-3
    top3 = np.argsort(sc[:e['max_t']])[-3:]
    if any(first <= t < corridor_end for t in top3): top3_hits += 1
    offsets.append(argmax_t - first)
    # Inside vs outside score
    inside_scores.append(float(sc[first:corridor_end].mean()))
    outside_feasible = [t for t in range(e['max_t']) if not e['k10_s'][t]]
    if outside_feasible:
        outside_scores.append(float(sc[outside_feasible].mean()))

print('  Top-1 corridor hit: {:.4f} ({}/{})'.format(top1_hits/max(total_opp,1), top1_hits, total_opp))
print('  Top-3 corridor hit: {:.4f} ({}/{})'.format(top3_hits/max(total_opp,1), top3_hits, total_opp))
if offsets:
    print('  Argmax offset: mean={:.1f} median={:.1f}'.format(np.mean(offsets), np.median(offsets)))
if inside_scores and outside_scores:
    print('  Inside score mean: {:.4f}  Outside score mean: {:.4f}'.format(
        np.mean(inside_scores), np.mean(outside_scores)))

# ── 7. Overall verdict ──
print('\n' + '='*60)
all_health = health['h1'] and health['h2'] and health['h4']
all_evaluable_pass = g3_all_pass and (not g4_evaluable or g4_pass) and (not g5_evaluable or g5_pass) and (not g6_evaluable or g6_pass) and (not g7_evaluable or g7_pass)
any_not_estimable = (not g3_evaluable_count) or (not g4_evaluable) or (not g5_evaluable) or (not g6_evaluable) or (not g7_evaluable)
any_evaluable_fail = (g3_evaluable_count > 0 and not g3_all_pass) or (g4_evaluable and not g4_pass) or (g5_evaluable and not g5_pass) or (g6_evaluable and not g6_pass) or (g7_evaluable and not g7_pass)

if not all_health:
    verdict = 'FAIL'
elif not g1_pass:
    verdict = 'FAIL'
elif any_evaluable_fail:
    verdict = 'FAIL'
elif any_not_estimable:
    verdict = 'PARTIAL_NOT_ESTIMABLE'
else:
    verdict = 'PASS'

print('C4 RAW RANKING: {}'.format(verdict))
print('  G1 (pooled AUROC >= 0.85): {} ({:.4f})'.format('PASS' if g1_pass else 'FAIL', g1_auroc))
print('  Health: {}'.format('PASS' if all_health else 'FAIL'))
print('  All evaluable G3-G7: {}'.format('PASS' if all_evaluable_pass else ('FAIL' if any_evaluable_fail else 'NOT_ESTIMABLE')))

# ── 8. Write outputs ──
print('\n--- Writing outputs ---')

# Per-split metrics (all C4 via o0_i0, so one split)
per_split = {'o0_i0': {'n_total': len(c4_eps), 'n_pos': n_pos, 'n_neg': n_neg,
    'ep_auroc': g1_auroc, 'ep_auprc': g2_auprc}}

# Save predictions
pred_out = {}
for eid in sorted(predictions.keys()):
    e = c4_eps[eid]
    pred_out[eid] = {
        'eid': eid, 'suite': e['suite'], 'T': e['T'],
        'has_opp': bool(e['has_opp']), 'absence_reason': e['absence_reason'],
        'episode_score_raw_logit': float(ep_scores[eid]),
        'step_scores_raw_logit': [float(x) for x in predictions[eid]]
    }
with open(os.path.join(OUT_ROOT, 'predictions', 'c4_predictions.json'), 'w') as f:
    json.dump(pred_out, f)

# Write metrics
metrics = {
    'schema': 'C4_RAW_RANKING_METRICS_V1',
    'verdict': verdict,
    'gates': {
        'G1_pooled_ep_auroc': {'value': g1_auroc, 'threshold': 0.85, 'pass': g1_pass, 'n_pos': n_pos, 'n_neg': n_neg},
        'G2_pooled_ep_auprc': {'value': g2_auprc, 'threshold': 0.85, 'pass': g2_pass, 'prevalence': prevalence, 'baseline': prevalence},
        'G3_per_suite': g3_results,
        'G4_opp_vs_F3': {'auc': g4_auc, 'evaluable': g4_evaluable, 'pass': g4_pass, 'n_f3': len(f3_eids)},
        'G5_opp_vs_F4': {'auc': g5_auc, 'evaluable': g5_evaluable, 'pass': g5_pass, 'n_f4': len(f4_eids)},
        'G6_f3_false_peak': {'rate': g6_rate, 'p10_ref': p10_ref, 'evaluable': g6_evaluable, 'pass': g6_pass},
        'G7_f4_false_peak': {'rate': g7_rate, 'p10_ref': p10_ref, 'evaluable': g7_evaluable, 'pass': g7_pass}
    },
    'health': health,
    'localization': {
        'top1_corridor_hit': top1_hits / max(total_opp, 1),
        'top3_corridor_hit': top3_hits / max(total_opp, 1),
        'mean_argmax_offset': float(np.mean(offsets)) if offsets else None,
        'median_argmax_offset': float(np.median(offsets)) if offsets else None,
        'mean_inside_score': float(np.mean(inside_scores)) if inside_scores else None,
        'mean_outside_score': float(np.mean(outside_scores)) if outside_scores else None
    },
    'denominator_audit': {
        'c4_manifest_count': len(c4_ids),
        'c4_loaded_count': len(c4_eps),
        'c4_missing': missing,
        'checkpoint_used': 'o0_i0',
        'n_train_eps_for_norm': len(train_eps_for_norm)
    }
}
with open(os.path.join(OUT_ROOT, 'per_split_metrics.json'), 'w') as f:
    json.dump(per_split, f, indent=2)
with open(os.path.join(OUT_ROOT, 'per_suite_metrics.json'), 'w') as f:
    json.dump(g3_results, f, indent=2)

# Hard negative analysis
hn = {
    'opp_count': len(opp_eids), 'f1_count': len(f1_eids),
    'f3_count': len(f3_eids), 'f4_count': len(f4_eids),
    'f3_scores': {eid: float(ep_scores[eid]) for eid in f3_eids},
    'f4_scores': {eid: float(ep_scores[eid]) for eid in f4_eids},
    'opp_p10': p10_ref, 'opp_median': float(np.median(opp_ep_scores)) if len(opp_ep_scores) > 0 else None
}
with open(os.path.join(OUT_ROOT, 'hard_negative_analysis.json'), 'w') as f:
    json.dump(hn, f, indent=2)

# Localization
loc = {'top1_hit_rate': top1_hits/max(total_opp,1), 'top3_hit_rate': top3_hits/max(total_opp,1),
       'offsets': [int(x) for x in offsets], 'inside_mean': float(np.mean(inside_scores)) if inside_scores else None,
       'outside_mean': float(np.mean(outside_scores)) if outside_scores else None}
with open(os.path.join(OUT_ROOT, 'localization_analysis.json'), 'w') as f:
    json.dump(loc, f, indent=2)

# Denominator audit
den = {'manifest': len(c4_ids), 'loaded': len(c4_eps), 'missing': missing,
       'checkpoint': 'o0_i0', 'n_norm_eps': len(train_eps_for_norm)}
with open(os.path.join(OUT_ROOT, 'denominator_audit.json'), 'w') as f:
    json.dump(den, f, indent=2)

# ── 9. Write C4 receipt and SHA256SUMS ──
receipt = {
    'schema': 'C4_RAW_RANKING_RECEIPT_V1',
    'verdict': verdict,
    'g1_auroc': g1_auroc, 'g2_auprc': g2_auprc,
    'checkpoint': 'o0_i0',
    'n_c4_loaded': len(c4_eps), 'n_c4_manifest': len(c4_ids),
    'health': health, 'gates_summary': {
        'G1': g1_pass, 'G2': g2_pass, 'G3_all_pass': g3_all_pass,
        'G4': g4_pass if g4_evaluable else 'NOT_ESTIMABLE',
        'G5': g5_pass if g5_evaluable else 'NOT_ESTIMABLE',
        'G6': g6_pass if g6_evaluable else 'NOT_ESTIMABLE',
        'G7': g7_pass if g7_evaluable else 'NOT_ESTIMABLE'
    }
}
with open(os.path.join(OUT_ROOT, 'C4_RAW_RANKING_RECEIPT_V1.json'), 'w') as f:
    json.dump(receipt, f, indent=2)

# Access ledger update
ledger_path = EVIDENCE + '/C4_ACCESS_LEDGER.json'
ledger = json.load(open(ledger_path))
ledger['c4_access_count'] = len(c4_eps)
ledger['access_log'].append({
    'timestamp': '2026-07-25',
    'action': 'C4_RAW_RANKING_INFERENCE',
    'identities_accessed': len(c4_eps),
    'checkpoint_used': 'o0_i0',
    'output_root': 'c4_raw_ranking_v1'
})
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

print('\nReceipt: {}'.format(os.path.join(OUT_ROOT, 'C4_RAW_RANKING_RECEIPT_V1.json')))
print('SHA256SUMS: {}'.format(sums_sha[:16]))
print('C4 access count updated: {} (P4=0, H2=0)'.format(ledger['c4_access_count']))
print('\nDONE. C4 raw ranking complete.')
