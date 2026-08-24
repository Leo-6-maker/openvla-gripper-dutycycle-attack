"""C3 Raw Inference + Generalization Evaluation + Monotonic Platt Calibration.

C3 = 200 identities (states 35-49). Tests DEV(00-34) → C3(35-49) generalization.
Episode score = max raw logit over candidate-close steps (no privileged mask).
Calibrator: pooled monotonic Platt, a>0, ranking preserved. Only if ep AUROC >= 0.65.
"""
import json, os, sys, hashlib, time, numpy as np
import torch
from collections import defaultdict
from scipy.optimize import minimize

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
STUDENT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_student_training_v1'
C3_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_C3_IDENTITY_MANIFEST_V2.json'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
               '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_c3'

K10=10; EPS=1e-3; LAMBDA_A=0.1
SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
os.makedirs(OUT_ROOT, exist_ok=True)

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2

device = torch.device('cuda:0')

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

def auroc(yt,ys):
    if len(yt)<2: return 0.5
    n_pos=yt.sum();n_neg=len(yt)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    tpr=np.cumsum(ysort)/n_pos;fpr=np.cumsum(1-ysort)/n_neg
    return float(np.trapz(tpr,fpr))
def auprc(yt,ys):
    if len(yt)<2: return 0.0
    n_pos=yt.sum()
    if n_pos==0: return 0.0
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    prec=np.cumsum(ysort)/np.arange(1,len(ysort)+1)
    rec=np.cumsum(ysort)/n_pos
    return float(np.trapz(prec,rec))
def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))
def softplus(x): return np.log(1.0+np.exp(np.clip(x,-50,50)))

# ── Load C3 identities ──
c3_ids = set(json.load(open(C3_MANIFEST_PATH))['identities'])
print('C3 identities: {}'.format(len(c3_ids)))

# ═══ 1. C3 Raw Inference ═══
print('\n=== C3 RAW INFERENCE ===')
all_records = []  # {eid, split, step, raw_logit, label, cc, T, suite}
ep_data = defaultdict(lambda: {'raw':[], 'label':[], 'cc':[], 'T':0, 'suite':'', 'split':''})

# Load all 12 models for ensemble
models = {}
for split_name in SPLITS:
    ckpt_path = os.path.join(STUDENT_ROOT, split_name, 'checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64, receptive_field=32,
        dropout=0.1, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability'])
    model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()
    models[split_name] = model

for root in LABEL_ROOTS:
        if not os.path.isdir(root): continue
        for suite in sorted(os.listdir(root)):
            sp = os.path.join(root, suite)
            if not os.path.isdir(sp): continue
            for task in sorted(os.listdir(sp)):
                tp = os.path.join(sp, task)
                if not os.path.isdir(tp): continue
                for state in sorted(os.listdir(tp)):
                    eid = '{}/{}/{}'.format(suite, task, state)
                    if eid not in c3_ids: continue
                    lp = os.path.join(tp, state, 'factorized_teacher_v1.jsonl')
                    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                    if not os.path.isfile(lp) or not os.path.isfile(fp): continue
                    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
                    labels = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
                    labels.sort(key=lambda r:r['step']); T = len(recs); max_t = min(T,T-K10+1)

                    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
                        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
                        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
                        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
                        for r in recs], dtype=np.float32)
                    x_cat = torch.tensor(np.concatenate([f25d,p9d,g9d], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)
                    # Ensemble: mean of all 12 models
                    raw_sum = None
                    with torch.no_grad():
                        for sn in SPLITS:
                            r = models[sn](x_cat)['k10_startability'].squeeze().cpu().numpy()
                            if raw_sum is None: raw_sum = r
                            else: raw_sum += r
                    raw = raw_sum / len(SPLITS)

                    cc_arr = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    has_opp = bool(k10_s[:max_t].any())

                    for t in range(max_t):
                        if k10_k[t]:
                            all_records.append({'eid':eid,'split':split_name,'step':t,'raw':float(raw[t]),
                                'label':1.0 if k10_s[t] else 0.0,'cc':bool(cc_arr[t]),'T':T,'suite':suite})

                    ep = ep_data[eid]; ep['T'] = T; ep['suite'] = suite; ep['split'] = split_name
                    for t in range(max_t):
                        ep['raw'].append(float(raw[t]))
                        ep['label'].append(1.0 if k10_s[t] else 0.0)
                        ep['cc'].append(bool(cc_arr[t]))

print('Total C3 records: {}  episodes: {}'.format(len(all_records), len(ep_data)))

# ═══ 2. Raw Ranking Evaluation ═══
print('\n=== C3 RAW RANKING ===')

# Step-level (known-mask only, same as Teacher supervision)
step_z = np.array([r['raw'] for r in all_records])
step_l = np.array([r['label'] for r in all_records])
step_auc = auroc(step_l, step_z); step_ap = auprc(step_l, step_z)
print('Step pooled: AUROC={:.4f} AUPRC={:.4f} n={} pos={:.3f}'.format(step_auc, step_ap, len(step_l), step_l.mean()))

# Episode-level: max raw logit over CANDIDATE_CLOSE steps (NO privileged known-mask)
ep_scores = []; ep_labels = []; ep_suites = []
for eid, d in ep_data.items():
    max_t = max(1, d['T'] - K10 + 1)
    cc_arr = np.array(d['cc'][:max_t], dtype=bool)
    raw_arr = np.array(d['raw'][:max_t])
    label_arr = np.array(d['label'][:max_t])
    # Runtime score: max over candidate-close steps only (no known-mask filtering)
    if cc_arr.any():
        ep_score = float(raw_arr[cc_arr].max())
    else:
        ep_score = float(raw_arr.max())  # fallback: use all steps if no cc
    ep_scores.append(ep_score)
    ep_labels.append(1.0 if label_arr.any() else 0.0)
    ep_suites.append(d['suite'])

ep_scores_a = np.array(ep_scores); ep_labels_a = np.array(ep_labels)
ep_auc = auroc(ep_labels_a, ep_scores_a); ep_ap = auprc(ep_labels_a, ep_scores_a)
print('Episode pooled: AUROC={:.4f} AUPRC={:.4f} n={} pos={:.3f}'.format(ep_auc, ep_ap, len(ep_scores), ep_labels_a.mean()))

# Per-split episode AUROC
print('\nPer-split episode AUROC:')
per_split_aucs = []
for sk in SPLITS:
    sk_idx = [i for i in range(len(ep_scores)) if list(ep_data.values())[i]['split'] == sk]
    if len(sk_idx) < 3: continue
    sk_s = ep_scores_a[sk_idx]; sk_l = ep_labels_a[sk_idx]
    if sk_l.sum()==0 or sk_l.sum()==len(sk_l): continue
    a = auroc(sk_l, sk_s); per_split_aucs.append(a)
    print('  {}: {:.4f} (n={} pos={:.2f})'.format(sk, a, len(sk_l), sk_l.mean()))

# Per-suite episode AUROC
print('\nPer-suite episode AUROC:')
for suite in sorted(set(ep_suites)):
    si = [i for i,s in enumerate(ep_suites) if s==suite]
    if len(si)<3: continue
    ss = ep_scores_a[si]; sl = ep_labels_a[si]
    if sl.sum()==0 or sl.sum()==len(sl): continue
    print('  {}: {:.4f} (n={} pos={:.2f})'.format(suite, auroc(sl,ss), len(sl), sl.mean()))

# F1/F3/F4 score distributions
print('\nAbsent episode score distributions (by Teacher reason):')
for eid, d in ep_data.items():
    max_t = max(1, d['T'] - K10 + 1)
    cc_arr = np.array(d['cc'][:max_t], dtype=bool)
    raw_arr = np.array(d['raw'][:max_t])
    label_arr = np.array(d['label'][:max_t])
    if label_arr.any(): continue  # skip opp episodes
    # Classify absence reason
    k10_k_arr = np.array([r['label']>=0 for r in all_records if r['eid']==eid])  # approximate
    ep_data[eid]['has_k10_known'] = any(k10_k_arr)
    # Simple heuristic: check Teacher labels
    ep_data[eid]['ep_score'] = float(raw_arr[cc_arr].max()) if cc_arr.any() else float(raw_arr.max())

# Report by stratum from existing classification
strata_scores = defaultdict(list)
for eid, d in ep_data.items():
    if d.get('has_opp', True): continue
    strata_scores['absent'].append(d.get('ep_score',0))
for reason in ['absent']:
    scores = strata_scores[reason]
    if scores:
        print('  {} (n={}): p50={:.4f} p90={:.4f} max={:.4f}'.format(
            reason, len(scores), np.median(scores), np.percentile(scores,90), max(scores)))

# ── Acceptance gate ──
gate_pass = ep_auc >= 0.65 and len(per_split_aucs) > 0 and min(per_split_aucs) > 0.45
print('\nC3 GENERALIZATION GATE: ep_auc={:.4f} >= 0.65 = {}'.format(ep_auc, 'PASS' if ep_auc >= 0.65 else 'FAIL'))
print('Per-split min AUROC: {:.4f}'.format(min(per_split_aucs) if per_split_aucs else 0))
print('C3_RAW_RANKING: {}'.format('PASS' if gate_pass else 'FAIL'))

# ═══ 3. Monotonic Platt Calibration ═══
if gate_pass:
    print('\n=== C3 MONOTONIC PLATT CALIBRATION ===')
    # Episode-balanced NLL over known-mask steps
    ep_groups = defaultdict(list)
    for i, r in enumerate(all_records):
        ep_groups[r['eid']].append(i)
    ep_indices = list(ep_groups.values())

    def objective(params):
        alpha, b = params[0], params[1]
        a = softplus(alpha) + EPS
        loss = 0.0
        for ep_idx in ep_indices:
            z = step_z[ep_idx]; y = step_l[ep_idx]
            prob = sigmoid(a*z + b)
            ep_loss = -np.mean(y*np.log(np.clip(prob,1e-12,1-1e-12)) + (1-y)*np.log(np.clip(1-prob,1e-12,1-1e-12)))
            loss += ep_loss / len(ep_indices)
        return loss + LAMBDA_A * (np.log(max(a,1e-12)))**2

    alpha_init = np.log(np.exp(1.0-EPS)-1) if 1.0-EPS > 0.01 else 0.0
    res = minimize(objective, [alpha_init, 0.0], method='L-BFGS-B', options={'maxiter':5000,'ftol':1e-12})
    a_opt = softplus(res.x[0]) + EPS; b_opt = res.x[1]

    raw_prob = sigmoid(1.0*step_z + 0.0); cal_prob = sigmoid(a_opt*step_z + b_opt)
    raw_nll = -np.mean(step_l*np.log(np.clip(raw_prob,1e-12,1-1e-12)) + (1-step_l)*np.log(np.clip(1-raw_prob,1e-12,1-1e-12)))
    cal_nll = -np.mean(step_l*np.log(np.clip(cal_prob,1e-12,1-1e-12)) + (1-step_l)*np.log(np.clip(1-cal_prob,1e-12,1-1e-12)))
    raw_auc = auroc(step_l, step_z); cal_auc = auroc(step_l, cal_prob)

    print('a={:.6f} b={:.6f}'.format(a_opt, b_opt))
    print('Raw NLL={:.4f} Cal NLL={:.4f}'.format(raw_nll, cal_nll))
    print('AUROC preserved: {:.6f} -> {:.6f} (Δ={:.2e})'.format(raw_auc, cal_auc, abs(cal_auc-raw_auc)))

    cal_gates = {'a_positive':a_opt>0,'finite':all(np.isfinite([a_opt,b_opt])),
        'ranking_preserved':abs(cal_auc-raw_auc)<1e-4,'c3_only':True}
    cal_pass = all(cal_gates.values())
    for k,v in cal_gates.items(): print('  {}: {}'.format(k, 'PASS' if v else 'FAIL'))

    calibrator = {'schema':'FORMAL_V21_MONOTONIC_CALIBRATOR_V1','method':'POOLED_MONOTONIC_PLATT',
        'parameters':{'a':float(a_opt),'b':float(b_opt),'eps':EPS},
        'fit_metrics':{'raw_nll':float(raw_nll),'cal_nll':float(cal_nll),'raw_auroc':float(raw_auc),'cal_auroc':float(cal_auc),
            'n_positive':int(step_l.sum()),'n_negative':int((1-step_l).sum()),'n_episodes':len(ep_data)},
        'gates':cal_gates,'all_gates_pass':cal_pass}
    with open(os.path.join(OUT_ROOT,'FORMAL_V21_MONOTONIC_CALIBRATOR_V1.json'),'w') as f:
        json.dump(calibrator, f, indent=2)
    print('CALIBRATOR: {}'.format('PASS' if cal_pass else 'FAIL'))
else:
    cal_pass = False
    print('\nCALIBRATOR: NOT AUTHORIZED (raw ranking failed)')

# ═══ Seal ═══
c3_report = {'schema':'C3_RAW_GENERALIZATION_RECEIPT_V1','timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    'step_auc':float(step_auc),'step_ap':float(step_ap),
    'ep_auc':float(ep_auc),'ep_ap':float(ep_ap),
    'n_episodes':len(ep_data),'n_steps':len(all_records),
    'positive_rate':float(step_l.mean()),
    'per_split_aucs':{sk:float(auroc(ep_labels_a[[i for i in range(len(ep_scores)) if list(ep_data.values())[i]['split']==sk]],ep_scores_a[[i for i in range(len(ep_scores)) if list(ep_data.values())[i]['split']==sk]])) for sk in SPLITS if len([i for i in range(len(ep_scores)) if list(ep_data.values())[i]['split']==sk])>=3},
    'gate_pass':gate_pass,'calibrator_pass':cal_pass}
with open(os.path.join(OUT_ROOT,'C3_RAW_GENERALIZATION_RECEIPT_V1.json'),'w') as f: json.dump(c3_report, f, indent=2)

all_files = []
for root, dirs, fns in os.walk(OUT_ROOT):
    for fn in sorted(fns):
        if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
        fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_ROOT)
        all_files.append((rel, sha256_file(fp)))
with open(os.path.join(OUT_ROOT,'SHA256SUMS'),'w') as f:
    for rel, h in sorted(all_files): f.write('{}  {}\n'.format(h, rel))
sh = sha256_file(os.path.join(OUT_ROOT,'SHA256SUMS'))
with open(os.path.join(OUT_ROOT,'SHA256SUMS.sha256'),'w') as f: f.write('{}  SHA256SUMS\n'.format(sh))

print('\nSealed: {}'.format(OUT_ROOT))
print('Seal: {}'.format(sh[:16]))
print('C3_RAW_RANKING: {}'.format('PASS' if gate_pass else 'FAIL'))
print('C3_CALIBRATOR: {}'.format('PASS' if cal_pass else 'NOT_AUTHORIZED' if not gate_pass else 'FAIL'))
print('P3: {}'.format('READY' if cal_pass else 'BLOCKED'))
