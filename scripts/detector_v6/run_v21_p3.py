"""P3 Inference + Scheduler Search + C3 Receipt Fix.

P3 = 300 enriched identities. Simple scheduler: cc + score>=tau for d steps → EMIT.
TP: emit step must be within external-K10 feasible corridor.
Reports: valid recall, mistimed rate, F1/F3/F4 conditional FS.
"""
import json, os, sys, hashlib, time, numpy as np
import torch
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
STUDENT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_student_training_v1'
C3_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_C3_IDENTITY_MANIFEST_V2.json'
P3_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_P3_IDENTITY_MANIFEST_V2.json'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
               '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_p3'

K10=10; SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
                   'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
os.makedirs(OUT_ROOT, exist_ok=True)

sys.path.insert(0,'/mnt/sdc/dty_user/openvla_attack/src')
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

# ── Load all 12 models (ensemble) ──
print('Loading models...')
models = {}
for sn in SPLITS:
    ckpt = torch.load(os.path.join(STUDENT_ROOT, sn, 'checkpoint.pt'), map_location=device, weights_only=False)
    model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64, receptive_field=32,
        dropout=0.1, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability'])
    model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()
    models[sn] = model

# ── Inference helper ──
def run_inference(id_set, label):
    records = []; ep_data = {}
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
                    if eid not in id_set: continue
                    lp = os.path.join(tp, state, 'factorized_teacher_v1.jsonl')
                    fp = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                    if not os.path.isfile(lp) or not os.path.isfile(fp): continue
                    recs = [json.loads(l) for l in open(fp).read().splitlines() if l.strip()]
                    labels_l = [json.loads(l) for l in open(lp).read().splitlines() if l.strip()]
                    labels_l.sort(key=lambda r:r['step']); T = len(recs); max_t = min(T,T-K10+1)

                    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
                        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
                        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
                        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
                        for r in recs], dtype=np.float32)
                    x_cat = torch.tensor(np.concatenate([f25d,p9d,g9d], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        raw_sum = sum(models[sn](x_cat)['k10_startability'].squeeze().cpu().numpy() for sn in SPLITS)
                    raw = raw_sum / len(SPLITS)
                    # FROZEN: score = raw ensemble logit (not sigmoid)
                    score = raw  # RAW_IDENTITY

                    cc_arr = np.array([labels_l[min(t,len(labels_l)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                    k10_s = np.array([labels_l[min(t,len(labels_l)-1)].get('strict_k10_feasible',False) and labels_l[min(t,len(labels_l)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    k10_k = np.array([labels_l[min(t,len(labels_l)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    has_opp = bool(k10_s[:max_t].any())

                    # Classify absence reason
                    absence_reason = 'OPPORTUNITY_PRESENT'
                    if not has_opp:
                        any_k10_known = any(k10_k[:max_t])
                        any_manip_known = any(labels_l[min(t,len(labels_l)-1)].get('manipulation_active_known_mask',False) for t in range(max_t))
                        any_grasp_known = any(labels_l[min(t,len(labels_l)-1)].get('grasp_established_known_mask',False) for t in range(max_t))
                        n_manip_pos = sum(1 for t in range(max_t) if labels_l[min(t,len(labels_l)-1)].get('manipulation_active',False) and labels_l[min(t,len(labels_l)-1)].get('manipulation_active_known_mask',False))
                        n_grasp_pos = sum(1 for t in range(max_t) if labels_l[min(t,len(labels_l)-1)].get('grasp_established',False) and labels_l[min(t,len(labels_l)-1)].get('grasp_established_known_mask',False))
                        if not any_k10_known: absence_reason = 'F1_STRUCTURAL_ZERO'
                        elif n_manip_pos == 0 and any_manip_known: absence_reason = 'F3_NO_MANIPULATION'
                        elif n_grasp_pos == 0 and any_grasp_known: absence_reason = 'F4_NO_STABLE_GRASP'
                        else: absence_reason = 'OTHER_ABSENT'

                    ep_data[eid] = {'T':T,'max_t':max_t,'score':score,'cc':cc_arr,'k10_s':k10_s,'k10_k':k10_k,
                                    'has_opp':has_opp,'absence_reason':absence_reason,'suite':suite,'eid':eid}
                    for t in range(max_t):
                        if k10_k[t]:
                            records.append({'eid':eid,'step':t,'score':float(score[t]),'label':1.0 if k10_s[t] else 0.0,
                                           'cc':bool(cc_arr[t]),'k10_feasible':bool(k10_s[t])})
    return records, ep_data

# ═══ 1. C3 Receipt Fix ═══
print('=== C3 RECEIPT FIX ===')
c3_ids = set(json.load(open(C3_MANIFEST_PATH))['identities'])
_, c3_eps = run_inference(c3_ids, 'C3')

# Count per-suite
c3_suites = defaultdict(lambda: {'opp':0,'abs':0})
for eid, d in c3_eps.items():
    if d['has_opp']: c3_suites[d['suite']]['opp'] += 1
    else: c3_suites[d['suite']]['abs'] += 1
print('C3 per-suite:')
for s in sorted(c3_suites):
    v = c3_suites[s]; print('  {}: opp={} abs={}'.format(s, v['opp'], v['abs']))

# Bootstrap CI for episode AUROC
ep_scores_c3 = []; ep_labels_c3 = []
for eid, d in c3_eps.items():
    max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['score'][:max_t]
    ep_scores_c3.append(float(sc[cc].max()) if cc.any() else float(sc.max()))
    ep_labels_c3.append(1.0 if d['has_opp'] else 0.0)
ep_scores_a = np.array(ep_scores_c3); ep_labels_a = np.array(ep_labels_c3)
ep_auc = auroc(ep_labels_a, ep_scores_a)

# Bootstrap
aucs_boot = []
for _ in range(1000):
    idx = np.random.choice(len(ep_labels_a), len(ep_labels_a), replace=True)
    sl = ep_labels_a[idx]; ss = ep_scores_a[idx]
    if sl.sum()==0 or sl.sum()==len(sl): continue
    aucs_boot.append(auroc(sl, ss))
ci_lo = np.percentile(aucs_boot, 2.5); ci_hi = np.percentile(aucs_boot, 97.5)
print('C3 episode AUROC: {:.4f} (95% CI: [{:.4f}, {:.4f}])'.format(ep_auc, ci_lo, ci_hi))

# Score definition
print('Score definition: RAW ensemble logit (not sigmoid). a=1, b=0.')

# ═══ 2. P3 Inference ═══
print('\n=== P3 INFERENCE ===')
p3_ids = set(json.load(open(P3_MANIFEST_PATH))['identities'])
print('P3 identities: {}'.format(len(p3_ids)))
_, p3_eps = run_inference(p3_ids, 'P3')

# P3 strata
p3_strata = defaultdict(int)
for eid, d in p3_eps.items():
    p3_strata[d['absence_reason']] += 1
print('P3 strata:')
for k in sorted(p3_strata): print('  {}: {}'.format(k, p3_strata[k]))

# ═══ 3. Simple Scheduler Search ═══
print('\n=== SCHEDULER SEARCH ===')

def simple_scheduler(d, tau, persistence):
    """Returns (emit, emit_step, valid_tp, mistimed).
    valid_tp: emit step IS within K10 feasible corridor.
    mistimed: opp episode, emit step NOT within feasible corridor.
    """
    max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['score'][:max_t]
    k10_s = d['k10_s'][:max_t]; k10_k = d['k10_k'][:max_t]
    T = d['T']
    cons = 0
    for t in range(max_t):
        if cc[t] and sc[t] >= tau:
            cons += 1
            if cons >= persistence:
                # Check if emit step is within K10 feasible corridor
                if k10_s[t] and k10_k[t]:
                    return True, t, 'valid_tp', ''
                elif d['has_opp']:
                    return True, t, 'mistimed', ''
                else:
                    return True, t, 'false_start', d['absence_reason']
        else:
            cons = 0
    return False, -1, ('fn' if d['has_opp'] else 'tn'), ''

# Score quantiles for threshold candidates
all_cc_scores = []
for eid, d in p3_eps.items():
    max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['score'][:max_t]
    if cc.any(): all_cc_scores.extend(sc[cc].tolist())
all_cc_scores = np.array(all_cc_scores)
print('CC score quantiles: p10={:.4f} p50={:.4f} p90={:.4f}'.format(
    np.percentile(all_cc_scores,10), np.percentile(all_cc_scores,50), np.percentile(all_cc_scores,90)))

# Build threshold candidates from actual score quantiles
tau_candidates = sorted(set([round(x,4) for x in np.percentile(all_cc_scores, np.linspace(5,95,19))]))
print('Threshold candidates: {} values from {:.4f} to {:.4f}'.format(len(tau_candidates), tau_candidates[0], tau_candidates[-1]))

best = None
for tau in tau_candidates:
    for d_persist in [1,2,3]:
        vt=0; mt=0; fs=0; fn=0; tn=0
        fs_f1=0; fs_f3=0; fs_f4=0; fs_other=0
        n_f3=0; n_f4=0; n_f1=0; n_abs=0; n_opp=0
        timing = []

        for eid, d_ep in p3_eps.items():
            emit, estep, kind, reason = simple_scheduler(d_ep, tau, d_persist)
            if d_ep['has_opp']:
                n_opp += 1
                if kind == 'valid_tp': vt += 1; timing.append(estep)
                elif kind == 'mistimed': mt += 1
                elif kind == 'fn': fn += 1
            else:
                n_abs += 1
                if kind == 'false_start':
                    fs += 1
                    if reason == 'F1_STRUCTURAL_ZERO': fs_f1 += 1; n_f1 += 1
                    elif reason == 'F3_NO_MANIPULATION': fs_f3 += 1; n_f3 += 1
                    elif reason == 'F4_NO_STABLE_GRASP': fs_f4 += 1; n_f4 += 1
                    else: fs_other += 1
                elif kind == 'tn': tn += 1
                # Count absent strata even if no FS
                if d_ep['absence_reason'] == 'F1_STRUCTURAL_ZERO': n_f1 += 1
                elif d_ep['absence_reason'] == 'F3_NO_MANIPULATION': n_f3 += 1
                elif d_ep['absence_reason'] == 'F4_NO_STABLE_GRASP': n_f4 += 1

        # Compute rates
        fs_all = fs / max(n_abs,1)
        fs_f3_rate = fs_f3 / max(n_f3,1)
        fs_f4_rate = fs_f4 / max(n_f4,1)
        recall = vt / max(n_opp,1)
        mistimed_rate = mt / max(n_opp,1)
        total_emit = vt + mt + fs

        # Gates
        if fs_all > 0.10: continue
        if fs_f3_rate > 0.10: continue
        if fs_f4_rate > 0.10: continue
        if total_emit == 0: continue
        if vt == 0: continue

        # K10 executable: emit step + K10 <= T
        k10_ok = 0
        for eid, d_ep in p3_eps.items():
            emit, estep, kind, _ = simple_scheduler(d_ep, tau, d_persist)
            if kind == 'valid_tp' and estep + K10 <= d_ep['T']: k10_ok += 1

        candidate = {'tau':tau,'d':d_persist,'recall':recall,'fs_all':fs_all,
            'fs_f3':fs_f3_rate,'fs_f4':fs_f4_rate,'mistimed':mistimed_rate,
            'vt':vt,'mt':mt,'fs':fs,'fn':fn,'tn':tn,'n_opp':n_opp,'n_abs':n_abs,
            'n_f3':n_f3,'n_f4':n_f4,'fs_f3_n':fs_f3,'fs_f4_n':fs_f4,
            'k10_ok':k10_ok,'avg_timing':np.mean(timing) if timing else 0}

        if best is None or recall > best['recall']:
            best = candidate

if best:
    print('\n=== SELECTED POLICY ===')
    print('tau={:.4f}  persistence={}  recall={:.4f}  FS_all={:.4f}'.format(best['tau'], best['d'], best['recall'], best['fs_all']))
    print('Valid TP={}  Mistimed={}  FS={}  FN={}  TN={}'.format(best['vt'], best['mt'], best['fs'], best['fn'], best['tn']))
    print('F3 FS: {}/{} = {:.4f}  F4 FS: {}/{} = {:.4f}'.format(best['fs_f3_n'], best['n_f3'], best['fs_f3'], best['fs_f4_n'], best['n_f4'], best['fs_f4']))
    print('Mistimed rate: {:.4f}  K10 executable: {}/{}'.format(best['mistimed'], best['k10_ok'], best['vt']))

    scheduler = {'schema':'FORMAL_V21_P3_SCHEDULER_FREEZE_V1','method':'SIMPLE_THRESHOLD_PERSISTENCE',
        'score_type':'RAW_ENSEMBLE_LOGIT','parameters':{'tau_start':best['tau'],'persistence':best['d'],'K10_window':K10},
        'P3_validation':{'valid_recall':best['recall'],'fs_all':best['fs_all'],'fs_f3':best['fs_f3'],
            'fs_f4':best['fs_f4'],'mistimed_rate':best['mistimed'],'k10_executable_rate':best['k10_ok']/max(best['vt'],1)},
    }
    with open(os.path.join(OUT_ROOT,'FORMAL_V21_P3_SCHEDULER_FREEZE_V1.json'),'w') as f: json.dump(scheduler,f,indent=2)
    print('P3_SCHEDULER_FREEZE = PASS')
else:
    print('\nNO FEASIBLE POLICY')
    print('P3_SCHEDULER_FREEZE = FAIL')

# ═══ Seal ═══
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
print('Seal: {}'.format(sh[:16]))
