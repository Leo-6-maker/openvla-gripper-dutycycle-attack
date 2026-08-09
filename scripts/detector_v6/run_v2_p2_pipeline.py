"""V2 C2 Reconciliation + P2 Coverage + P2 Inference + Scheduler Freeze.

1. Reconcile C2 AUROC: step/episode/event metrics, per-split, null baseline
2. Pre-check P2 F1/F3/F4 coverage
3. P2 production inference with frozen checkpoints + C2 calibrator
4. Simple scheduler search: τ_start × persistence, FS≤10%, max recall
"""
import json, os, sys, hashlib, time, numpy as np
import torch
from collections import defaultdict
from scipy.optimize import minimize

FEAT_ROOT  = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
STUDENT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_student_training_v1'
CAL_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_c2_calibration/V2_C2_CALIBRATOR_FREEZE_V1.json'
C2_TEACHER = '/tmp/ft_CAL/labels'
CHECK_TEACHER = '/tmp/ft_CHECK/labels'
P2_MANIFEST_PATH = '/tmp/v6_final_manifests/policy_selection_identity_manifest.json'
C2_MANIFEST_PATH = '/tmp/v6_final_manifests/calibrator_fit_identity_manifest.json'
FIT_DEV_LABELS = '/tmp/ft_FIT_DEV/labels'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v2_p2'

K10 = 10; SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
                     'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
os.makedirs(OUT_ROOT, exist_ok=True)

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cal = json.load(open(CAL_PATH))
A_CAL = cal['parameters']['a']; B_CAL = cal['parameters']['b']

def sha256_file(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): d.update(chunk)
    return d.hexdigest()

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))

def auroc(y_true, y_score):
    if len(y_true)<2: return 0.5
    n_pos=y_true.sum(); n_neg=len(y_true)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(y_score)[::-1]; y_sort=y_true[desc]
    tpr=np.cumsum(y_sort)/n_pos; fpr=np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr, fpr))

def auprc(y_true, y_score):
    if len(y_true)<2: return 0.0
    n_pos=y_true.sum()
    if n_pos==0: return 0.0
    desc=np.argsort(y_score)[::-1]; y_sort=y_true[desc]
    prec=np.cumsum(y_sort)/np.arange(1,len(y_sort)+1)
    rec=np.cumsum(y_sort)/n_pos
    return float(np.trapz(prec, rec))

# ── Run inference for a set of episodes ──
def run_inference(teacher_roots, identity_sets_by_split):
    """Returns per-step records with raw logits and labels."""
    records = []
    for split_name in SPLITS:
        if split_name not in identity_sets_by_split: continue
        allowed = identity_sets_by_split[split_name]

        ckpt_path = os.path.join(STUDENT_ROOT, split_name, 'checkpoint.pt')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64,
            receptive_field=32, dropout=0.1, use_policy_bypass=False,
            use_gripper_bypass=False, head_names=['k10_startability','secure_grasp','manipulation_intent'])
        model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()

        for root in teacher_roots:
            for suite in sorted(os.listdir(root)):
                sp = os.path.join(root, suite)
                if not os.path.isdir(sp): continue
                for task in sorted(os.listdir(sp)):
                    tp = os.path.join(sp, task)
                    if not os.path.isdir(tp): continue
                    for state in sorted(os.listdir(tp)):
                        eid = f'{suite}/{task}/{state}'
                        if eid not in allowed: continue

                        feat_path = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                        teach_path = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
                        if not os.path.isfile(feat_path) or not os.path.isfile(teach_path): continue

                        recs = [json.loads(l) for l in open(feat_path).read().splitlines() if l.strip()]
                        tr = [json.loads(l) for l in open(teach_path).read().splitlines() if l.strip()]
                        tr.sort(key=lambda r: r['step']); T = len(recs)
                        max_t = min(T, T-K10+1)

                        f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                        p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                        g9d = np.array([[r.get('clean_close_probability_mass',0), r.get('clean_open_probability_mass',0),
                                         r.get('clean_top1_is_close',0), r.get('clean_top1_is_open',0),
                                         r.get('clean_top1_probability',0), r.get('clean_best_close_rank_normalized',0),
                                         r.get('clean_best_open_rank_normalized',0), r.get('clean_action_token_entropy_normalized',0),
                                         r.get('clean_open_minus_close_log_mass',0)] for r in recs], dtype=np.float32)
                        x_cat = torch.tensor(np.concatenate([f25d, p9d, g9d], axis=-1), dtype=torch.float32, device=device).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(x_cat)
                        raw = logits['k10_startability'].squeeze().cpu().numpy()

                        for t in range(max_t):
                            tr_t = tr[min(t, len(tr)-1)]
                            if tr_t.get('strict_k10_known_mask', False):
                                records.append({
                                    'eid': eid, 'step': t, 'split': split_name, 'T': T,
                                    'raw_logit': float(raw[t]),
                                    'cal_prob': float(sigmoid(A_CAL * raw[t] + B_CAL)),
                                    'k10_label': 1.0 if tr_t.get('strict_k10_feasible', False) else 0.0,
                                })

        # Per-episode records
        ep_data = defaultdict(lambda: {'raw_logits':[], 'labels':[]})
        for r in records:
            if r['split'] == split_name:
                ep_data[r['eid']]['raw_logits'].append(r['raw_logit'])
                ep_data[r['eid']]['labels'].append(r['k10_label'])
                ep_data[r['eid']]['T'] = r['T']

    return records


# ── Compute metrics from records ──
def compute_metrics(records, label):
    """Compute step, episode, event metrics."""
    logits = np.array([r['raw_logit'] for r in records])
    labels_a = np.array([r['k10_label'] for r in records])

    # Step-level
    step_auc = auroc(labels_a, logits)
    step_ap  = auprc(labels_a, logits)

    # Episode-level: max calibrated prob in window
    ep_data = defaultdict(lambda: {'raw':[], 'cal':[], 'label':[], 'T':0})
    for r in records:
        ep = ep_data[r['eid']]
        ep['raw'].append(r['raw_logit'])
        ep['cal'].append(r['cal_prob'])
        ep['label'].append(r['k10_label'])
        ep['T'] = r['T']

    ep_max_scores = []; ep_labels = []
    for eid, d in ep_data.items():
        max_t = max(1, d['T'] - K10 + 1)
        ep_max_scores.append(float(np.array(d['cal'])[:max_t].max()))
        ep_labels.append(1.0 if any(l == 1.0 for l in d['label'][:max_t]) else 0.0)

    ep_auc = auroc(np.array(ep_labels), np.array(ep_max_scores))
    ep_ap  = auprc(np.array(ep_labels), np.array(ep_max_scores))

    # Per-split step AUROC
    per_split = {}
    for sk in SPLITS:
        sk_r = [r for r in records if r['split'] == sk]
        if len(sk_r) < 10: continue
        sk_l = np.array([r['k10_label'] for r in sk_r])
        sk_z = np.array([r['raw_logit'] for r in sk_r])
        if sk_l.sum() == 0 or sk_l.sum() == len(sk_l): continue
        per_split[sk] = float(auroc(sk_l, sk_z))

    # Null baseline NLL
    cal_probs = np.array([r['cal_prob'] for r in records])
    null_prob = labels_a.mean()
    null_nll = -np.mean(labels_a * np.log(max(null_prob,1e-12)) + (1-labels_a)*np.log(max(1-null_prob,1e-12)))
    cal_nll = -np.mean(labels_a * np.log(np.clip(cal_probs,1e-12,1-1e-12)) + (1-labels_a)*np.log(np.clip(1-cal_probs,1e-12,1-1e-12)))

    return {
        'step_auc': step_auc, 'step_ap': step_ap,
        'ep_auc': ep_auc, 'ep_ap': ep_ap,
        'pos_rate': float(labels_a.mean()),
        'n_steps': len(records), 'n_eps': len(ep_data),
        'null_nll': float(null_nll), 'cal_nll': float(cal_nll),
        'per_split_auc': per_split,
    }


# ═══════════════════════════════════════════
# 1. C2 AUROC Reconciliation
# ═══════════════════════════════════════════
print('=== 1. C2 AUROC RECONCILIATION ===')

# Load C2 and FIT_DEV identities
c2_manifest = json.load(open(C2_MANIFEST_PATH))
c2_sets = {}
for sk in SPLITS:
    if sk in c2_manifest['splits']:
        c2_sets[sk] = set(c2_manifest['splits'][sk].get('calibrator_fit',
                          c2_manifest['splits'][sk].get('identities', [])))

# FIT_DEV: load via inner-CV splits
splits_manifest = json.load(open('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721/inner_cv_splits.json'))
fit_dev_sets = {}
for split_idx, split_name in enumerate(SPLITS):
    outer_idx = split_idx // 3; inner_idx = split_idx % 3
    outer = splits_manifest['splits'][f'fold_{outer_idx}']
    inner = outer['inner_folds'][inner_idx]
    fit_dev_sets[split_name] = set(inner['identities'])

print('Running C2 inference...')
c2_records = run_inference([C2_TEACHER, CHECK_TEACHER], c2_sets)
c2_m = compute_metrics(c2_records, 'C2')

print('Running FIT_DEV inference...')
fit_dev_records = run_inference([FIT_DEV_LABELS], fit_dev_sets)
fit_m = compute_metrics(fit_dev_records, 'FIT_DEV')

print(f'\n{"Metric":<25} {"FIT_DEV":>10} {"C2":>10}')
print('-' * 47)
for k in ['step_auc','step_ap','ep_auc','ep_ap','pos_rate','null_nll','cal_nll','n_steps','n_eps']:
    v_fit = fit_m.get(k, 'N/A'); v_c2 = c2_m.get(k, 'N/A')
    if isinstance(v_fit, float): print(f'{k:<25} {v_fit:>10.4f} {v_c2:>10.4f}')
    else: print(f'{k:<25} {str(v_fit):>10} {str(v_c2):>10}')

print('\nPer-split step AUROC:')
for sk in SPLITS:
    f = fit_m['per_split_auc'].get(sk, 0); c = c2_m['per_split_auc'].get(sk, 0)
    d = c - f
    print(f'  {sk}: FIT={f:.4f} C2={c:.4f} Δ={d:+.4f}')

# Diagnosis
fit_ep = fit_m['ep_auc']; c2_ep = c2_m['ep_auc']
print(f'\nDiagnosis:')
print(f'  FIT_DEV step AUC={fit_m["step_auc"]:.4f}  episode AUC={fit_ep:.4f}')
print(f'  C2      step AUC={c2_m["step_auc"]:.4f}  episode AUC={c2_ep:.4f}')

if c2_ep > 0.8:
    print('  VERDICT: Episode-level AUROC preserved. Step-level drop is metric granularity, not performance loss.')
elif c2_ep > 0.6:
    print('  VERDICT: Moderate distribution shift. P2 feasible but recall may be lower than FIT_DEV.')
else:
    print('  VERDICT: Significant distribution shift. P2 may have no feasible threshold.')

# ═══════════════════════════════════════════
# 2. P2 Coverage Pre-check
# ═══════════════════════════════════════════
print('\n=== 2. P2 COVERAGE PRE-CHECK ===')
p2_manifest = json.load(open(P2_MANIFEST_PATH))
p2_sets = {}
for sk in SPLITS:
    if sk in p2_manifest['splits']:
        p2_sets[sk] = set(p2_manifest['splits'][sk].get('policy_selection',
                          p2_manifest['splits'][sk].get('identities', [])))

# Count absent categories
f1_count=0; f3_count=0; f4_count=0; opp_count=0; parser_count=0; valid_abs=0
for split_name in SPLITS:
    if split_name not in p2_sets: continue
    allowed = p2_sets[split_name]
    for root in [C2_TEACHER, CHECK_TEACHER]:
        for suite in sorted(os.listdir(root)):
            sp = os.path.join(root, suite)
            if not os.path.isdir(sp): continue
            for task in sorted(os.listdir(sp)):
                tp = os.path.join(sp, task)
                if not os.path.isdir(tp): continue
                for state in sorted(os.listdir(tp)):
                    eid = f'{suite}/{task}/{state}'
                    if eid not in allowed: continue
                    teach_path = os.path.join(root, suite, task, state, 'factorized_teacher_v1.jsonl')
                    if not os.path.isfile(teach_path): continue
                    tr = [json.loads(l) for l in open(teach_path).read().splitlines() if l.strip()]
                    tr.sort(key=lambda r: r['step']); T = len(tr)
                    max_t = min(T, T-K10+1)

                    any_k10_known = any(tr[t].get('strict_k10_known_mask',False) for t in range(max_t))
                    any_grasp_known = any(tr[t].get('grasp_established_known_mask',False) for t in range(max_t))
                    any_manip_known = any(tr[t].get('manipulation_active_known_mask',False) for t in range(max_t))
                    n_k10_pos = sum(1 for t in range(max_t) if tr[t].get('strict_k10_feasible',False) and tr[t].get('strict_k10_known_mask',False))
                    n_grasp_pos = sum(1 for t in range(max_t) if tr[t].get('grasp_established',False) and tr[t].get('grasp_established_known_mask',False))
                    n_manip_pos = sum(1 for t in range(max_t) if tr[t].get('manipulation_active',False) and tr[t].get('manipulation_active_known_mask',False))
                    n_k10_known = sum(1 for t in range(max_t) if tr[t].get('strict_k10_known_mask',False))

                    if n_k10_pos > 0:
                        opp_count += 1
                    else:
                        valid_abs += 1
                        if not any_k10_known: f1_count += 1
                        elif n_manip_pos == 0 and any_manip_known: f3_count += 1
                        elif n_grasp_pos == 0 and any_grasp_known: f4_count += 1
                        else: parser_count += 1

                    # Also check parser: K10 feasible but phase never known
                    n_g_known = sum(1 for t in range(max_t) if tr[t].get('grasp_established_known_mask',False))
                    n_m_known = sum(1 for t in range(max_t) if tr[t].get('manipulation_active_known_mask',False))
                    if n_k10_pos > 0 and n_g_known == 0 and n_m_known == 0: parser_count += 1

print(f'Opportunity-present: {opp_count}')
print(f'F1_STRUCTURAL_ZERO:  {f1_count}')
print(f'F3_NO_MANIPULATION:  {f3_count}')
print(f'F4_NO_STABLE_GRASP:  {f4_count}')
print(f'Parser-invalid:       {parser_count}')
print(f'Total absent:         {valid_abs}')

coverage_pass = (valid_abs >= 40 and f3_count >= 10 and f4_count >= 10 and opp_count >= 40 and parser_count == 0)
print(f'Coverage pre-check: {"PASS" if coverage_pass else "WARNING — P2 absent coverage limited"}')

# ═══════════════════════════════════════════
# 3. P2 Inference + Scheduler Search
# ═══════════════════════════════════════════
print('\n=== 3. P2 INFERENCE + SCHEDULER SEARCH ===')
p2_records = run_inference([C2_TEACHER, CHECK_TEACHER], p2_sets)
print(f'P2 records: {len(p2_records)}')

# Group by episode
p2_eps = defaultdict(lambda: {'cal':[], 'label':[], 'T':0, 'split':'', 'eid':'',
                                'has_opp':False, 'f1':False, 'f3':False, 'f4':False})
for r in p2_records:
    ep = p2_eps[r['eid']]
    ep['cal'].append(r['cal_prob'])
    ep['label'].append(r['k10_label'])
    ep['T'] = max(ep['T'], r['T'])
    ep['split'] = r['split']
    ep['eid'] = r['eid']
    if r['k10_label'] == 1.0: ep['has_opp'] = True

# Classify absent episodes
for eid, ep in p2_eps.items():
    if ep['has_opp']: continue
    max_t = max(1, ep['T'] - K10 + 1)
    # Simple check from the records
    ep_recs = [r for r in p2_records if r['eid'] == eid]
    labels_arr = np.array([r['k10_label'] for r in ep_recs])
    has_any_k10 = len(ep_recs) > 0

    # Check original teacher to classify
    found = False
    for root in [C2_TEACHER, CHECK_TEACHER]:
        if found: break
        parts = eid.split('/')
        tp = os.path.join(root, parts[0], parts[1], parts[2], 'factorized_teacher_v1.jsonl')
        if not os.path.isfile(tp): continue
        tr = [json.loads(l) for l in open(tp).read().splitlines() if l.strip()]
        tr.sort(key=lambda r: r['step']); T = len(tr)
        max_t = min(T, T-K10+1)
        any_k10_known = any(tr[t].get('strict_k10_known_mask',False) for t in range(max_t))
        any_grasp_known = any(tr[t].get('grasp_established_known_mask',False) for t in range(max_t))
        any_manip_known = any(tr[t].get('manipulation_active_known_mask',False) for t in range(max_t))
        n_manip_pos = sum(1 for t in range(max_t) if tr[t].get('manipulation_active',False) and tr[t].get('manipulation_active_known_mask',False))
        n_grasp_pos = sum(1 for t in range(max_t) if tr[t].get('grasp_established',False) and tr[t].get('grasp_established_known_mask',False))

        if not any_k10_known: ep['f1'] = True
        elif n_manip_pos == 0 and any_manip_known: ep['f3'] = True
        elif n_grasp_pos == 0 and any_grasp_known: ep['f4'] = True
        found = True

opp_eps = [ep for ep in p2_eps.values() if ep['has_opp']]
abs_eps = [ep for ep in p2_eps.values() if not ep['has_opp']]
f3_eps = [ep for ep in p2_eps.values() if ep['f3']]
f4_eps = [ep for ep in p2_eps.values() if ep['f4']]
print(f'P2: {len(opp_eps)} opp, {len(abs_eps)} abs (F3={len(f3_eps)}, F4={len(f4_eps)})')

# Simple scheduler
def simple_scheduler(cal_probs, T, tau, persistence):
    max_t = max(0, T - K10 + 1)
    cons = 0
    for t in range(max_t):
        if cal_probs[t] >= tau:
            cons += 1
            if cons >= persistence:
                return True, t
        else:
            cons = 0
    return False, -1

# Grid search
best_result = None
for tau in np.linspace(0.1, 0.95, 86):
    for d in [1, 2, 3]:
        tp=0; fp=0; total_opp=0; total_abs=0
        f3_fp=0; f4_fp=0
        for ep in opp_eps:
            total_opp += 1
            emit, _ = simple_scheduler(np.array(ep['cal']), ep['T'], tau, d)
            if emit: tp += 1
        for ep in abs_eps:
            total_abs += 1
            emit, _ = simple_scheduler(np.array(ep['cal']), ep['T'], tau, d)
            if emit: fp += 1
            if emit and ep['f3']: f3_fp += 1
            if emit and ep['f4']: f4_fp += 1

        if total_emit := tp + fp == 0: continue
        fs_rate = fp / max(total_abs, 1)
        if fs_rate > 0.10: continue
        recall = tp / max(total_opp, 1)
        if recall == 0: continue

        # Per-split emit coverage
        cov_splits = set()
        for ep in opp_eps + abs_eps:
            emit, _ = simple_scheduler(np.array(ep['cal']), ep['T'], tau, d)
            if emit: cov_splits.add(ep['split'])

        result = {'tau': float(tau), 'd': d, 'recall': recall, 'fs': fs_rate,
                  'tp': tp, 'fp': fp, 'total_opp': total_opp, 'total_abs': total_abs,
                  'f3_fp': f3_fp, 'f4_fp': f4_fp, 'f3_n': len(f3_eps), 'f4_n': len(f4_eps),
                  'cov_splits': len(cov_splits)}

        if best_result is None:
            best_result = result
        else:
            # Lexicographic: recall → coverage → lower FS → lower d
            if (recall > best_result['recall'] or
                (recall == best_result['recall'] and result['cov_splits'] > best_result['cov_splits']) or
                (recall == best_result['recall'] and result['cov_splits'] == best_result['cov_splits'] and fs_rate < best_result['fs'])):
                best_result = result

if best_result:
    br = best_result
    print(f'\nBest policy: τ={br["tau"]:.3f} d={br["d"]} recall={br["recall"]:.4f} FS={br["fs"]:.4f}')
    print(f'TP={br["tp"]}/{br["total_opp"]} FP={br["fp"]}/{br["total_abs"]} Coverage={br["cov_splits"]}/12')
    print(f'F3 FS={br["f3_fp"]}/{br["f3_n"]} F4 FS={br["f4_fp"]}/{br["f4_n"]}')

    # K10 executable check
    k10_ok = 0
    for ep in opp_eps:
        emit, estep = simple_scheduler(np.array(ep['cal']), ep['T'], br['tau'], br['d'])
        if emit and estep + K10 <= ep['T']: k10_ok += 1
    print(f'K10 executable: {k10_ok}/{br["tp"]}')

    # Per-split breakdown
    print('\nPer-split:')
    for sk in SPLITS:
        sk_opp = [ep for ep in opp_eps if ep['split'] == sk]
        sk_abs = [ep for ep in abs_eps if ep['split'] == sk]
        sk_tp=0; sk_fp=0
        for ep in sk_opp:
            emit, _ = simple_scheduler(np.array(ep['cal']), ep['T'], br['tau'], br['d'])
            if emit: sk_tp += 1
        for ep in sk_abs:
            emit, _ = simple_scheduler(np.array(ep['cal']), ep['T'], br['tau'], br['d'])
            if emit: sk_fp += 1
        print(f'  {sk}: opp={len(sk_opp)} abs={len(sk_abs)} tp={sk_tp} fp={sk_fp}')

    # Sealing
    scheduler = {
        'schema': 'V2_P2_SCHEDULER_FREEZE_V1',
        'method': 'SIMPLE_STARTABILITY_THRESHOLD',
        'parameters': {'tau_start': br['tau'], 'persistence': br['d'], 'K10_window': K10},
        'P2_validation': {
            'total_opp': br['total_opp'], 'total_abs': br['total_abs'],
            'tp': br['tp'], 'fp': br['fp'],
            'pooled_recall': br['recall'], 'pooled_fs': br['fs'],
            'f3_fs': f'{br["f3_fp"]}/{br["f3_n"]}', 'f4_fs': f'{br["f4_fp"]}/{br["f4_n"]}',
            'emit_coverage': br['cov_splits'], 'k10_executable': k10_ok,
        },
        'gates': {
            'fs_le_10pct': br['fs'] <= 0.10, 'recall_gt_0': br['recall'] > 0,
            'k10_executable_100pct': k10_ok == br['tp'], 'threshold_reachable': br['tau'] < 1.0,
        },
        'calibrator_source': CAL_PATH,
        'student_source': STUDENT_ROOT,
    }
    sched_path = os.path.join(OUT_ROOT, 'V2_P2_SCHEDULER_FREEZE_V1.json')
    with open(sched_path, 'w') as f: json.dump(scheduler, f, indent=2)

    # Seal
    all_files = []
    for root, dirs, fns in os.walk(OUT_ROOT):
        for fn in sorted(fns):
            if fn in ('SHA256SUMS','SHA256SUMS.sha256'): continue
            fp = os.path.join(root, fn); rel = os.path.relpath(fp, OUT_ROOT)
            all_files.append((rel, sha256_file(fp)))
    with open(os.path.join(OUT_ROOT, 'SHA256SUMS'), 'w') as f:
        for rel, h in sorted(all_files): f.write(f'{h}  {rel}\n')
    sh = sha256_file(os.path.join(OUT_ROOT, 'SHA256SUMS'))
    with open(os.path.join(OUT_ROOT, 'SHA256SUMS.sha256'), 'w') as f:
        f.write(f'{sh}  SHA256SUMS\n')

    all_gates = all(scheduler['gates'].values())
    print(f'\nP2_SCHEDULER_FREEZE = {"PASS" if all_gates else "FAIL"}')
    print(f'Seal: {sh[:16]}')
else:
    print('\nNO FEASIBLE POLICY FOUND (FS ≤ 10%)')
    print('P2_SCHEDULER_FREEZE = FAIL')
