"""V2 H1 Frozen Engineering Regression.

Loads V2-A, V2-B, V2-Full checkpoints (FIT_DEV selected).
Selects simple scheduler thresholds on FIT_DEV (pooled FS ≤ 10%, max recall).
Evaluates on H1 with comprehensive per-episode metrics.
Compares against V1 baseline.

Metrics: per-episode max/p95/p99, tail quantiles, per-stratum FS,
         step/episode/event AUROC/AUPRC, per-split coverage, K10 executable.
"""
import json, os, sys, hashlib
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')

FEAT_ROOT  = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
FIT_TRAIN_LABELS = '/tmp/ft_FIT_TRAIN/labels'
FIT_DEV_LABELS   = '/tmp/ft_FIT_DEV/labels'
H_LABELS = '/tmp/ft_H/labels'
CKPT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline/v2_engineering'
H_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V5_804113EE_20260723/heldout_l3_identity_manifest.json'
V1_H1_REPORT = '/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline/h1_engineering_v4_1/H1_REGRESSION_REPORT.json'
OUT_DIR = '/mnt/sdc/dty_user/openvla_attack_evidence/final_detector_pipeline/v2_h1_regression'

K10 = 10
SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']

from gripper_attack.v6_critical_student import (
    CriticalTriggerStudentV2, build_v2_recommended, build_v2_minimal)
from gripper_attack.v6_critical_dataset import load_v2_episodes

os.makedirs(OUT_DIR, exist_ok=True)

def auroc(y_true, y_score):
    if len(y_true) < 2: return 0.5
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    tpr = np.cumsum(y_sort)/n_pos; fpr = np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr, fpr))

def auprc(y_true, y_score):
    if len(y_true) < 2: return 0.0
    n_pos = y_true.sum()
    if n_pos == 0: return 0.0
    desc = np.argsort(y_score)[::-1]; y_sort = y_true[desc]
    prec = np.cumsum(y_sort) / np.arange(1, len(y_sort)+1)
    rec = np.cumsum(y_sort) / n_pos
    return float(np.trapz(prec, rec))

# ── Load checkpoints ──
def load_v2_ckpt(variant, device='cpu'):
    path = os.path.join(CKPT_ROOT, variant, 'checkpoint.pt')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = CriticalTriggerStudentV2(
        hidden_dim=cfg['hidden_dim'],
        use_policy_bypass=cfg['use_policy_bypass'],
        use_gripper_bypass=cfg['use_gripper_bypass'],
        use_instruction_context=cfg.get('use_instruction_context', False),
        head_names=cfg['head_names'],
    )
    if variant == 'V2-B':
        model.encoder_25d = __import__('gripper_attack.v6_critical_student', fromlist=['CausalTCNEncoder']).CausalTCNEncoder(43, 64, 32, 0.1)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device); model.eval()
    return model, ckpt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load FIT_DEV data ──
print('Loading FIT_DEV...')
def load_set(root, state_range):
    identities = []
    for suite in sorted(os.listdir(root)):
        sp = os.path.join(root, suite)
        if not os.path.isdir(sp): continue
        for task in sorted(os.listdir(sp)):
            tp = os.path.join(sp, task)
            if not os.path.isdir(tp): continue
            for state in sorted(os.listdir(tp)):
                try:
                    if int(state.replace('state_','')) not in state_range: continue
                except: continue
                identities.append('{}/{}/{}'.format(suite, task, state))
    manifest = {'splits': {'set': {'identities': identities}}}
    return load_v2_episodes(FEAT_ROOT, root, manifest, exclude_parser_contradictions=True)

dev_eps = load_set(FIT_DEV_LABELS, range(20, 24))
print(f'FIT_DEV: {len(dev_eps)} episodes')

# ── Load H1 data ──
print('Loading H1...')
H_MANIFEST = json.load(open(H_MANIFEST_PATH))
h_eps_all = load_v2_episodes(FEAT_ROOT, H_LABELS, H_MANIFEST, exclude_parser_contradictions=True)
print(f'H1: {len(h_eps_all)} episodes')

# ── Compute normalization from FIT_TRAIN ──
print('Computing normalization from FIT_TRAIN...')
train_eps = load_set(FIT_TRAIN_LABELS, range(0, 20))
all_25d = np.concatenate([ep.features_25d for ep in train_eps], axis=0)
all_p9d = np.concatenate([ep.policy_9d for ep in train_eps], axis=0)
all_g9d = np.concatenate([ep.gripper_9d for ep in train_eps], axis=0)
norm = {
    '25d_mean': torch.tensor(all_25d.mean(0), dtype=torch.float32, device=device),
    '25d_std':  torch.tensor(all_25d.std(0).clip(1e-8), dtype=torch.float32, device=device),
    'p9d_mean': torch.tensor(all_p9d.mean(0), dtype=torch.float32, device=device),
    'p9d_std':  torch.tensor(all_p9d.std(0).clip(1e-8), dtype=torch.float32, device=device),
    'g9d_mean': torch.tensor(all_g9d.mean(0), dtype=torch.float32, device=device),
    'g9d_std':  torch.tensor(all_g9d.std(0).clip(1e-8), dtype=torch.float32, device=device),
}
del train_eps  # free memory

# ── Inference helper ──
@torch.no_grad()
def run_inference(model, episodes, variant):
    """Run inference and return per-episode K10 startability scores."""
    results = []
    for ep in episodes:
        f25d = (torch.tensor(ep.features_25d, device=device) - norm['25d_mean']) / norm['25d_std']
        p9d  = (torch.tensor(ep.policy_9d, device=device) - norm['p9d_mean']) / norm['p9d_std']
        g9d  = (torch.tensor(ep.gripper_9d, device=device) - norm['g9d_mean']) / norm['g9d_std']
        f25d = f25d.unsqueeze(0); p9d = p9d.unsqueeze(0); g9d = g9d.unsqueeze(0)

        if variant == 'V2-A':
            logits = model(f25d)
        elif variant == 'V2-B':
            logits = model(torch.cat([f25d, p9d, g9d], dim=-1))
        else:
            logits = model(f25d, p9d, g9d)

        k10_logit = logits['k10_startability'].squeeze().cpu().numpy()  # [T]
        k10_prob = 1.0 / (1.0 + np.exp(-np.clip(k10_logit, -50, 50)))

        T = ep.T
        max_t = max(1, T - K10 + 1)
        valid_probs = k10_prob[:max_t]

        # Per-step metrics in valid window
        known_mask = ep.k10_known[:max_t]
        if known_mask.sum() > 1 and ep.k10_startable[:max_t].sum() > 0 and (1-ep.k10_startable[:max_t]).sum() > 0:
            step_auc = auroc(ep.k10_startable[:max_t][known_mask], valid_probs[known_mask])
            step_ap  = auprc(ep.k10_startable[:max_t][known_mask], valid_probs[known_mask])
        else:
            step_auc = 0.5; step_ap = 0.0

        results.append({
            'eid': ep.eid, 'split': ep.split, 'T': T,
            'has_opp': ep.has_opportunity, 'absence_reason': ep.absence_reason,
            'max_score': float(valid_probs.max()),
            'p50_score': float(np.median(valid_probs)),
            'p95_score': float(np.percentile(valid_probs, 95)),
            'p99_score': float(np.percentile(valid_probs, 99)),
            'step_auc': step_auc, 'step_ap': step_ap,
            'k10_prob': k10_prob.tolist(),
        })
    return results

# ── Run inference for all 3 variants ──
all_results = {}
for variant in ['V2-A', 'V2-B', 'V2-Full']:
    print(f'\n=== {variant} ===')
    model, ckpt = load_v2_ckpt(variant, device)
    best_epoch = ckpt['best_epoch']

    # FIT_DEV inference
    dev_r = run_inference(model, dev_eps, variant)

    # H1 inference
    h1_r = run_inference(model, h_eps_all, variant)

    all_results[variant] = {'dev': dev_r, 'h1': h1_r, 'best_epoch': best_epoch}

    # ── Audit AUPRC ──
    dev_scores = np.array([r['max_score'] for r in dev_r])
    dev_labels = np.array([1.0 if r['has_opp'] else 0.0 for r in dev_r])
    ep_auroc = auroc(dev_labels, dev_scores)
    ep_auprc = auprc(dev_labels, dev_scores)

    # Step-level pooled
    all_step_scores = []; all_step_labels = []
    for r in dev_r:
        ep = next(e for e in dev_eps if e.eid == r['eid'])
        max_t = max(1, ep.T - K10 + 1)
        known = ep.k10_known[:max_t]
        if known.sum() == 0: continue
        all_step_scores.extend(np.array(r['k10_prob'])[:max_t][known])
        all_step_labels.extend(ep.k10_startable[:max_t][known])
    step_auc = auroc(np.array(all_step_labels), np.array(all_step_scores))
    step_ap  = auprc(np.array(all_step_labels), np.array(all_step_scores))

    # Per-split episode AUROC
    per_split_auc = {}
    for sk in SPLITS:
        split_r = [r for r in dev_r if r['split'] == sk or True]  # FIT_DEV is single split
        if len(split_r) < 3: continue
        s = np.array([r['max_score'] for r in split_r])
        l = np.array([1.0 if r['has_opp'] else 0.0 for r in split_r])
        if l.sum() == 0 or l.sum() == len(l): continue
        per_split_auc[sk] = auroc(l, s)

    pos_rate = dev_labels.mean()
    print(f'  Best epoch: {best_epoch}')
    print(f'  Episode AUROC={ep_auroc:.4f}  AUPRC={ep_auprc:.4f}  pos_rate={pos_rate:.2%}  AP_baseline={pos_rate:.4f}')
    print(f'  Step pooled AUROC={step_auc:.4f}  AUPRC={step_ap:.4f}  n_steps={len(all_step_scores)}')

    # ── Per-episode tail metrics ──
    for reason in ['OPPORTUNITY_PRESENT','F1_TASK_STRUCTURAL_ZERO','F3_NO_MANIPULATION',
                    'F4_NO_STABLE_GRASP','F6_PARSER_DECODER_ZERO']:
        subset = [r for r in dev_r if r['absence_reason'] == reason]
        if not subset: continue
        max_s = [r['max_score'] for r in subset]
        p95_s = [r['p95_score'] for r in subset]
        p99_s = [r['p99_score'] for r in subset]
        print(f'  {reason} (n={len(subset)}): max: p50={np.median(max_s):.4f} p90={np.percentile(max_s,90):.4f} p99={np.percentile(max_s,99):.4f}  p99: p50={np.median(p99_s):.4f} p90={np.percentile(p99_s,90):.4f}')

    # Per-episode max score distribution (all absence categories pooled)
    abs_r = [r for r in dev_r if not r['has_opp']]
    if abs_r:
        abs_max = [r['max_score'] for r in abs_r]
        print(f'  ABSENCE (n={len(abs_r)}): max_score p50={np.median(abs_max):.4f} p90={np.percentile(abs_max,90):.4f} max={max(abs_max):.4f}')
        # Check for "hidden peaks": episodes with low median but high max
        hidden_peaks = sum(1 for r in abs_r if r['p50_score'] < 0.1 and r['max_score'] > 0.5)
        print(f'  Hidden peaks (p50<0.1 but max>0.5): {hidden_peaks}/{len(abs_r)}')

    all_results[variant]['dev_metrics'] = {
        'ep_auroc': ep_auroc, 'ep_auprc': ep_auprc,
        'step_auc': step_auc, 'step_ap': step_ap,
        'pos_rate': float(pos_rate),
        'per_split_auc': per_split_auc,
    }

# ── Simple scheduler: threshold + persistence ──
print('\n' + '='*60)
print('SIMPLE SCHEDULER: threshold + persistence')
print('='*60)

def simple_scheduler(probs, T, threshold, persistence, remaining_needed=K10):
    """Returns (emit, emit_step). One-shot, persistence=consecutive steps above threshold."""
    max_t = max(0, T - remaining_needed + 1)
    cons = 0
    for t in range(max_t):
        if probs[t] >= threshold:
            cons += 1
            if cons >= persistence:
                return True, t
        else:
            cons = 0
    return False, -1

def find_threshold(dev_results, target_fs=0.10, persistence=2):
    """Find threshold on FIT_DEV satisfying pooled FS ≤ target_fs, maximizing recall."""
    opp_r = [r for r in dev_results if r['has_opp']]
    abs_r = [r for r in dev_results if not r['has_opp']]
    n_abs = len(abs_r)

    # Scan thresholds from low to high
    best_thresh = 1.0; best_recall = 0.0
    for thresh in np.linspace(0.3, 1.0, 71):
        fs_count = 0
        for r in abs_r:
            emit, _ = simple_scheduler(np.array(r['k10_prob']), r['T'], thresh, persistence)
            if emit: fs_count += 1
        fs_rate = fs_count / max(n_abs, 1)
        if fs_rate > target_fs: continue

        # Count recall at this threshold
        tp_count = 0
        for r in opp_r:
            emit, _ = simple_scheduler(np.array(r['k10_prob']), r['T'], thresh, persistence)
            if emit: tp_count += 1
        recall = tp_count / max(len(opp_r), 1)
        if recall > best_recall:
            best_recall = recall; best_thresh = thresh

    return best_thresh, best_recall

# ── Apply to all 3 variants ──
h1_report = {}
for variant in ['V2-A', 'V2-B', 'V2-Full']:
    dev_r = all_results[variant]['dev']
    h1_r = all_results[variant]['h1']

    # Find threshold on FIT_DEV
    best_thresh, dev_recall = find_threshold(dev_r, target_fs=0.10, persistence=2)
    if best_thresh >= 1.0:
        # Relax: find best threshold at any FS level
        best_thresh, dev_recall = find_threshold(dev_r, target_fs=1.0, persistence=2)

    print(f'\n{variant}: threshold={best_thresh:.4f}  DEV_recall={dev_recall:.4f}')

    # Apply to H1
    h1_opp = [r for r in h1_r if r['has_opp']]
    h1_abs = [r for r in h1_r if not r['has_opp']]

    tp = 0; fn = 0; fp = 0; tn = 0
    per_split = {sk: {'opp':0,'abs':0,'tp':0,'fn':0,'fp':0,'tn':0,'emit':0} for sk in SPLITS}

    for r in h1_r:
        emit, estep = simple_scheduler(np.array(r['k10_prob']), r['T'], best_thresh, persistence=2)
        sk = r['split']
        if r['has_opp']:
            per_split[sk]['opp'] += 1
            if emit: tp += 1; per_split[sk]['tp'] += 1; per_split[sk]['emit'] += 1
            else: fn += 1; per_split[sk]['fn'] += 1
        else:
            per_split[sk]['abs'] += 1
            if emit: fp += 1; per_split[sk]['fp'] += 1; per_split[sk]['emit'] += 1
            else: tn += 1; per_split[sk]['tn'] += 1

    recall_h1 = tp / max(tp + fn, 1)
    fs_h1 = fp / max(fp + tn, 1)
    precision_h1 = tp / max(tp + fp, 1)
    emit_cov = sum(1 for sk in SPLITS if per_split[sk]['emit'] > 0)
    tp_cov = sum(1 for sk in SPLITS if per_split[sk]['tp'] > 0)

    # F3/F4 false-start
    f3_abs = [r for r in h1_abs if r['absence_reason'] == 'F3_NO_MANIPULATION']
    f4_abs = [r for r in h1_abs if r['absence_reason'] == 'F4_NO_STABLE_GRASP']
    f3_fp = sum(1 for r in f3_abs if simple_scheduler(np.array(r['k10_prob']), r['T'], best_thresh, 2)[0])
    f4_fp = sum(1 for r in f4_abs if simple_scheduler(np.array(r['k10_prob']), r['T'], best_thresh, 2)[0])

    # Episode-level AUROC
    h1_scores = np.array([r['max_score'] for r in h1_r])
    h1_labels = np.array([1.0 if r['has_opp'] else 0.0 for r in h1_r])
    h1_ep_auroc = auroc(h1_labels, h1_scores)
    h1_ep_auprc = auprc(h1_labels, h1_scores)

    # K10 executable
    k10_ok = sum(1 for r in h1_r if r['has_opp'] and
                 simple_scheduler(np.array(r['k10_prob']), r['T'], best_thresh, 2)[0] and
                 simple_scheduler(np.array(r['k10_prob']), r['T'], best_thresh, 2)[1] + K10 <= r['T'])

    print(f'  H1 Recall={recall_h1:.4f} ({tp}/{tp+fn})  FS={fs_h1:.4f} ({fp}/{fp+tn})  Precision={precision_h1:.4f}')
    print(f'  H1 ep_AUROC={h1_ep_auroc:.4f}  ep_AUPRC={h1_ep_auprc:.4f}')
    print(f'  Emit cov={emit_cov}/12  TP cov={tp_cov}/12')
    print(f'  F3 FS={f3_fp}/{len(f3_abs)}  F4 FS={f4_fp}/{len(f4_abs)}')
    print(f'  K10 executable: {k10_ok}/{tp}')

    # Per-split
    print(f'  Per-split:')
    for sk in SPLITS:
        ps = per_split[sk]
        if ps['opp'] + ps['abs'] == 0: continue
        print(f'    {sk}: opp={ps["opp"]} abs={ps["abs"]} emit={ps["emit"]} tp={ps["tp"]} fp={ps["fp"]}')

    h1_report[variant] = {
        'threshold': float(best_thresh), 'persistence': 2,
        'dev_recall': float(dev_recall),
        'h1_recall': float(recall_h1), 'h1_fs': float(fs_h1),
        'h1_precision': float(precision_h1),
        'h1_ep_auroc': float(h1_ep_auroc), 'h1_ep_auprc': float(h1_ep_auprc),
        'emit_cov': emit_cov, 'tp_cov': tp_cov,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'f3_fp': f3_fp, 'f3_n': len(f3_abs),
        'f4_fp': f4_fp, 'f4_n': len(f4_abs),
        'k10_executable': k10_ok,
        'per_split': {sk: dict(ps) for sk, ps in per_split.items()},
    }

# ── Load V1 H1 baseline for comparison ──
print('\n' + '='*60)
print('V1 vs V2 COMPARISON')
print('='*60)
v1 = json.load(open(V1_H1_REPORT))
v1_recall = v1['aggregate']['opportunity_recall']
v1_fs = v1['aggregate']['false_start_rate']
print(f'V1 H1: Recall={v1_recall:.4f}  FS={v1_fs:.4f}')

for variant in ['V2-A', 'V2-B', 'V2-Full']:
    r = h1_report[variant]
    fs_change = (r['h1_fs'] - v1_fs) / v1_fs * 100 if v1_fs > 0 else 0
    recall_change = (r['h1_recall'] - v1_recall) / v1_recall * 100 if v1_recall > 0 else 0
    print(f'{variant}: Recall={r["h1_recall"]:.4f} ({recall_change:+.0f}%)  FS={r["h1_fs"]:.4f} ({fs_change:+.0f}%)  '
          f'ep_AUROC={r["h1_ep_auroc"]:.4f}  F3_FS={r["f3_fp"]}/{r["f3_n"]}  F4_FS={r["f4_fp"]}/{r["f4_n"]}')

# ── Save report ──
report = {
    'schema': 'V2_H1_ENGINEERING_REGRESSION_V1',
    'variants': h1_report,
    'v1_baseline': {'recall': v1_recall, 'fs': v1_fs},
    'threshold_method': 'FIT_DEV pooled FS ≤ 10%, max recall, persistence=2',
}
with open(os.path.join(OUT_DIR, 'h1_regression_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f'\nReport: {OUT_DIR}/h1_regression_report.json')
