"""V2.1 Loss Ablation: grouped state-block CV on DEV 1300.

E0: step BCE baseline
E1: + episode smooth-max penalty on absent episodes
E2: + F3/F4 top-k negative penalty
E3: full (E1 + E2)

Key: worst-block episode AUROC, recall at FPR<=10%, F3/F4 false-start.
"""
import json, os, sys, hashlib, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
DEV_LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels']
NEW_LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels'
DEV_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_DEV_IDENTITY_MANIFEST_V2.json'
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/v21_ablation'

K10 = 10; SEED = 42; BATCH = 4; HIDDEN = 64; RF = 32; DROPOUT = 0.1
LR = 1e-3; WD = 1e-4; EPOCHS = 20
TOP_K_NEG = 10  # top-K absent steps for hard-negative penalty

sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v6_critical_student import CriticalTriggerStudentV2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_ROOT, exist_ok=True)

def auroc(y_true, y_score):
    if len(y_true)<2: return 0.5
    n_pos=y_true.sum(); n_neg=len(y_true)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(y_score)[::-1]; y_sort=y_true[desc]
    tpr=np.cumsum(y_sort)/n_pos; fpr=np.cumsum(1-y_sort)/n_neg
    return float(np.trapz(tpr,fpr))

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))

# ── Load DEV 1300 identities ──
dev_manifest = json.load(open(DEV_MANIFEST))
dev_id_set = set(dev_manifest['identities'])
print('DEV identities: {}'.format(len(dev_id_set)))

# ── Load episodes ──
def load_ep(label_roots, id_set):
    eps = []
    for root in label_roots:
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
                    label_path = os.path.join(tp, state, 'factorized_teacher_v1.jsonl')
                    feat_path = os.path.join(FEAT_ROOT, suite, task, state, 'step_records.jsonl')
                    if not os.path.isfile(label_path) or not os.path.isfile(feat_path): continue
                    recs = [json.loads(l) for l in open(feat_path).read().splitlines() if l.strip()]
                    labels = [json.loads(l) for l in open(label_path).read().splitlines() if l.strip()]
                    labels.sort(key=lambda r: r['step']); T = len(recs); max_t = min(T, T-K10+1)
                    f25d = np.array([r['features_25d'] for r in recs], dtype=np.float32)
                    p9d = np.array([r.get('clean_policy_intent_9d', np.zeros(9)) for r in recs], dtype=np.float32)
                    g9d = np.array([[r.get('clean_close_probability_mass',0),r.get('clean_open_probability_mass',0),
                        r.get('clean_top1_is_close',0),r.get('clean_top1_is_open',0),r.get('clean_top1_probability',0),
                        r.get('clean_best_close_rank_normalized',0),r.get('clean_best_open_rank_normalized',0),
                        r.get('clean_action_token_entropy_normalized',0),r.get('clean_open_minus_close_log_mass',0)]
                        for r in recs], dtype=np.float32)
                    k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                    g_l = np.array([labels[min(t,len(labels)-1)].get('grasp_established',False) for t in range(T)], dtype=bool)
                    m_l = np.array([labels[min(t,len(labels)-1)].get('manipulation_active',False) for t in range(T)], dtype=bool)
                    cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                    has_opp = bool(k10_s[:max_t].any())
                    eps.append({'eid':eid,'suite':suite,'task':task,'state':state,'T':T,
                                'f25d':f25d,'p9d':p9d,'g9d':g9d,
                                'k10_s':k10_s,'k10_k':k10_k,'g_l':g_l,'m_l':m_l,'cc':cc,
                                'has_opp':has_opp,'max_t':max_t})
    return eps

all_eps = load_ep(DEV_LABEL_ROOTS + [NEW_LABEL_ROOT], dev_id_set)
print('Loaded {} episodes'.format(len(all_eps)))

# ── Suite-level CV folds (4 folds: leave-one-suite-out) ──
SUITES = ['libero_10','libero_goal','libero_object','libero_spatial']
suite_folds = {s: [ep for ep in all_eps if ep['suite']==s] for s in SUITES}
print('Suite folds: {}'.format({s: len(v) for s,v in suite_folds.items()}))

# ── Model builder ──
def build_model():
    return CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=HIDDEN, receptive_field=RF,
        dropout=DROPOUT, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability'])

# ── Train/eval one fold ──
def run_fold(train_eps, val_eps, variant):
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Normalization from train
    cat_25d = np.concatenate([ep['f25d'] for ep in train_eps], axis=0)
    cat_p9d = np.concatenate([ep['p9d'] for ep in train_eps], axis=0)
    cat_g9d = np.concatenate([ep['g9d'] for ep in train_eps], axis=0)
    n25d_m = torch.tensor(cat_25d.mean(0), device=device); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=device)
    np9d_m = torch.tensor(cat_p9d.mean(0), device=device); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=device)
    ng9d_m = torch.tensor(cat_g9d.mean(0), device=device); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=device)

    best_auroc = -1.0; best_state = None; history = []

    for epoch in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; n_batches = 0
        random.shuffle(train_eps)
        for i in range(0, len(train_eps), BATCH):
            batch = train_eps[i:i+BATCH]
            max_T = max(ep['T'] for ep in batch)
            B = len(batch)

            # Build tensors
            x_cat = torch.zeros(B, max_T, 43, device=device)
            k10_s = torch.zeros(B, max_T, 1, device=device)
            k10_k = torch.zeros(B, max_T, 1, device=device)
            cc = torch.zeros(B, max_T, 1, device=device)
            has_opp_flags = torch.zeros(B, device=device)

            for b, ep in enumerate(batch):
                T = ep['T']
                f25d_n = (torch.tensor(ep['f25d'], device=device) - n25d_m) / n25d_s
                p9d_n = (torch.tensor(ep['p9d'], device=device) - np9d_m) / np9d_s
                g9d_n = (torch.tensor(ep['g9d'], device=device) - ng9d_m) / ng9d_s
                x_cat[b,:T] = torch.cat([f25d_n, p9d_n, g9d_n], dim=-1)
                k10_s[b,:T,0] = torch.tensor(ep['k10_s'], device=device).float()
                k10_k[b,:T,0] = torch.tensor(ep['k10_k'], device=device).float()
                cc[b,:T,0] = torch.tensor(ep['cc'], device=device).float()
                has_opp_flags[b] = 1.0 if ep['has_opp'] else 0.0

            logits = model(x_cat)
            logit = logits['k10_startability']  # [B, max_T, 1]

            # E0: step BCE
            bce = nn.functional.binary_cross_entropy_with_logits(logit, k10_s, reduction='none')
            loss_step = (bce * k10_k).sum() / k10_k.sum().clamp(min=1)

            loss = loss_step

            # E1/E3: episode smooth-max penalty on absent episodes
            if variant in ('E1','E3'):
                abs_mask = (has_opp_flags == 0)
                if abs_mask.any():
                    abs_logits = logit[abs_mask]  # [n_abs, max_T, 1]
                    # Smooth max via logsumexp over candidate-close steps
                    cc_mask = cc[abs_mask]  # [n_abs, max_T, 1]
                    # Only penalize steps where candidate_close is true
                    masked = abs_logits - (1-cc_mask) * 1e9  # large negative for non-cc steps
                    smooth_max = torch.logsumexp(masked, dim=1)  # [n_abs, 1]
                    loss_ep = smooth_max.mean() * 0.1  # soft penalty weight
                    loss = loss + loss_ep

            # E2/E3: F3/F4 top-k negative penalty
            if variant in ('E2','E3'):
                abs_mask = (has_opp_flags == 0) & (k10_k.sum(dim=[1,2]) > 0)  # has known steps
                if abs_mask.any():
                    abs_logits = logit[abs_mask]  # [n_abs, max_T, 1]
                    # Take top-K scores per episode where k10_known is true
                    abs_k_mask = k10_k[abs_mask]  # [n_abs, max_T, 1]
                    abs_logits_masked = abs_logits - (1-abs_k_mask) * 1e9
                    top_k_vals, _ = torch.topk(abs_logits_masked.squeeze(-1), k=min(TOP_K_NEG, max_T), dim=1)
                    loss_hard = torch.sigmoid(top_k_vals).mean() * 0.5  # push top-K toward 0
                    loss = loss + loss_hard

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        # Evaluate on val
        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for ep in val_eps:
                f25d_n = (torch.tensor(ep['f25d'], device=device) - n25d_m) / n25d_s
                p9d_n = (torch.tensor(ep['p9d'], device=device) - np9d_m) / np9d_s
                g9d_n = (torch.tensor(ep['g9d'], device=device) - ng9d_m) / ng9d_s
                x_cat = torch.cat([f25d_n, p9d_n, g9d_n], dim=-1).unsqueeze(0)
                out = model(x_cat)
                prob = torch.sigmoid(out['k10_startability']).squeeze().cpu().numpy()
                val_scores.append(float(prob[:ep['max_t']].max()))
                val_labels.append(1.0 if ep['has_opp'] else 0.0)

        val_labels_a = np.array(val_labels); val_scores_a = np.array(val_scores)
        ep_auc = auroc(val_labels_a, val_scores_a)
        history.append({'epoch':epoch,'loss':total_loss/max(n_batches,1),'ep_auc':ep_auc})
        if ep_auc > best_auroc:
            best_auroc = ep_auc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {'best_ep_auc': best_auroc, 'history': history, 'n_train': len(train_eps), 'n_val': len(val_eps)}

# ── Run all folds for one variant ──
def run_variant(variant):
    print('\n=== {} ==='.format(variant))
    fold_results = []
    for test_suite in SUITES:
        train_eps = []
        for s in SUITES:
            if s != test_suite:
                train_eps.extend(suite_folds[s])
        val_eps = suite_folds[test_suite]
        if len(val_eps) < 3: continue

        r = run_fold(train_eps, val_eps, variant)
        fold_results.append(r)
        print('  {}: n_train={} n_val={} ep_auc={:.4f}'.format(
            test_suite, r['n_train'], r['n_val'], r['best_ep_auc']))

    aucs = [r['best_ep_auc'] for r in fold_results]
    print('{}: mean_auc={:.4f} median={:.4f} min={:.4f} max={:.4f}'.format(
        variant, np.mean(aucs), np.median(aucs), min(aucs), max(aucs)))
    return {'variant': variant, 'fold_results': fold_results, 'mean_auc': float(np.mean(aucs)),
            'median_auc': float(np.median(aucs)), 'min_auc': float(min(aucs)), 'max_auc': float(max(aucs))}

# ── Main ──
parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='E0', choices=['E0','E1','E2','E3'])
parser.add_argument('--all', action='store_true')
args = parser.parse_args()

if args.all:
    results = {}
    for v in ['E0','E1','E2','E3']:
        results[v] = run_variant(v)

    print('\n=== SUMMARY ===')
    for v in ['E0','E1','E2','E3']:
        r = results[v]
        print('{}: mean={:.4f} median={:.4f} min={:.4f}'.format(v, r['mean_auc'], r['median_auc'], r['min_auc']))

    with open(os.path.join(OUT_ROOT, 'ablation_results.json'), 'w') as f:
        json.dump({v: {k: r[k] for k in ['mean_auc','median_auc','min_auc','max_auc']} for v,r in results.items()}, f, indent=2)
else:
    run_variant(args.variant)

print('Done.')
