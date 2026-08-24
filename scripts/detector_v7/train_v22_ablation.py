"""V2.2 Ablation: stratum-balanced sampling + pairwise ranking + within-H head.

A0: baseline (same as V2.1 E0)
A1: + stratum-balanced episode sampling
A2: + pairwise hard-negative ranking loss
A3: + within-H auxiliary head
A4: full (A1+A2+A3)

Evaluation: leave-one-suite-out + inner threshold + outer one-shot replay.
"""
import json, os, sys, hashlib, random, numpy as np, torch, torch.nn as nn
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
DEV2_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/v22_ablation'
os.makedirs(OUT_ROOT, exist_ok=True)

K10=10; SEED=42; EPOCHS=20; LR=1e-3; WD=1e-4; BATCH=4
WITHIN_H=5; RANK_MARGIN=2.0; RANK_LAMBDA=0.1

sys.path.insert(0,'/mnt/sdc/dty_user/openvla_attack/src')
from gripper_attack.v7_localization_student import LocalizationStudentV22

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev2_ids = set(json.load(open(DEV2_MANIFEST))['identities'])
print('DEV2: {}'.format(len(dev2_ids)))

def auroc(yt,ys):
    if len(yt)<2: return 0.5
    n_pos=yt.sum();n_neg=len(yt)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    tpr=np.cumsum(ysort)/n_pos;fpr=np.cumsum(1-ysort)/n_neg
    return float(np.trapz(tpr,fpr))

# ── Load DEV2 episodes ──
all_eps = {}
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
                if eid not in dev2_ids: continue
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
                k10_s = np.array([labels[min(t,len(labels)-1)].get('strict_k10_feasible',False) and labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                k10_k = np.array([labels[min(t,len(labels)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                cc = np.array([labels[min(t,len(labels)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                has_opp = bool(k10_s[:max_t].any())

                # Within-H target: any K10 feasible in next H steps
                within_h = np.zeros(T, dtype=bool)
                for t in range(max_t):
                    end = min(T, t+WITHIN_H+1)
                    within_h[t] = bool(k10_s[t:end].any() and k10_k[t:end].any())

                # Absence reason
                absence_reason = 'OPPORTUNITY_PRESENT'
                if not has_opp:
                    any_k10_known = any(k10_k[:max_t])
                    any_manip_known = any(labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False) for t in range(max_t))
                    any_grasp_known = any(labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False) for t in range(max_t))
                    n_manip_pos = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('manipulation_active',False) and labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False))
                    n_grasp_pos = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('grasp_established',False) and labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False))
                    if not any_k10_known: absence_reason = 'F1_STRUCTURAL_ZERO'
                    elif n_manip_pos == 0 and any_manip_known: absence_reason = 'F3_NO_MANIPULATION'
                    elif n_grasp_pos == 0 and any_grasp_known: absence_reason = 'F4_NO_STABLE_GRASP'
                    else: absence_reason = 'OTHER_ABSENT'

                all_eps[eid] = {'eid':eid,'T':T,'max_t':max_t,'f25d':f25d,'p9d':p9d,'g9d':g9d,
                    'k10_s':k10_s,'k10_k':k10_k,'cc':cc,'has_opp':has_opp,
                    'absence_reason':absence_reason,'suite':suite,'within_h':within_h}

print('Loaded {} episodes'.format(len(all_eps)))
ep_list = list(all_eps.values())

# Stratum groups for balanced sampling
opp_eps = [e for e in ep_list if e['has_opp']]
f3_eps = [e for e in ep_list if e['absence_reason']=='F3_NO_MANIPULATION']
f4_eps = [e for e in ep_list if e['absence_reason']=='F4_NO_STABLE_GRASP']
other_abs = [e for e in ep_list if not e['has_opp'] and e['absence_reason'] not in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP')]
print('Strata: opp={} F3={} F4={} other_abs={}'.format(len(opp_eps),len(f3_eps),len(f4_eps),len(other_abs)))

SUITES = sorted(set(e['suite'] for e in ep_list))

# ── Train one fold ──
def run_fold(train_eps, val_eps, variant):
    use_stratum = variant in ('A1','A2','A3','A4')
    use_rank = variant in ('A2','A4')
    use_within = variant in ('A3','A4')

    model = LocalizationStudentV22(use_within_h=use_within, use_onset_band=False, within_h=WITHIN_H).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # Normalization from train
    cat_25d = np.concatenate([e['f25d'] for e in train_eps], axis=0)
    cat_p9d = np.concatenate([e['p9d'] for e in train_eps], axis=0)
    cat_g9d = np.concatenate([e['g9d'] for e in train_eps], axis=0)
    n25d_m = torch.tensor(cat_25d.mean(0), device=device); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=device)
    np9d_m = torch.tensor(cat_p9d.mean(0), device=device); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=device)
    ng9d_m = torch.tensor(cat_g9d.mean(0), device=device); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=device)

    # Stratum groups for balanced sampling
    t_opp = [e for e in train_eps if e['has_opp']]
    t_f3 = [e for e in train_eps if e['absence_reason']=='F3_NO_MANIPULATION']
    t_f4 = [e for e in train_eps if e['absence_reason']=='F4_NO_STABLE_GRASP']
    t_other = [e for e in train_eps if not e['has_opp'] and e['absence_reason'] not in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP')]

    best_auc = -1.0; best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; n_batches = 0
        random.shuffle(t_opp); random.shuffle(t_f3); random.shuffle(t_f4); random.shuffle(t_other)

        n_batches_est = max(len(t_opp), len(t_f3), len(t_f4)) // max(BATCH//4, 1) + 1
        for bi in range(n_batches_est):
            # Stratum-balanced: pick 1 opp + 1 F3 + 1 F4 + 1 other per batch (if available)
            batch = []
            if use_stratum:
                batch.append(t_opp[bi % len(t_opp)])
                if t_f3: batch.append(t_f3[bi % len(t_f3)])
                if t_f4: batch.append(t_f4[bi % len(t_f4)])
                if t_other: batch.append(t_other[bi % len(t_other)])
            else:
                idx = random.sample(range(len(train_eps)), min(BATCH, len(train_eps)))
                batch = [train_eps[i] for i in idx]

            max_T = max(e['T'] for e in batch); B = len(batch)
            x_cat = torch.zeros(B, max_T, 43, device=device)
            k10_s = torch.zeros(B, max_T, 1, device=device); k10_k = torch.zeros(B, max_T, 1, device=device)
            cc = torch.zeros(B, max_T, 1, device=device)
            within_labels = torch.zeros(B, max_T, 1, device=device)
            has_opp_flags = torch.zeros(B, device=device)

            for b, e in enumerate(batch):
                T = e['T']
                x_cat[b,:T] = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1)
                k10_s[b,:T,0] = torch.tensor(e['k10_s'],device=device).float()
                k10_k[b,:T,0] = torch.tensor(e['k10_k'],device=device).float()
                cc[b,:T,0] = torch.tensor(e['cc'],device=device).float()
                within_labels[b,:T,0] = torch.tensor(e['within_h'],device=device).float()
                has_opp_flags[b] = 1.0 if e['has_opp'] else 0.0

            out = model(x_cat)
            logit_now = out['k10_now']

            # Step BCE (primary loss)
            bce = nn.functional.binary_cross_entropy_with_logits(logit_now, k10_s, reduction='none')
            loss = (bce * k10_k).sum() / k10_k.sum().clamp(min=1)

            # Within-H auxiliary loss
            if use_within and 'within_H' in out:
                within_logit = out['within_H']
                w_bce = nn.functional.binary_cross_entropy_with_logits(within_logit, within_labels.float(), reduction='none')
                loss = loss + 0.3 * (w_bce * k10_k).sum() / k10_k.sum().clamp(min=1)

            # Pairwise ranking loss: margin between opp corridor max and F3/F4 max
            if use_rank:
                rank_loss = 0.0; n_pairs = 0
                for b in range(B):
                    if has_opp_flags[b] < 0.5: continue
                    Tb = batch[b]['T']; max_tb = batch[b]['max_t']
                    cc_b = batch[b]['cc'][:max_tb]; ks_b = batch[b]['k10_s'][:max_tb]
                    kk_b = batch[b]['k10_k'][:max_tb]
                    sc = logit_now[b,:max_tb,0]
                    # Positive: max score within feasible corridor
                    pos_mask = ks_b & kk_b & cc_b
                    if not pos_mask.any(): continue
                    pos_max = sc[pos_mask].max()
                    # Negative: max score in F3/F4 absent episodes within this batch
                    for bb in range(B):
                        if has_opp_flags[bb] > 0.5: continue  # skip opp
                        reason = batch[bb]['absence_reason']
                        if reason not in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP'): continue
                        Tbb = batch[bb]['T']; max_tbb = batch[bb]['max_t']
                        cc_bb = batch[bb]['cc'][:max_tbb]
                        sc_neg = logit_now[bb,:max_tbb,0]
                        if cc_bb.any():
                            neg_max = sc_neg[cc_bb].max()
                            rank_loss += torch.clamp(RANK_MARGIN - pos_max + neg_max, min=0)
                            n_pairs += 1
                if n_pairs > 0:
                    loss = loss + RANK_LAMBDA * rank_loss / n_pairs

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        # Evaluate on val: episode-max AUROC + first-crossing one-shot replay
        model.eval()
        val_opp = []; val_abs = []
        with torch.no_grad():
            for e in val_eps:
                x_cat = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1).unsqueeze(0)
                out = model(x_cat)
                sc = out['k10_now'].squeeze().cpu().numpy()
                ep_sc = float(sc[:e['max_t']][e['cc'][:e['max_t']]].max()) if e['cc'][:e['max_t']].any() else float(sc[:e['max_t']].max())
                if e['has_opp']: val_opp.append(ep_sc)
                else: val_abs.append(ep_sc)

        if val_opp and val_abs:
            all_s = np.array(val_opp + val_abs); all_l = np.array([1.]*len(val_opp)+[0.]*len(val_abs))
            ep_auc = auroc(all_l, all_s)
        else: ep_auc = 0.5

        if ep_auc > best_auc:
            best_auc = ep_auc; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── Inner threshold + one-shot replay ──
    # Pre-compute scores for all val episodes (avoid re-running inference per threshold)
    val_scores = {}
    with torch.no_grad():
        for e in val_eps:
            x_cat = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1).unsqueeze(0)
            val_scores[e['eid']] = model(x_cat)['k10_now'].squeeze().cpu().numpy()

    best_recall = -1; best_tau = 0; best_d = 1
    val_opp_eps = [e for e in val_eps if e['has_opp']]
    val_abs_eps = [e for e in val_eps if not e['has_opp']]
    val_f3_eps = [e for e in val_abs_eps if e['absence_reason']=='F3_NO_MANIPULATION']
    val_f4_eps = [e for e in val_abs_eps if e['absence_reason']=='F4_NO_STABLE_GRASP']

    for tau in np.linspace(-5, 10, 31):
        for d in [1,2]:
            vt=0; fs=0; fs_f3=0; fs_f4=0
            for e in val_opp_eps:
                sc = val_scores[e['eid']]
                cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t] >= tau:
                        cons+=1
                        if cons>=d:
                            if e['k10_s'][t] and e['k10_k'][t]: vt+=1
                            break
                    else: cons=0
            for e in val_abs_eps:
                sc = val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t] >= tau:
                        cons+=1
                        if cons>=d: fs+=1; break
                    else: cons=0
            for e in val_f3_eps:
                sc = val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t] >= tau:
                        cons+=1
                        if cons>=d: fs_f3+=1; break
                    else: cons=0
            for e in val_f4_eps:
                sc = val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t] >= tau:
                        cons+=1
                        if cons>=d: fs_f4+=1; break
                    else: cons=0

                fs_all = fs / max(len(val_abs_eps),1)
                fs_f3_r = fs_f3 / max(len(val_f3_eps),1)
                fs_f4_r = fs_f4 / max(len(val_f4_eps),1)
                if fs_all > 0.10 or fs_f3_r > 0.10 or fs_f4_r > 0.10: continue
                rec = vt / max(len(val_opp_eps),1)
                if rec > best_recall: best_recall=rec; best_tau=tau; best_d=d

    return {'ep_auc':best_auc,'recall':best_recall,'tau':float(best_tau),'d':best_d,
            'n_train':len(train_eps),'n_val':len(val_eps)}

# ── Suite-level CV ──
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--variant',default='A0',choices=['A0','A1','A2','A3','A4'])
parser.add_argument('--all',action='store_true')
args = parser.parse_args()

def run_variant(variant):
    print('\n=== {} ==='.format(variant))
    results = []
    for test_suite in SUITES:
        train = [e for e in ep_list if e['suite']!=test_suite]
        val = [e for e in ep_list if e['suite']==test_suite]
        if len(val) < 3: continue
        r = run_fold(train, val, variant)
        results.append(r)
        print('  {}: ep_auc={:.4f} recall={:.4f} tau={:.2f} d={} n_val={}'.format(
            test_suite, r['ep_auc'], r['recall'], r['tau'], r['d'], r['n_val']))
    aucs = [r['ep_auc'] for r in results]; recs = [r['recall'] for r in results]
    print('{}: mean_auc={:.4f} mean_recall={:.4f} min_recall={:.4f}'.format(
        variant, np.mean(aucs), np.mean(recs), min(recs) if recs else 0))
    return {'variant':variant,'mean_auc':float(np.mean(aucs)),'mean_recall':float(np.mean(recs)),
            'min_recall':float(min(recs)),'results':results}

if args.all:
    all_res = {}
    for v in ['A0','A1','A2','A3','A4']:
        all_res[v] = run_variant(v)
    print('\n=== SUMMARY ===')
    for v in ['A0','A1','A2','A3','A4']:
        r = all_res[v]
        print('{}: auc={:.4f} recall={:.4f} min_rec={:.4f}'.format(v, r['mean_auc'], r['mean_recall'], r['min_recall']))
    with open(os.path.join(OUT_ROOT,'ablation.json'),'w') as f:
        json.dump({v:{'mean_auc':all_res[v]['mean_auc'],'mean_recall':all_res[v]['mean_recall'],'min_recall':all_res[v]['min_recall']} for v in all_res},f,indent=2)
else:
    run_variant(args.variant)
