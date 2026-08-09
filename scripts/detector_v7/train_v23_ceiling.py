"""V2.3 No-Vision Observability Ceiling.

N0: W32 baseline (A4 recipe)
N1: W64 multiscale causal encoder
N2: W128 multiscale causal encoder
N3: W128 + explicit command-qpos response proxies
N4: W128 + proxies + GroupDRO

All: leave-one-suite-out CV, FS_F3/F4/all <= 10%, report worst-suite recall.
"""
import json, os, sys, hashlib, random, numpy as np, torch, torch.nn as nn
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
DEV2_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V22_ROLE_ALLOCATION_20260725/V22_DEV2_IDENTITY_MANIFEST_V1.json'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
    '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/v23_ceiling'
os.makedirs(OUT_ROOT, exist_ok=True)

K10=10; SEED=42; EPOCHS=20; LR=1e-3; WD=1e-4; BATCH=4
HIDDEN=64; DROPOUT=0.1
RANK_MARGIN=2.0; RANK_LAMBDA=0.1
SUITES = ['libero_10','libero_goal','libero_object','libero_spatial']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev2_ids = set(json.load(open(DEV2_MANIFEST))['identities'])

def auroc(yt,ys):
    if len(yt)<2: return 0.5
    n_pos=yt.sum();n_neg=len(yt)-n_pos
    if n_pos==0 or n_neg==0: return 0.5
    desc=np.argsort(ys)[::-1];ysort=yt[desc]
    tpr=np.cumsum(ysort)/n_pos;fpr=np.cumsum(1-ysort)/n_neg
    return float(np.trapz(tpr,fpr))

# ── Response proxies from 43D ──
def compute_proxies(f25d, p9d, g9d, T):
    """Derive physical response features from existing 43D data.
    f25d: [T,25], p9d: [T,9], g9d: [T,9]
    Returns: [T, P] additional features
    """
    proxies = np.zeros((T, 8), dtype=np.float32)
    # gripper_command is f25d[:,0], gripper_qpos is f25d[:,1]
    cmd = f25d[:,0]; qpos = f25d[:,1]
    # 1. command-qpos residual (normalized)
    proxies[:,0] = cmd - qpos
    # 2. close command indicator (cmd < 0 means close)
    proxies[:,1] = (cmd < 0).astype(np.float32)
    # 3. qpos change rate (step difference)
    proxies[1:,2] = np.diff(qpos); proxies[0,2] = 0
    # 4. close duration: cumulative close command count
    close_dur = np.zeros(T); dur = 0
    for t in range(T):
        if cmd[t] < 0: dur += 1
        else: dur = 0
        close_dur[t] = dur
    proxies[:,3] = close_dur
    # 5. EEF speed magnitude
    eef_v = np.sqrt(f25d[:,6]**2 + f25d[:,7]**2 + f25d[:,8]**2)
    proxies[:,4] = eef_v
    # 6. qpos stability after close (variance in 5-step window)
    for t in range(T):
        w_start = max(0, t-4); w_end = min(T, t+1)
        proxies[t,5] = np.var(qpos[w_start:w_end]) if w_end-w_start > 1 else 0
    # 7. policy close probability (from gripper 9D)
    proxies[:,6] = g9d[:,0]  # close_probability_mass
    # 8. action entropy (uncertainty)
    proxies[:,7] = g9d[:,7]  # action_token_entropy_normalized
    return np.nan_to_num(proxies, 0).astype(np.float32)

# ── Multiscale Causal Encoder ──
class MultiScaleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, short_rf=32, long_rf=128, dropout=0.1):
        super().__init__()
        from gripper_attack.v6_critical_student import CausalTCNEncoder
        self.short_tcn = CausalTCNEncoder(input_dim, hidden_dim, short_rf, dropout)
        self.long_tcn = CausalTCNEncoder(input_dim, hidden_dim, long_rf, dropout)
        self.fusion = nn.Linear(hidden_dim*2, hidden_dim)
    def forward(self, x):
        hs = self.short_tcn(x)
        hl = self.long_tcn(x)
        return self.fusion(torch.cat([hs, hl], dim=-1))

class SimpleEncoder(nn.Module):
    """Single-scale encoder reusing proven V2-B CausalTCN logic."""
    def __init__(self, input_dim, hidden_dim, rf, dropout=0.1):
        super().__init__()
        from gripper_attack.v6_critical_student import CausalTCNEncoder
        self.tcn = CausalTCNEncoder(input_dim, hidden_dim, rf, dropout)
    def forward(self, x):
        return self.tcn(x)

# ── Load data ──
print('Loading DEV2...')
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
                absence_reason = 'OPPORTUNITY_PRESENT'
                if not has_opp:
                    any_k10_known = any(k10_k[:max_t])
                    any_mk = any(labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False) for t in range(max_t))
                    any_gk = any(labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False) for t in range(max_t))
                    n_mp = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('manipulation_active',False) and labels[min(t,len(labels)-1)].get('manipulation_active_known_mask',False))
                    n_gp = sum(1 for t in range(max_t) if labels[min(t,len(labels)-1)].get('grasp_established',False) and labels[min(t,len(labels)-1)].get('grasp_established_known_mask',False))
                    if not any_k10_known: absence_reason='F1_STRUCTURAL_ZERO'
                    elif n_mp==0 and any_mk: absence_reason='F3_NO_MANIPULATION'
                    elif n_gp==0 and any_gk: absence_reason='F4_NO_STABLE_GRASP'
                    else: absence_reason='OTHER_ABSENT'
                proxies = compute_proxies(f25d, p9d, g9d, T)
                all_eps[eid] = {'eid':eid,'T':T,'max_t':max_t,'f25d':f25d,'p9d':p9d,'g9d':g9d,
                    'proxies':proxies,'k10_s':k10_s,'k10_k':k10_k,'cc':cc,
                    'has_opp':has_opp,'absence_reason':absence_reason,'suite':suite}
ep_list = list(all_eps.values())
opp_eps = [e for e in ep_list if e['has_opp']]
f3_eps = [e for e in ep_list if e['absence_reason']=='F3_NO_MANIPULATION']
f4_eps = [e for e in ep_list if e['absence_reason']=='F4_NO_STABLE_GRASP']
other_abs = [e for e in ep_list if not e['has_opp'] and e['absence_reason'] not in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP')]
print('DEV2: {} eps (opp={} F3={} F4={} other={})'.format(len(ep_list),len(opp_eps),len(f3_eps),len(f4_eps),len(other_abs)))

# ── Train one fold ──
def run_fold(train_eps, val_eps, variant):
    input_dim = 43 if variant == 'N0' else (43+8 if variant in ('N3','N4') else 43)
    use_multiscale = variant in ('N1','N2')
    use_proxies = variant in ('N3','N4')
    use_gdro = variant == 'N4'
    rf = {'N0':32,'N1':64,'N2':128,'N3':128,'N4':128}[variant]

    if use_multiscale:
        encoder = MultiScaleEncoder(input_dim, HIDDEN, 32, rf, DROPOUT).to(device)
    else:
        encoder = SimpleEncoder(input_dim, HIDDEN, rf, DROPOUT).to(device)
    head = nn.Linear(HIDDEN, 1).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=LR, weight_decay=WD)

    # Normalization
    cat_25d = np.concatenate([e['f25d'] for e in train_eps], axis=0)
    cat_p9d = np.concatenate([e['p9d'] for e in train_eps], axis=0)
    cat_g9d = np.concatenate([e['g9d'] for e in train_eps], axis=0)
    n25d_m = torch.tensor(cat_25d.mean(0), device=device); n25d_s = torch.tensor(cat_25d.std(0).clip(1e-8), device=device)
    np9d_m = torch.tensor(cat_p9d.mean(0), device=device); np9d_s = torch.tensor(cat_p9d.std(0).clip(1e-8), device=device)
    ng9d_m = torch.tensor(cat_g9d.mean(0), device=device); ng9d_s = torch.tensor(cat_g9d.std(0).clip(1e-8), device=device)

    t_opp = [e for e in train_eps if e['has_opp']]; t_f3 = [e for e in train_eps if e['absence_reason']=='F3_NO_MANIPULATION']
    t_f4 = [e for e in train_eps if e['absence_reason']=='F4_NO_STABLE_GRASP']
    t_other = [e for e in train_eps if not e['has_opp'] and e['absence_reason'] not in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP')]

    # GroupDRO weights
    group_weights = {'opp':1.0,'F3':1.0,'F4':1.0,'other':1.0}

    best_auc = -1.0; best_state = None
    for epoch in range(1, EPOCHS+1):
        encoder.train(); head.train(); total_loss = 0.0; n_batches = 0
        random.shuffle(t_opp); random.shuffle(t_f3); random.shuffle(t_f4); random.shuffle(t_other)
        n_iter = max(len(t_opp), len(t_f3), len(t_f4)) // max(BATCH//4, 1) + 1
        group_losses = {'opp':0.0,'F3':0.0,'F4':0.0,'other':0.0}; group_counts = {'opp':0,'F3':0,'F4':0,'other':0}

        for bi in range(n_iter):
            batch = [t_opp[bi % len(t_opp)]]
            if t_f3: batch.append(t_f3[bi % len(t_f3)])
            if t_f4: batch.append(t_f4[bi % len(t_f4)])
            if t_other: batch.append(t_other[bi % len(t_other)])

            max_T = max(e['T'] for e in batch); B = len(batch)
            inp_dim = input_dim
            x_cat = torch.zeros(B, max_T, inp_dim, device=device)
            k10_s = torch.zeros(B, max_T, 1, device=device); k10_k = torch.zeros(B, max_T, 1, device=device)
            cc = torch.zeros(B, max_T, 1, device=device)

            for b, e in enumerate(batch):
                T = e['T']; base = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1)
                if use_proxies:
                    base = torch.cat([base, torch.tensor(e['proxies'],device=device)], dim=-1)
                x_cat[b,:T] = base
                k10_s[b,:T,0] = torch.tensor(e['k10_s'],device=device).float()
                k10_k[b,:T,0] = torch.tensor(e['k10_k'],device=device).float()
                cc[b,:T,0] = torch.tensor(e['cc'],device=device).float()

            h = encoder(x_cat); logit = head(h)
            bce = nn.functional.binary_cross_entropy_with_logits(logit, k10_s, reduction='none')
            loss = (bce * k10_k).sum() / k10_k.sum().clamp(min=1)

            # GroupDRO: track per-group loss
            if use_gdro:
                per_group_loss = {}
                for b, e in enumerate(batch):
                    g = 'opp' if e['has_opp'] else (e['absence_reason'] if e['absence_reason'] in ('F3_NO_MANIPULATION','F4_NO_STABLE_GRASP') else 'other')
                    if g == 'F3_NO_MANIPULATION': g = 'F3'
                    elif g == 'F4_NO_STABLE_GRASP': g = 'F4'
                    gb = (bce[b] * k10_k[b]).sum() / k10_k[b].sum().clamp(min=1)
                    per_group_loss.setdefault(g, []).append(gb)
                for g, losses_g in per_group_loss.items():
                    gl = torch.stack(losses_g).mean()
                    group_losses[g] += gl.item(); group_counts[g] += 1
                # Apply group weights
                weighted = sum(group_weights.get(g,1.0)*torch.stack(v).mean() for g,v in per_group_loss.items())
                loss = weighted / sum(group_weights.get(g,1.0) for g in per_group_loss)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1

        # GroupDRO update: increase weight for worst-performing group
        if use_gdro and epoch % 3 == 0:
            avg_group_loss = {g: group_losses[g]/max(group_counts[g],1) for g in group_losses if group_counts[g]>0}
            if avg_group_loss:
                worst_g = max(avg_group_loss, key=avg_group_loss.get)
                group_weights[worst_g] *= 1.5

        # Eval
        encoder.eval(); head.eval()
        val_scores = {}
        with torch.no_grad():
            for e in val_eps:
                base = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                    (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                    (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1)
                if use_proxies: base = torch.cat([base, torch.tensor(e['proxies'],device=device)], dim=-1)
                h = encoder(base.unsqueeze(0)); sc = head(h).squeeze().cpu().numpy()
                val_scores[e['eid']] = sc

        val_opp_sc = []; val_abs_sc = []
        for e in val_eps:
            sc = val_scores[e['eid']]
            ep_sc = float(sc[:e['max_t']][e['cc'][:e['max_t']]].max()) if e['cc'][:e['max_t']].any() else float(sc[:e['max_t']].max())
            if e['has_opp']: val_opp_sc.append(ep_sc)
            else: val_abs_sc.append(ep_sc)
        ep_auc = auroc(np.array([1.]*len(val_opp_sc)+[0.]*len(val_abs_sc)), np.array(val_opp_sc+val_abs_sc)) if val_opp_sc and val_abs_sc else 0.5

        if ep_auc > best_auc:
            best_auc = ep_auc
            best_state = {'enc':{k:v.cpu().clone() for k,v in encoder.state_dict().items()},
                          'head':{k:v.cpu().clone() for k,v in head.state_dict().items()}}

    encoder.load_state_dict(best_state['enc']); head.load_state_dict(best_state['head'])
    encoder.eval(); head.eval()

    # Threshold search with first-crossing one-shot replay
    val_opp_eps = [e for e in val_eps if e['has_opp']]; val_abs_eps = [e for e in val_eps if not e['has_opp']]
    val_f3_eps = [e for e in val_abs_eps if e['absence_reason']=='F3_NO_MANIPULATION']
    val_f4_eps = [e for e in val_abs_eps if e['absence_reason']=='F4_NO_STABLE_GRASP']

    with torch.no_grad():
        for e in val_eps:
            base = torch.cat([(torch.tensor(e['f25d'],device=device)-n25d_m)/n25d_s,
                (torch.tensor(e['p9d'],device=device)-np9d_m)/np9d_s,
                (torch.tensor(e['g9d'],device=device)-ng9d_m)/ng9d_s], dim=-1)
            if use_proxies: base = torch.cat([base, torch.tensor(e['proxies'],device=device)], dim=-1)
            val_scores[e['eid']] = head(encoder(base.unsqueeze(0))).squeeze().cpu().numpy()

    best_recall=-1; best_tau=0; best_d=1
    for tau in np.linspace(-5, 10, 31):
        for d in [1,2]:
            vt=0; fs=0; fs_f3=0; fs_f4=0
            for e in val_opp_eps:
                sc=val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t]>=tau:
                        cons+=1
                        if cons>=d:
                            if e['k10_s'][t] and e['k10_k'][t]: vt+=1
                            break
                    else: cons=0
            for e in val_abs_eps:
                sc=val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t]>=tau: cons+=1
                    if cons>=d: fs+=1; break
                    else: cons=0
            for e in val_f3_eps:
                sc=val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t]>=tau: cons+=1
                    if cons>=d: fs_f3+=1; break
                    else: cons=0
            for e in val_f4_eps:
                sc=val_scores[e['eid']]; cons=0
                for t in range(e['max_t']):
                    if e['cc'][t] and sc[t]>=tau: cons+=1
                    if cons>=d: fs_f4+=1; break
                    else: cons=0
            fs_all=fs/max(len(val_abs_eps),1); fs_f3_r=fs_f3/max(len(val_f3_eps),1); fs_f4_r=fs_f4/max(len(val_f4_eps),1)
            if fs_all>0.10 or fs_f3_r>0.10 or fs_f4_r>0.10: continue
            rec=vt/max(len(val_opp_eps),1)
            if rec>best_recall: best_recall=rec; best_tau=tau; best_d=d

    return {'ep_auc':best_auc,'recall':best_recall,'tau':float(best_tau),'d':best_d,
            'n_train':len(train_eps),'n_val':len(val_eps)}

# ── Run all variants ──
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--variant',default='N0',choices=['N0','N1','N2','N3','N4'])
parser.add_argument('--all',action='store_true')
args = parser.parse_args()

def run_variant(variant):
    print('\n=== {} ==='.format(variant))
    results = []
    for test_suite in SUITES:
        train = [e for e in ep_list if e['suite']!=test_suite]
        val = [e for e in ep_list if e['suite']==test_suite]
        if len(val)<3: continue
        r = run_fold(train, val, variant)
        results.append(r)
        print('  {}: ep_auc={:.4f} recall={:.4f} tau={:.1f} d={} n_val={}'.format(
            test_suite, r['ep_auc'], r['recall'], r['tau'], r['d'], r['n_val']))
    recs = [r['recall'] for r in results]
    print('{}: mean_recall={:.4f} min_recall={:.4f}'.format(variant, np.mean(recs), min(recs)))
    return {'variant':variant,'mean_recall':float(np.mean(recs)),'min_recall':float(min(recs)),
            'every_suite_ge_0.50':all(r>=0.50 for r in recs),'results':results}

if args.all:
    all_res = {}
    for v in ['N0','N1','N2','N3','N4']: all_res[v] = run_variant(v)
    print('\n=== CEILING SUMMARY ===')
    for v in ['N0','N1','N2','N3','N4']:
        r = all_res[v]
        print('{}: mean_rec={:.4f} min_rec={:.4f} all>=0.50={}'.format(v, r['mean_recall'], r['min_recall'], r['every_suite_ge_0.50']))
    ceiling_pass = any(r['every_suite_ge_0.50'] for r in all_res.values())
    print('NONVISUAL_CEILING: {}'.format('PASS' if ceiling_pass else 'CONFIRMED_INSUFFICIENT'))
    with open(os.path.join(OUT_ROOT,'ceiling_results.json'),'w') as f:
        json.dump({v:{'mean_recall':all_res[v]['mean_recall'],'min_recall':all_res[v]['min_recall'],'every_suite_ge_0.50':all_res[v]['every_suite_ge_0.50']} for v in all_res},f,indent=2)
else:
    run_variant(args.variant)
