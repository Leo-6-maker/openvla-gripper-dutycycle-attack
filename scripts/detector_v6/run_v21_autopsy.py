"""V2.1 P3 Autopsy + Time-to-K10 Localization Probe.

1. Episode-max AUROC: opp vs F1/F3/F4
2. Top-K corridor hit in positive episodes
3. Localization margin: inside vs outside corridor
4. Per-split logit scale + within-split ranking
5. F3/F4 peak timing analysis
6. Ordinal time-to-next-K10-start probe on frozen hidden states
"""
import json, os, sys, hashlib, time, numpy as np
import torch, torch.nn as nn
from collections import defaultdict

FEAT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
STUDENT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v21_student_training_v1'
P3_MANIFEST_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/V21_TEACHER_AND_ROLES_20260725/V21_P3_IDENTITY_MANIFEST_V2.json'
LABEL_ROOTS = ['/tmp/ft_FIT_TRAIN/labels','/tmp/ft_FIT_DEV/labels','/tmp/ft_CAL/labels','/tmp/ft_CHECK/labels','/tmp/ft_H/labels',
               '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_TEACHER_STATES_35_49_20260725/labels']
OUT_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/v21_autopsy'
os.makedirs(OUT_ROOT, exist_ok=True)

SPLITS = ['o0_i0','o0_i1','o0_i2','o1_i0','o1_i1','o1_i2',
          'o2_i0','o2_i1','o2_i2','o3_i0','o3_i1','o3_i2']
K10 = 10

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

# ── Load models and run P3 inference with hidden states ──
print('Loading models + running P3 inference with hidden states...')
models = {}
for sn in SPLITS:
    ckpt = torch.load(os.path.join(STUDENT_ROOT, sn, 'checkpoint.pt'), map_location=device, weights_only=False)
    model = CriticalTriggerStudentV2(input_dim_25d=43, hidden_dim=64, receptive_field=32,
        dropout=0.1, use_policy_bypass=False, use_gripper_bypass=False,
        head_names=['k10_startability'])
    model.load_state_dict(ckpt['state_dict']); model.to(device); model.eval()
    models[sn] = model

p3_ids = set(json.load(open(P3_MANIFEST_PATH))['identities'])
p3_eps = {}

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
                if eid not in p3_ids: continue
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
                    # Also extract hidden states from one model (o0_i0)
                    # V2-B: encoder expects 43D, bypass disabled. get_hidden passes through encoder+fusion.
                    hidden = models['o0_i0'].get_hidden(x_cat).squeeze(0).cpu().numpy()

                raw = raw_sum / len(SPLITS)
                cc = np.array([labels_l[min(t,len(labels_l)-1)].get('candidate_close',False) for t in range(T)], dtype=bool)
                k10_s = np.array([labels_l[min(t,len(labels_l)-1)].get('strict_k10_feasible',False) and labels_l[min(t,len(labels_l)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                k10_k = np.array([labels_l[min(t,len(labels_l)-1)].get('strict_k10_known_mask',False) for t in range(T)], dtype=bool)
                has_opp = bool(k10_s[:max_t].any())

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

                # time_to_start: distance to next K10 feasible step
                time_to_start = np.full(T, -1, dtype=int)
                next_start = -1
                for t in range(T-1, -1, -1):
                    if t < max_t and k10_s[t] and k10_k[t]:
                        next_start = t
                    if next_start >= 0:
                        time_to_start[t] = next_start - t

                p3_eps[eid] = {'T':T,'max_t':max_t,'raw':raw,'cc':cc,'k10_s':k10_s,'k10_k':k10_k,
                    'has_opp':has_opp,'absence_reason':absence_reason,'suite':suite,'eid':eid,
                    'hidden':hidden,'time_to_start':time_to_start}

print('P3 episodes: {}'.format(len(p3_eps)))

# ═══ 1. Episode-max AUROC: opp vs F1/F3/F4 ═══
print('\n=== 1. EPISODE-MAX AUROC ===')
def ep_max(d):
    max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['raw'][:max_t]
    return float(sc[cc].max()) if cc.any() else float(sc.max())

opp_scores = [ep_max(d) for d in p3_eps.values() if d['has_opp']]
f1_scores = [ep_max(d) for d in p3_eps.values() if d['absence_reason']=='F1_STRUCTURAL_ZERO']
f3_scores = [ep_max(d) for d in p3_eps.values() if d['absence_reason']=='F3_NO_MANIPULATION']
f4_scores = [ep_max(d) for d in p3_eps.values() if d['absence_reason']=='F4_NO_STABLE_GRASP']

for name, neg_scores in [('F1',f1_scores),('F3',f3_scores),('F4',f4_scores)]:
    if len(neg_scores) < 2: continue
    all_s = np.concatenate([opp_scores, neg_scores])
    all_l = np.array([1.0]*len(opp_scores) + [0.0]*len(neg_scores))
    print('  AUROC(opp vs {}): {:.4f} (opp={} neg={})'.format(name, auroc(all_l,all_s), len(opp_scores), len(neg_scores)))

# Overlap analysis
print('  Score ranges: opp=[{:.2f},{:.2f}] F3=[{:.2f},{:.2f}] F4=[{:.2f},{:.2f}]'.format(
    min(opp_scores), max(opp_scores), min(f3_scores) if f3_scores else 0, max(f3_scores) if f3_scores else 0,
    min(f4_scores) if f4_scores else 0, max(f4_scores) if f4_scores else 0))
overlap_f3 = sum(1 for s in f3_scores if s > np.percentile(opp_scores, 10))
overlap_f4 = sum(1 for s in f4_scores if s > np.percentile(opp_scores, 10))
print('  F3 scores > opp_p10: {}/{}  F4 scores > opp_p10: {}/{}'.format(overlap_f3, len(f3_scores), overlap_f4, len(f4_scores)))

# ═══ 2. Positive episode Top-K corridor hit ═══
print('\n=== 2. TOP-K CORRIDOR HIT (positive episodes) ===')
top1_hit=0; top3_hit=0; top5_hit=0; total_opp=0
margins = []; offsets = []
for eid, d in p3_eps.items():
    if not d['has_opp']: continue
    total_opp += 1
    max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['raw'][:max_t]
    ks = d['k10_s'][:max_t]; kk = d['k10_k'][:max_t]
    # Get top-K steps (by score) among cc steps
    cc_indices = np.where(cc)[0]
    if len(cc_indices) == 0: continue
    cc_scores = sc[cc_indices]
    topk_idx = cc_indices[np.argsort(cc_scores)[::-1]]

    # Check if any top-K step is in K10 corridor
    for k, (name, count) in enumerate([('top1',1),('top3',3),('top5',5)]):
        if k == 0: top1_hit += int(any(ks[idx] and kk[idx] for idx in topk_idx[:count]))
        elif k == 1: top3_hit += int(any(ks[idx] and kk[idx] for idx in topk_idx[:count]))
        else: top5_hit += int(any(ks[idx] and kk[idx] for idx in topk_idx[:count]))

    # Localization margin: max(inside corridor) - max(outside corridor)
    inside_mask = ks & kk; outside_mask = cc & (~inside_mask)
    inside_max = sc[inside_mask].max() if inside_mask.any() else -1e9
    outside_max = sc[outside_mask].max() if outside_mask.any() else -1e9
    margins.append(inside_max - outside_max)

    # Timing offset: argmax - first feasible start
    first_feas = np.where(ks & kk)[0]
    argmax_idx = cc_indices[np.argmax(cc_scores)]
    if len(first_feas) > 0:
        offsets.append(argmax_idx - first_feas[0])

print('  Top-1 hit: {:.1%} ({}/{})'.format(top1_hit/total_opp, top1_hit, total_opp))
print('  Top-3 hit: {:.1%} ({}/{})'.format(top3_hit/total_opp, top3_hit, total_opp))
print('  Top-5 hit: {:.1%} ({}/{})'.format(top5_hit/total_opp, top5_hit, total_opp))
print('  Localization margin: mean={:.2f} median={:.2f} p25={:.2f} (positive=inside>outside)'.format(
    np.mean(margins), np.median(margins), np.percentile(margins,25)))
margin_pos = sum(1 for m in margins if m > 0)
print('  Margin > 0: {}/{} ({:.1%})'.format(margin_pos, len(margins), margin_pos/max(len(margins),1)))
print('  Timing offset (argmax - first_feas): mean={:.1f} median={:.1f}'.format(np.mean(offsets), np.median(offsets)))

# ═══ 3. Per-split logit scale ═══
print('\n=== 3. PER-SPLIT LOGIT SCALE (ensemble output) ===')
for sn in SPLITS:
    sn_scores = []
    for eid, d in p3_eps.items():
        max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['raw'][:max_t]
        if cc.any(): sn_scores.extend(sc[cc].tolist())
    if sn_scores:
        arr = np.array(sn_scores)
        print('  {}: p50={:.2f} IQR={:.2f} p10={:.2f} p90={:.2f}'.format(sn, np.median(arr), np.percentile(arr,75)-np.percentile(arr,25), np.percentile(arr,10), np.percentile(arr,90)))

# ═══ 4. F3/F4 peak timing ═══
print('\n=== 4. F3/F4 PEAK TIMING ===')
for stratum_name in ['F3_NO_MANIPULATION','F4_NO_STABLE_GRASP']:
    eps_list = [(eid,d) for eid,d in p3_eps.items() if d['absence_reason']==stratum_name]
    peak_positions = []; peak_durations = []; peak_values = []
    for eid, d in eps_list:
        max_t = d['max_t']; cc = d['cc'][:max_t]; sc = d['raw'][:max_t]
        if not cc.any(): continue
        cc_indices = np.where(cc)[0]; cc_scores = sc[cc_indices]
        peak_idx = cc_indices[np.argmax(cc_scores)]; peak_val = cc_scores.max()
        peak_positions.append(peak_idx / max(d['T'],1))  # normalized position
        peak_values.append(peak_val)
        # Duration: consecutive cc steps above half-peak
        half_peak = peak_val * 0.5
        dur = 0
        for t in range(peak_idx, min(d['T'], peak_idx+20)):
            if t < d['max_t'] and cc[t] and sc[t] > half_peak: dur += 1
            else: break
        peak_durations.append(dur)
    if peak_positions:
        print('  {}: peak_pos_norm mean={:.2f} median={:.2f} (0=start,1=end)'.format(stratum_name, np.mean(peak_positions), np.median(peak_positions)))
        print('    peak_value mean={:.2f} median={:.2f}'.format(np.mean(peak_values), np.median(peak_values)))
        print('    peak_duration mean={:.1f} median={:.1f} steps'.format(np.mean(peak_durations), np.median(peak_durations)))
        long_peaks = sum(1 for d in peak_durations if d > 5)
        print('    long peaks (>5 steps): {}/{}'.format(long_peaks, len(peak_durations)))

# ═══ 5. Time-to-Start Ordinal Probe ═══
print('\n=== 5. TIME-TO-K10-START PROBE ===')
# Build dataset: for each step with candidate_close=true, predict distance to next K10 start
# Bins: NOW(0), 1-2, 3-5, 6-10, NO_START(11+)
BINS = [(0,0), (1,2), (3,5), (6,10), (11,999)]
def bin_idx(dist):
    if dist < 0: return 4  # NO_START
    for i, (lo, hi) in enumerate(BINS):
        if lo <= dist <= hi: return i
    return 4

X_hidden = []; y_bin = []; y_opp = []; ep_ids = []; suites_list = []
for eid, d in p3_eps.items():
    max_t = d['max_t']; cc = d['cc'][:max_t]; hidden = d['hidden'][:max_t]
    tts = d['time_to_start'][:max_t]
    for t in range(max_t):
        if cc[t]:
            X_hidden.append(hidden[t])
            y_bin.append(bin_idx(tts[t]))
            y_opp.append(1.0 if d['has_opp'] else 0.0)
            ep_ids.append(eid)
            suites_list.append(d['suite'])

X_h = np.stack(X_hidden); y_b = np.array(y_bin); y_o = np.array(y_opp)
print('Probe data: {} cc steps from {} episodes'.format(len(X_h), len(set(ep_ids))))
bin_counts = np.bincount(y_b, minlength=5)
print('Bin distribution: NOW={} 1-2={} 3-5={} 6-10={} NO_START={}'.format(*bin_counts))

# Leave-one-suite-out probe
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SUITES = sorted(set(suites_list))
probe_results = []
for test_suite in SUITES:
    train_idx = [i for i,s in enumerate(suites_list) if s != test_suite]
    test_idx = [i for i,s in enumerate(suites_list) if s == test_suite]
    if len(test_idx) < 10: continue
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_h[train_idx]); X_te = sc.transform(X_h[test_idx])
    clf = LogisticRegression(max_iter=5000, C=0.1, multi_class='multinomial')
    clf.fit(X_tr, y_b[train_idx])
    acc = clf.score(X_te, y_b[test_idx])
    probe_results.append({'suite':test_suite,'n_test':len(test_idx),'acc':acc,
        'now_recall':float((clf.predict(X_te)==0).mean() if (y_b[test_idx]==0).sum()>0 else 0)})
    print('  {}: acc={:.3f} n={}'.format(test_suite, acc, len(test_idx)))

if probe_results:
    accs = [r['acc'] for r in probe_results]
    print('Probe mean acc: {:.3f} (baseline={:.3f})'.format(np.mean(accs), max(bin_counts)/bin_counts.sum()))

# Bin-level: can NOW be distinguished from NOT_NOW?
y_now = (y_b == 0).astype(int)  # NOW vs all others
probe_now_results = []
for test_suite in SUITES:
    train_idx = [i for i,s in enumerate(suites_list) if s != test_suite]
    test_idx = [i for i,s in enumerate(suites_list) if s == test_suite]
    if len(test_idx) < 10 or y_now[test_idx].sum()==0 or y_now[test_idx].sum()==len(test_idx): continue
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_h[train_idx]); X_te = sc.transform(X_h[test_idx])
    clf = LogisticRegression(max_iter=5000, C=0.1, class_weight='balanced')
    clf.fit(X_tr, y_now[train_idx])
    acc = clf.score(X_te, y_now[test_idx])
    prob = clf.predict_proba(X_te)[:,1]
    auc = auroc(y_now[test_idx], prob) if y_now[test_idx].sum()>0 and (1-y_now[test_idx]).sum()>0 else 0.5
    probe_now_results.append({'suite':test_suite,'acc':acc,'auc':auc})
    print('  NOW vs NOT_NOW {}: acc={:.3f} auc={:.3f}'.format(test_suite, acc, auc))

if probe_now_results:
    now_aucs = [r['auc'] for r in probe_now_results]
    print('NOW probe mean AUC: {:.3f}'.format(np.mean(now_aucs)))

# ═══ Seal ═══
autopsy = {
    'schema':'V21_P3_AUTOPSY_V1',
    'episode_max_auroc': {'opp_vs_f3': float(auroc(np.concatenate([np.ones(len(opp_scores)),np.zeros(len(f3_scores))]),np.concatenate([opp_scores,f3_scores]))) if f3_scores else None,
        'opp_vs_f4': float(auroc(np.concatenate([np.ones(len(opp_scores)),np.zeros(len(f4_scores))]),np.concatenate([opp_scores,f4_scores]))) if f4_scores else None},
    'topk_hit': {'top1':float(top1_hit/total_opp),'top3':float(top3_hit/total_opp),'top5':float(top5_hit/total_opp)},
    'localization_margin_mean': float(np.mean(margins)),
    'timing_offset_median': float(np.median(offsets)) if offsets else None,
    'probe': {'time_to_start_acc_mean': float(np.mean(accs)) if probe_results else None,
        'now_vs_notnow_auc_mean': float(np.mean(now_aucs)) if probe_now_results else None},
}
with open(os.path.join(OUT_ROOT,'V21_P3_AUTOPSY_V1.json'),'w') as f: json.dump(autopsy,f,indent=2)

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
print('\nSealed: {}'.format(sh[:16]))
