#!/usr/bin/env python3
"""Full logit-margin analysis: distribution, ranking, threshold sweep, classifier."""
import csv, glob, os, re, sys
from collections import defaultdict
import numpy as np

runs = '/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs'
shared = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
repo = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'

# Load labels
with open(shared + '/object_phase_response_labels_v2.csv') as f:
    labels = list(csv.DictReader(f))
label_keys = {}
for r in labels:
    key = (r['task_key'].strip(), r['state_id'].strip(),
           r['window_start'].strip(), r['window_end'].strip())
    label_keys[key] = r

# Load mechanism taxonomy
mech_path = repo + '/tables/vulnerability_mechanism_taxonomy_audit.csv'
mech_keys = {}
if os.path.exists(mech_path):
    with open(mech_path) as f:
        for r in csv.DictReader(f):
            key = (r['task_key'].strip(), r['state_id'].strip(),
                   r['window_start'].strip(), r['window_end'].strip())
            mech_keys[key] = r.get('mechanism_type', '')

# Scan traces for logit data
traces = sorted(glob.glob(runs + '/vis_*_clean_full_d18_*_trace.csv'),
                key=os.path.getmtime, reverse=True)
candidates = []
seen_keys = set()

for t in traces:
    fname = os.path.basename(t)
    m = re.search(r'vis_(\w+)_state(\d+)_clean.*_w(\d+)_(\d+)_seed(\d+)_(\d+)_trace', fname)
    if not m: continue
    task, sid, ws, we = m.group(1), m.group(2), m.group(3), m.group(4)
    key = (task, sid, ws, we)
    if key in seen_keys: continue
    seen_keys.add(key)

    with open(t) as f:
        reader = csv.DictReader(f)
        if 'gripper_logit_margin' not in (reader.fieldnames or []): continue
        rows = list(reader)

    window_rows = [r for r in rows if r.get('in_window') == 'True']
    if len(window_rows) < 2: continue

    margins, entropies, open_masses, top2s, all_ents = [], [], [], [], []
    for r in window_rows:
        try:
            mv = float(r.get('gripper_logit_margin', 0))
            if abs(mv) > 0.001: margins.append(mv)
        except: pass
        try: entropies.append(float(r.get('gripper_entropy', 0)))
        except: pass
        try: open_masses.append(float(r.get('gripper_logit_open_mass', 0)))
        except: pass
        try: top2s.append(float(r.get('gripper_top2_margin', 0)))
        except: pass
        try: all_ents.append(float(r.get('all_action_entropy', 0)))
        except: pass
    if len(margins) < 2: continue

    ma = np.array(margins)
    n_low = int(np.sum(np.abs(ma) < 0.1))
    streak = max_streak = 0
    for mv in ma:
        if abs(mv) < 0.1: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    label = label_keys.get(key, {})
    candidates.append({
        'task_key': task, 'state_id': sid, 'window_start': ws, 'window_end': we,
        'n_steps': len(window_rows),
        'logit_margin_min': float(np.min(ma)), 'logit_margin_max': float(np.max(ma)),
        'logit_margin_mean': float(np.mean(ma)), 'logit_margin_std': float(np.std(ma)),
        'logit_margin_range': float(np.max(ma) - np.min(ma)),
        'entropy_mean': float(np.mean(entropies)) if entropies else 0,
        'entropy_max': float(np.max(entropies)) if entropies else 0,
        'top2_margin_min': float(np.min(top2s)) if top2s else 0,
        'top2_margin_mean': float(np.mean(top2s)) if top2s else 0,
        'open_mass_max': float(np.max(open_masses)) if open_masses else 0,
        'open_mass_mean': float(np.mean(open_masses)) if open_masses else 0,
        'all_action_entropy_mean': float(np.mean(all_ents)) if all_ents else 0,
        'low_margin_step_count': int(n_low),
        'longest_low_margin_streak': int(max_streak),
        'label_status': label.get('label_status', '?'),
        'train_use': label.get('label_use', '?'),
        'taxonomy': label.get('taxonomy', '?'),
        'mechanism_type': mech_keys.get(key, '?'),
    })

train = [c for c in candidates if c['label_status'] in ('positive','negative')]
pos = [c for c in train if c['label_status'] == 'positive']
neg = [c for c in train if c['label_status'] == 'negative']
print('Train candidates: %d (pos=%d, neg=%d)' % (len(train), len(pos), len(neg)))

if len(neg) < 3:
    print('NOT ENOUGH NEGATIVES (%d) — exiting' % len(neg))
    sys.exit(0)

# ═══════════════════════════════════════════════════════════════
# 1. FEATURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
dist_cols = ['task_key','state_id','window_start','window_end','label_status',
    'mechanism_type','taxonomy',
    'logit_margin_min','logit_margin_max','logit_margin_mean','logit_margin_std','logit_margin_range',
    'entropy_mean','entropy_max','top2_margin_min','top2_margin_mean',
    'open_mass_max','open_mass_mean','all_action_entropy_mean',
    'low_margin_step_count','longest_low_margin_streak','n_steps']

with open(repo + '/tables/online_safe_v3_logit_feature_distribution.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=dist_cols, extrasaction='ignore')
    w.writeheader(); w.writerows(candidates)

# Per-mechanism stats
FEATS = ['logit_margin_mean','logit_margin_min','logit_margin_max','logit_margin_std',
         'logit_margin_range','entropy_mean','entropy_max','top2_margin_mean','top2_margin_min',
         'open_mass_max','open_mass_mean','all_action_entropy_mean',
         'low_margin_step_count','longest_low_margin_streak']

mechanisms = sorted(set(c['mechanism_type'] for c in train if c['mechanism_type'] != '?'))
print('\n=== FEATURE DISTRIBUTION BY MECHANISM ===')
for mech in mechanisms:
    subset = [c for c in train if c['mechanism_type'] == mech]
    print('\n%s (n=%d):' % (mech, len(subset)))
    for feat in FEATS:
        vals = [c[feat] for c in subset]
        print('  %s: mean=%.4f std=%.4f min=%.4f max=%.4f' % (
            feat, np.mean(vals), np.std(vals), np.min(vals), np.max(vals)))

# ═══════════════════════════════════════════════════════════════
# 2. RANKING METRICS
# ═══════════════════════════════════════════════════════════════
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

print('\n=== RANKING METRICS ===')

# Use logit_margin_mean as primary score (higher = less negative = closer to OPEN = more vulnerable)
X = np.array([[float(c.get(f, 0) or 0) for f in FEATS] for c in train])
y = np.array([1 if c['label_status']=='positive' else 0 for c in train])

# Simple score: -logit_margin_mean (higher score = less certain CLOSE = more vulnerable)
simple_score = -np.array([c['logit_margin_mean'] for c in train])  # negate so higher = more vulnerable
# Also try margin_range as score
range_score = np.array([c['logit_margin_range'] for c in train])
# And entropy
entropy_score = np.array([c['entropy_mean'] for c in train])

ranking_rows = []
for score_name, scores in [('neg_mean_margin', simple_score), ('margin_range', range_score), ('entropy_mean', entropy_score)]:
    try:
        auroc = roc_auc_score(y, scores)
    except: auroc = float('nan')
    try:
        auprc = average_precision_score(y, scores)
    except: auprc = float('nan')
    try:
        u_stat, p_val = mannwhitneyu([s for s,l in zip(scores,y) if l==1],
                                       [s for s,l in zip(scores,y) if l==0],
                                       alternative='two-sided')
    except: u_stat, p_val = float('nan'), float('nan')
    pos_ranks = np.argsort(np.argsort(scores))[y == 1] + 1
    neg_ranks = np.argsort(np.argsort(scores))[y == 0] + 1
    pos_mean_rank = np.mean(pos_ranks) if len(pos_ranks) > 0 else 0
    neg_mean_rank = np.mean(neg_ranks) if len(neg_ranks) > 0 else 0

    print('%s: AUROC=%.4f AUPRC=%.4f MW_p=%.4f pos_rank=%.1f neg_rank=%.1f' % (
        score_name, auroc, auprc, p_val, pos_mean_rank, neg_mean_rank))
    ranking_rows.append({
        'score_name': score_name, 'AUROC': round(auroc, 4), 'AUPRC': round(auprc, 4),
        'MW_U': round(u_stat, 1) if not np.isnan(u_stat) else '',
        'MW_p': round(p_val, 4) if not np.isnan(p_val) else '',
        'pos_mean_rank': round(pos_mean_rank, 1), 'neg_mean_rank': round(neg_mean_rank, 1),
    })

with open(repo + '/tables/online_safe_v3_logit_ranking_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=ranking_rows[0].keys())
    w.writeheader(); w.writerows(ranking_rows)

# ═══════════════════════════════════════════════════════════════
# 3. THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score, confusion_matrix

print('\n=== THRESHOLD SWEEP ===')
score = simple_score  # neg_mean_margin as score
thresholds = np.linspace(np.min(score), np.max(score), 20)
sweep_rows = []
for thresh in thresholds:
    preds = (score >= thresh).astype(int)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    pos_rec = recall_score(y, preds, pos_label=1) if 1 in y else 0
    neg_rec = recall_score(y, preds, pos_label=0) if 0 in y else 0
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    bal = balanced_accuracy_score(y, preds)
    mcc = matthews_corrcoef(y, preds)
    sweep_rows.append({
        'threshold': round(thresh, 2), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'PosRec': round(pos_rec, 4), 'NegRec': round(neg_rec, 4),
        'FPR': round(fpr, 4), 'BalAcc': round(bal, 4), 'MCC': round(mcc, 4),
    })

with open(repo + '/tables/online_safe_v3_logit_threshold_sweep.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
    w.writeheader(); w.writerows(sweep_rows)

# Best threshold by BalAcc
best = max(sweep_rows, key=lambda r: r['BalAcc'])
print('Best threshold: %.2f -> TP=%d FP=%d TN=%d FN=%d BalAcc=%.4f PosRec=%.4f NegRec=%.4f FPR=%.4f' % (
    best['threshold'], best['tp'], best['fp'], best['tn'], best['fn'],
    best['BalAcc'], best['PosRec'], best['NegRec'], best['FPR']))

# ═══════════════════════════════════════════════════════════════
# 4. CLASSIFIER (LR+RF, LOTO)
# ═══════════════════════════════════════════════════════════════
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print('\n=== CLASSIFIER ===')
X = np.nan_to_num(np.array([[float(c.get(f, 0) or 0) for f in FEATS] for c in train]))
y = np.array([1 if c['label_status']=='positive' else 0 for c in train])
X = StandardScaler().fit_transform(X)

tasks = sorted(set(c['task_key'] for c in train))
for model_name, model_fn in [('LR', lambda: LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)),
                              ('RF', lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))]:
    preds = np.zeros(len(y))
    for ht in tasks:
        tr_idx = [i for i,c in enumerate(train) if c['task_key']!=ht]
        te_idx = [i for i,c in enumerate(train) if c['task_key']==ht]
        if len(tr_idx)<2 or len(te_idx)==0: continue
        if len(set(y[tr_idx])) < 2: continue
        model = model_fn()
        model.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] = model.predict(X[te_idx])

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    bal = balanced_accuracy_score(y, preds)
    pos_rec = recall_score(y, preds, pos_label=1) if 1 in y else 0
    neg_rec = recall_score(y, preds, pos_label=0) if 0 in y else 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    mcc = matthews_corrcoef(y, preds)
    print('%s: BalAcc=%.4f PosRec=%.4f NegRec=%.4f FPR=%.4f MCC=%.4f TP=%d FP=%d TN=%d FN=%d' % (
        model_name, bal, pos_rec, neg_rec, fpr, mcc, tp, fp, tn, fn))

    # Which claim_usable missed?
    missed_v1 = [c for c in pos if c['taxonomy'] == 'claim_usable']
    recovered = 0
    for c in missed_v1:
        idx = train.index(c)
        if preds[idx] == 1:
            recovered += 1
    print('  Claim_usable recovered: %d/%d' % (recovered, len(missed_v1)))

# ═══════════════════════════════════════════════════════════════
# 5. DECISION
# ═══════════════════════════════════════════════════════════════
print('\n=== DECISION ===')
auroc = roc_auc_score(y, simple_score) if len(set(y)) > 1 else 0
best_fpr = best['FPR']
best_bal = best['BalAcc']

print('AUROC=%.4f | Best FPR=%.4f | Best BalAcc=%.4f' % (auroc, best_fpr, best_bal))
if auroc <= 0.55 or best_fpr >= 0.7:
    print('VERDICT: STOP passive logit detector. Turn to active probe.')
    print('Reason: AUROC=%.4f <= 0.55 OR FPR=%.4f >= 0.7' % (auroc, best_fpr))
elif best_bal >= 0.65 and best_fpr <= 0.4:
    print('VERDICT: CONTINUE full 31/31 logit validation.')
else:
    print('VERDICT: BORDERLINE — need more data but signal is weak.')
    print('AUROC=%.4f, FPR=%.4f, BalAcc=%.4f' % (auroc, best_fpr, best_bal))
