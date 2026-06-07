#!/usr/bin/env python3
"""Early readout: analyze logit-margin features, train quick detector."""
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

# Scan all traces for logit data (most recent first)
traces = sorted(glob.glob(runs + '/vis_*_clean_full_d18_*_trace.csv'),
                key=os.path.getmtime, reverse=True)
candidates = []
seen_keys = set()

for t in traces:
    fname = os.path.basename(t)
    m = re.search(r'vis_(\w+)_state(\d+)_clean.*_w(\d+)_(\d+)_seed(\d+)_(\d+)_trace', fname)
    if not m:
        continue
    task, sid, ws, we = m.group(1), m.group(2), m.group(3), m.group(4)
    key = (task, sid, ws, we)
    if key in seen_keys:
        continue
    seen_keys.add(key)

    with open(t) as f:
        reader = csv.DictReader(f)
        if 'gripper_logit_margin' not in (reader.fieldnames or []):
            continue
        rows = list(reader)

    window_rows = [r for r in rows if r.get('in_window') == 'True']
    if len(window_rows) < 2:
        continue

    margins = []
    entropies = []
    open_masses = []
    top2s = []
    all_ents = []
    for r in window_rows:
        try:
            m_val = float(r.get('gripper_logit_margin', 0))
            if abs(m_val) > 0.001:
                margins.append(m_val)
        except: pass
        try: entropies.append(float(r.get('gripper_entropy', 0)))
        except: pass
        try: open_masses.append(float(r.get('gripper_logit_open_mass', 0)))
        except: pass
        try: top2s.append(float(r.get('gripper_top2_margin', 0)))
        except: pass
        try: all_ents.append(float(r.get('all_action_entropy', 0)))
        except: pass

    if len(margins) < 2:
        continue

    ma = np.array(margins)
    n_low = int(np.sum(np.abs(ma) < 0.1))
    streak = max_streak = 0
    for mv in ma:
        if abs(mv) < 0.1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    label = label_keys.get(key, {})

    candidates.append({
        'task_key': task, 'state_id': sid,
        'window_start': ws, 'window_end': we,
        'n_steps': len(window_rows),
        'logit_margin_min': float(np.min(ma)),
        'logit_margin_max': float(np.max(ma)),
        'logit_margin_mean': float(np.mean(ma)),
        'logit_margin_std': float(np.std(ma)),
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
    })

print('Candidates: %d (pos=%d, neg=%d, ignore=%d)' % (
    len(candidates),
    sum(1 for c in candidates if c['label_status'] == 'positive'),
    sum(1 for c in candidates if c['label_status'] == 'negative'),
    sum(1 for c in candidates if c['label_status'] == 'ignore')))

# Write feature distribution
dist_csv = repo + '/tables/online_safe_v3_logit_feature_distribution.csv'
cols = ['task_key','state_id','window_start','window_end','label_status','taxonomy',
    'logit_margin_min','logit_margin_max','logit_margin_mean','logit_margin_std','logit_margin_range',
    'entropy_mean','entropy_max','top2_margin_min','top2_margin_mean',
    'open_mass_max','open_mass_mean','all_action_entropy_mean',
    'low_margin_step_count','longest_low_margin_streak','n_steps']
with open(dist_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader(); w.writerows(candidates)
print('Wrote %s' % dist_csv)

# Per-candidate detail
print()
for c in sorted(candidates, key=lambda x: (x['label_status'], x['logit_margin_mean'])):
    print('%s s%s [%s,%s] | %s | margin=[%.1f,%.1f] mean=%.1f std=%.1f range=%.1f ent=%.6f top2=%.1f low=%d strk=%d' % (
        c['task_key'], c['state_id'], c['window_start'], c['window_end'],
        c['label_status'], c['logit_margin_min'], c['logit_margin_max'],
        c['logit_margin_mean'], c['logit_margin_std'], c['logit_margin_range'],
        c['entropy_mean'], c['top2_margin_mean'],
        c['low_margin_step_count'], c['longest_low_margin_streak']))

# Train detector
train = [c for c in candidates if c['label_status'] in ('positive','negative')]
if len(train) >= 6 and sum(1 for c in train if c['label_status']=='positive') >= 2:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix

    FEATS = ['logit_margin_mean','logit_margin_std','logit_margin_min','logit_margin_range',
             'entropy_mean','entropy_max','top2_margin_mean','top2_margin_min',
             'open_mass_max','all_action_entropy_mean',
             'low_margin_step_count','longest_low_margin_streak']

    X = np.array([[float(c.get(f, 0) or 0) for f in FEATS] for c in train])
    y = np.array([1 if c['label_status']=='positive' else 0 for c in train])
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]; train_arr = [train[i] for i in range(len(train)) if valid[i]]
    y = y[valid]

    if len(set(y)) >= 2:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        tasks = sorted(set(c['task_key'] for c in train_arr))
        preds = np.zeros(len(y))
        for ht in tasks:
            tr_idx = [i for i,c in enumerate(train_arr) if c['task_key']!=ht]
            te_idx = [i for i,c in enumerate(train_arr) if c['task_key']==ht]
            if len(tr_idx)<2 or len(te_idx)==0: continue
            lr = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
            lr.fit(X[tr_idx], y[tr_idx])
            preds[te_idx] = lr.predict(X[te_idx])

        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        bal = balanced_accuracy_score(y, preds)
        pos_rec = recall_score(y, preds, pos_label=1) if 1 in y else 0
        neg_rec = recall_score(y, preds, pos_label=0) if 0 in y else 0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0

        print()
        print('=== DETECTOR ===')
        print('N=%d pos=%d neg=%d' % (len(y), int(sum(y)), int(len(y)-sum(y))))
        print('TP=%d FP=%d TN=%d FN=%d' % (tp,fp,tn,fn))
        print('BalAcc=%.4f PosRec=%.4f NegRec=%.4f FPR=%.4f' % (bal,pos_rec,neg_rec,fpr))
        print()
        print('Phase-only v1: BalAcc=0.611 PosRec=0.222 NegRec=1.000')
        print('Logit-margin:   BalAcc=%.4f PosRec=%.4f NegRec=%.4f' % (bal,pos_rec,neg_rec))

        for i, c in enumerate(train_arr):
            marker = 'OK' if preds[i]==y[i] else ('MISSED_POS' if y[i]==1 else 'FP')
            print('  %s s%s [%s,%s] true=%d pred=%d %s' % (
                c['task_key'],c['state_id'],c['window_start'],c['window_end'],
                int(y[i]),int(preds[i]),marker))
