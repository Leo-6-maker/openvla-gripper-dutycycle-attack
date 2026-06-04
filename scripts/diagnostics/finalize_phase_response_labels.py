#!/usr/bin/env python3
"""Finalize phase-response labels from 9 VIS outcomes + train smoke detector."""
import csv, numpy as np
from collections import defaultdict

OUTCOMES = [
    ('B1', 'alphabet_soup','0', 3,20, 0.027619,False,False,'weak_physical_uncertain',True),
    ('B2b','alphabet_soup','2',11,28, 0.037643,False,True,'action_physical_strong_task_positive',True),
    ('B2b','bbq_sauce',    '0',25,42, 0.038055,True, False,'physical_strong_task_negative',True),
    ('B2b','bbq_sauce',    '4',14,31, 0.037853,True, False,'physical_strong_task_negative',True),
    ('B1', 'butter',       '0',29,46, 0.037905,False,True,'action_physical_strong_task_positive',True),
    ('B2b','butter',       '0',32,49, 0.037934,True, False,'physical_strong_task_negative',True),
    ('B2b','butter',       '2',23,40, 0.037462,False,True,'action_physical_strong_task_positive',True),
    ('B1', 'ketchup',      '0',16,33, 0.038042,False,True,'action_physical_strong_task_positive',True),
    ('B2b','ketchup',      '1',28,45, 0.037948,True, False,'physical_strong_task_negative',True),
]

phase_map = {}
with open('tables/object_teacher_window_phase_descriptors.csv') as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r.get('state_id','0'), r['window_start'], r['window_end'])
        phase_map[key] = r

labels = []
for batch, task, state, ws, we, qpos, done, claim, taxonomy, denom in OUTCOMES:
    ph = phase_map.get((task, state, str(ws), str(we)), {})
    ph_bin = ph.get('phase_bin_proxy','')
    lead = ph.get('relative_lead','')
    if claim: vuln, status, reason = 1, 'positive', ''
    elif 'weak' in taxonomy: vuln, status, reason = '', 'ignore', 'weak_physical_uncertain_manual_audit'
    else: vuln, status, reason = 0, 'negative', taxonomy
    phys = 1 if qpos>=0.03 else (0.5 if qpos>=0.01 else 0)
    labels.append(dict(task_key=task, state_id=state, window_start=ws, window_end=we,
        phase_bin_proxy=ph_bin, lead=lead, VIS_OPEN='18/18',
        qpos_opening_delta=round(qpos,6), qpos_label='strong' if qpos>=0.03 else 'weak',
        done=done, taxonomy=taxonomy, denominator_clean=denom, claim_usable=claim,
        label_action_bridge=1, label_physical_response=phys,
        label_task_failure=0 if done else 1, label_vulnerability_ready=vuln,
        label_status=status, exclusion_or_uncertain_reason=reason))

lf = ['task_key','state_id','window_start','window_end','phase_bin_proxy','lead',
      'VIS_OPEN','qpos_opening_delta','qpos_label','done','taxonomy','denominator_clean','claim_usable',
      'label_action_bridge','label_physical_response','label_task_failure',
      'label_vulnerability_ready','label_status','exclusion_or_uncertain_reason']
with open('tables/object_phase_response_labels_v0.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=lf, extrasaction='ignore')
    w.writeheader(); w.writerows(labels)

pos = [l for l in labels if l['label_vulnerability_ready']==1]
neg = [l for l in labels if l['label_vulnerability_ready']==0]
ign = [l for l in labels if l['label_status']=='ignore']
print('Labels: %d total, pos=%d, neg=%d, ignore=%d' % (len(labels), len(pos), len(neg), len(ign)))
print('Pos: %s' % [p['task_key']+'_s'+p['state_id'] for p in pos])

# Smoke detector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import confusion_matrix

train_rows = [l for l in labels if l['label_status'] in ('positive','negative')]
y = np.array([l['label_vulnerability_ready'] for l in train_rows])
task_groups = np.array([l['task_key'] for l in train_rows])

desc_fields = ['clean_open_ratio','raw_gripper_mean','qpos_start','qpos_min','eef_speed_mean','eef_z_delta']
X = np.zeros((len(train_rows), len(desc_fields)))
for i, l in enumerate(train_rows):
    key = (l['task_key'], l['state_id'], str(l['window_start']), str(l['window_end']))
    ph = phase_map.get(key, {})
    for j, fld in enumerate(desc_fields):
        v = ph.get(fld, 0)
        try: X[i,j] = float(v) if v else 0.0
        except: X[i,j] = 0.0
X = np.nan_to_num(X, 0.0)
print('Training: %d rows, %d features, class=%d/%d' % (len(train_rows), X.shape[1], int(sum(y==1)), int(sum(y==0))))

results = []
for name, model in [('LR', LogisticRegression(max_iter=1000, class_weight='balanced')),
                     ('RF', RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)),
                     ('GB', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))]:
    try:
        logo = LeaveOneGroupOut()
        preds = cross_val_predict(model, X, y, groups=task_groups, cv=logo)
        cm = confusion_matrix(y, preds)
        tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1 = 2*prec*rec/max(prec+rec,1e-8)
        acc = np.mean(preds==y)
        results.append(dict(model=name, accuracy=round(acc,4), precision=round(prec,4),
                            recall=round(rec,4), f1=round(f1,4), tp=tp, fp=fp, fn=fn, tn=tn))
        print('  %s: acc=%.3f prec=%.3f rec=%.3f f1=%.3f' % (name, acc, prec, rec, f1))
    except Exception as e:
        print('  %s: FAIL %s' % (name, str(e)[:50]))

with open('tables/vulnerability_ready_smoke_metrics_v0.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','accuracy','precision','recall','f1','tp','fp','fn','tn'])
    w.writeheader(); w.writerows(results)
print('\nDone.')
