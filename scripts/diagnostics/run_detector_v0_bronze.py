#!/usr/bin/env python3
"""Detector v0 Bronze exploratory readout — CPU-only."""
import csv, os, sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

BRONZE_LABELS = '/tmp/bronze_labels.csv'
CANDIDATES = '/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608'
os.makedirs(OUT, exist_ok=True)

# Load data
with open(BRONZE_LABELS) as f:
    labels = list(csv.DictReader(f))
with open(CANDIDATES) as f:
    candidates = list(csv.DictReader(f))

cand_lookup = {}
for c in candidates:
    key = (c['task_key'], c['state_id'], c['window_start'], c['window_end'])
    cand_lookup[key] = c

# Merge
feature_rows = []
for r in labels:
    key = (r['task_key'], r['state_id'], r['window_start'], r['window_end'])
    c = cand_lookup.get(key, {})
    if not c:
        continue
    feature_rows.append({
        'pair_id': r['pair_id'], 'task_key': r['task_key'],
        'state_id': r['state_id'], 'seed': r['seed'],
        'window_start': int(r['window_start']), 'window_end': int(r['window_end']),
        'target_cmd': int(r['cmd_susceptible']),
        'target_phys': int(r['vis_specific_physical_response']),
        'target_rand': int(r['random_confounded']),
        'stratum': c.get('candidate_stratum','?'),
        'clean_open_count': int(c.get('clean_open_count',0)),
        'clean_open_frac': float(c.get('clean_open_frac',0)),
        'raw_gripper_mean': float(c.get('raw_gripper_mean',0)),
        'raw_gripper_max': float(c.get('raw_gripper_max',0)),
        'qpos_pre': float(c.get('qpos_abs_sum_pre',0)),
        'qpos_mean': float(c.get('qpos_abs_sum_window_mean',0)),
        'qpos_max': float(c.get('qpos_abs_sum_window_max',0)),
        'qpos_slope': float(c.get('qpos_abs_sum_slope',0)),
        'eef_disp': float(c.get('eef_displacement',0)),
        'max_step': int(c.get('actual_max_step',0)),
    })

print('Feature rows:', len(feature_rows))

# Encode task
tasks = sorted(set(r['task_key'] for r in feature_rows))
task_to_id = {t: i for i, t in enumerate(tasks)}
task_ids = np.array([task_to_id[r['task_key']] for r in feature_rows])

stratum_map = {'high_opportunity': 0, 'medium_opportunity': 1, 'hard_negative_or_idle': 2}
stratum_ids = np.array([stratum_map.get(r['stratum'], 1) for r in feature_rows])

ws_arr = np.array([r['window_start'] for r in feature_rows])
wc_arr = (ws_arr + np.array([r['window_end'] for r in feature_rows])) / 2.0
rel_timing = wc_arr / np.maximum(np.array([r['max_step'] for r in feature_rows]), 1)

X_base = np.column_stack([
    [r['clean_open_count'] for r in feature_rows],
    [r['clean_open_frac'] for r in feature_rows],
    [r['raw_gripper_mean'] for r in feature_rows],
    [r['raw_gripper_max'] for r in feature_rows],
    [r['qpos_pre'] for r in feature_rows],
    [r['qpos_mean'] for r in feature_rows],
    [r['qpos_max'] for r in feature_rows],
    [r['qpos_slope'] for r in feature_rows],
    [r['eef_disp'] for r in feature_rows],
])

X_no_timing = np.column_stack([task_ids, stratum_ids, X_base])
X_with_timing = np.column_stack([task_ids, stratum_ids, X_base, wc_arr, rel_timing])

y_cmd = np.array([r['target_cmd'] for r in feature_rows])
y_phys = np.array([r['target_phys'] for r in feature_rows])
groups = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r['seed']) for r in feature_rows])

n_splits = min(5, len(set(groups)))

def grouped_eval(name, X, y, groups, model_fn):
    pos = int(y.sum())
    if pos < 3:
        return {'model': name, 'n_pos': pos, 'status': 'underpowered'}
    try:
        gkf = GroupKFold(n_splits=n_splits)
        y_prob = np.zeros(len(y))
        for ti, te in gkf.split(X, y, groups):
            X_tr, X_te = X[ti], X[te]; y_tr = y[ti]
            s = StandardScaler(); X_tr_s = s.fit_transform(X_tr); X_te_s = s.transform(X_te)
            m = model_fn(); m.fit(X_tr_s, y_tr)
            y_prob[te] = m.predict_proba(X_te_s)[:, 1]
        order = np.argsort(-y_prob)
        prev = pos / max(len(y), 1)
        p3 = y[order[:min(3,len(y))]].sum() / min(3, len(y))
        p5 = y[order[:min(5,len(y))]].sum() / min(5, len(y))
        p10 = y[order[:min(10,len(y))]].sum() / min(10, len(y))
        return {'model': name, 'n_pos': pos, 'status': 'ok',
                'AUROC': round(roc_auc_score(y, y_prob), 3),
                'AUPRC': round(average_precision_score(y, y_prob), 3),
                'P@3': round(p3, 3), 'P@5': round(p5, 3), 'P@10': round(p10, 3),
                'prevalence': round(prev, 3),
                'enrich_P@5': round(p5 / max(prev, 0.01), 1)}
    except Exception as e:
        return {'model': name, 'n_pos': pos, 'status': 'error: %s' % str(e)[:60]}

results = []
prev_cmd = y_cmd.sum() / len(y_cmd)
prev_phys = y_phys.sum() / len(y_phys)
print('Prev cmd=%.3f phys=%.3f' % (prev_cmd, prev_phys))

for target_name, y, X_nt in [
    ('command_susceptible', y_cmd, X_no_timing),
    ('vis_specific_physical', y_phys, X_no_timing),
]:
    print('\n--- %s (n=%d) ---' % (target_name, y.sum()))
    for name, fn in [
        ('LR', lambda: LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('RF', lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('GB', lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        r = grouped_eval(name, X_nt, y, groups, fn)
        r['target'] = target_name
        results.append(r)
        print('  %s: AUROC=%s AUPRC=%s P@5=%s enrich=%sx' % (
            name, r.get('AUROC','?'), r.get('AUPRC','?'), r.get('P@5','?'), r.get('enrich_P@5','?')))

# TaskOnly baseline
task_cmd_pred = np.array([1 if r['task_key'] == 'butter' else 0 for r in feature_rows])
task_p5_cmd = y_cmd[task_cmd_pred == 1].sum() / max(task_cmd_pred.sum(), 1)
print('\nTaskOnly (butter) P@5 cmd: %.3f (enrich %.1fx)' % (task_p5_cmd, task_p5_cmd / max(prev_cmd, 0.01)))

# Write outputs
with open(os.path.join(OUT, 'metrics_bronze.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['target','model','n_pos','status','AUROC','AUPRC','P@3','P@5','P@10','prevalence','enrich_P@5'])
    w.writeheader(); w.writerows(results)

with open(os.path.join(OUT, 'feature_table_bronze.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=feature_rows[0].keys())
    w.writeheader(); w.writerows(feature_rows)

print('\nDone. Outputs in', OUT)
