#!/usr/bin/env python3
"""Detector v0 SilverOverride — CPU-only."""
import csv, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict, Counter

BRONZE_LABELS = '/tmp/bronze_labels.csv'
SILVER_LABELS = '/tmp/silver_p1a_labels.csv'
CANDIDATES = '/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608'
os.makedirs(OUT, exist_ok=True)

with open(BRONZE_LABELS) as f: bronze = {r['pair_id']: r for r in csv.DictReader(f)}
with open(SILVER_LABELS) as f: silver = list(csv.DictReader(f))
with open(CANDIDATES) as f: candidates = list(csv.DictReader(f))

# Silver stability grouping
sp = defaultdict(list)
for r in silver:
    parent = r['pair_id'].replace('silver_bronze_', 'bronze_').rsplit('_r', 1)[0]
    sp[parent].append(r)

silver_override = {}
for parent, reps in sp.items():
    n = len(reps)
    vc = sum(1 for r in reps if int(r.get('vis_open_count', 0)) >= 6)
    vp = sum(1 for r in reps if float(r.get('vis_qpos_delta_shifted', 0)) >= 0.01)
    rc_r = sum(1 for r in reps if int(r.get('rand_open_count', 0)) >= 6)
    vr = vc / max(n, 1); rr = rc_r / max(n, 1); pr = vp / max(n, 1)
    if vr >= 0.67 and rr <= 0.33:
        silver_override[parent] = {'cmd': 1, 'phys': 1 if pr >= 0.67 else 0, 'rand': 0, 'tier': 'silver_cmd'}
    elif pr >= 0.67 and rr <= 0.33:
        silver_override[parent] = {'cmd': 0, 'phys': 1, 'rand': 0, 'tier': 'silver_phys'}
    elif rr >= 0.67:
        silver_override[parent] = {'cmd': 0, 'phys': 0, 'rand': 1, 'tier': 'silver_rand'}
    elif vr <= 0.33 and rr <= 0.33:
        silver_override[parent] = {'cmd': 0, 'phys': 0, 'rand': 0, 'tier': 'silver_hard_neg'}
    else:
        silver_override[parent] = {'cmd': -1, 'phys': -1, 'rand': -1, 'tier': 'silver_unstable'}

# Build feature table
cand_lookup = {}
for c in candidates:
    key = (c['task_key'], c['state_id'], c['window_start'], c['window_end'])
    cand_lookup[key] = c

feature_rows = []
for pid, bl in bronze.items():
    key = (bl['task_key'], bl['state_id'], bl['window_start'], bl['window_end'])
    c = cand_lookup.get(key, {})
    if not c:
        continue
    ov = silver_override.get(pid, {})
    cmd_label = ov.get('cmd', int(bl['cmd_susceptible'])) if ov.get('cmd', -1) >= 0 else -1
    phys_label = ov.get('phys', int(bl['vis_specific_physical_response'])) if ov.get('phys', -1) >= 0 else -1
    rand_label = ov.get('rand', int(bl['random_confounded'])) if ov.get('rand', -1) >= 0 else -1
    if cmd_label < 0:
        continue
    feature_rows.append({
        'pair_id': pid, 'task_key': bl['task_key'], 'state_id': bl['state_id'],
        'window_start': int(bl['window_start']), 'window_end': int(bl['window_end']),
        'target_cmd': cmd_label, 'target_phys': phys_label, 'target_rand': rand_label,
        'label_tier': ov.get('tier', 'bronze_only'),
        'clean_open_count': int(c.get('clean_open_count', 0)),
        'clean_open_frac': float(c.get('clean_open_frac', 0)),
        'raw_gripper_mean': float(c.get('raw_gripper_mean', 0)),
        'qpos_mean': float(c.get('qpos_abs_sum_window_mean', 0)),
        'qpos_slope': float(c.get('qpos_abs_sum_slope', 0)),
        'eef_disp': float(c.get('eef_displacement', 0)),
        'stratum': c.get('candidate_stratum', '?'),
    })

print('Feature rows:', len(feature_rows))
print('cmd_pos=%d phys_pos=%d rand_pos=%d' % (
    sum(1 for r in feature_rows if r['target_cmd'] == 1),
    sum(1 for r in feature_rows if r['target_phys'] == 1),
    sum(1 for r in feature_rows if r['target_rand'] == 1)))
print('tiers:', dict(Counter(r['label_tier'] for r in feature_rows)))

# Features
tasks = sorted(set(r['task_key'] for r in feature_rows))
task_ids = np.array([tasks.index(r['task_key']) for r in feature_rows])
stratum_map = {'high_opportunity': 0, 'medium_opportunity': 1, 'hard_negative_or_idle': 2}
stratum_ids = np.array([stratum_map.get(r['stratum'], 1) for r in feature_rows])
X = np.column_stack([
    task_ids, stratum_ids,
    [r['clean_open_count'] for r in feature_rows],
    [r['clean_open_frac'] for r in feature_rows],
    [r['raw_gripper_mean'] for r in feature_rows],
    [r['qpos_mean'] for r in feature_rows],
    [r['qpos_slope'] for r in feature_rows],
    [r['eef_disp'] for r in feature_rows],
])
groups = np.array(['%s_%s' % (r['task_key'], r['state_id']) for r in feature_rows])

# Evaluate
results = []
for target_name, y in [
    ('cmd_susceptible', np.array([r['target_cmd'] for r in feature_rows])),
    ('vis_specific_physical', np.array([r['target_phys'] for r in feature_rows])),
    ('random_sensitive', np.array([r['target_rand'] for r in feature_rows])),
]:
    pos = int(y.sum())
    if pos < 3:
        results.append({'target': target_name, 'model': 'ALL', 'n_pos': pos,
                        'prevalence': round(pos/len(y), 3), 'P@5': 'underpowered'})
        continue
    prev = pos / len(y)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for name, fn in [
        ('LR', lambda: LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('RF', lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('GB', lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        try:
            y_prob = np.zeros(len(y))
            for ti, te in gkf.split(X, y, groups):
                X_tr, X_te = X[ti], X[te]; y_tr = y[ti]
                s = StandardScaler(); X_tr_s = s.fit_transform(X_tr); X_te_s = s.transform(X_te)
                m = fn(); m.fit(X_tr_s, y_tr)
                y_prob[te] = m.predict_proba(X_te_s)[:, 1]
            order = np.argsort(-y_prob)
            p3 = y[order[:min(3, len(y))]].sum() / min(3, len(y))
            p5 = y[order[:min(5, len(y))]].sum() / min(5, len(y))
            p10 = y[order[:min(10, len(y))]].sum() / min(10, len(y))
            results.append({
                'target': target_name, 'model': name, 'n_pos': pos,
                'prevalence': round(prev, 3),
                'P@3': round(p3, 3), 'P@5': round(p5, 3), 'P@10': round(p10, 3),
                'enrich_P@5': round(p5 / max(prev, 0.01), 1),
                'AUROC': round(roc_auc_score(y, y_prob), 3),
                'AUPRC': round(average_precision_score(y, y_prob), 3),
            })
        except Exception as e:
            results.append({'target': target_name, 'model': name, 'n_pos': pos,
                           'prevalence': round(prev, 3), 'P@5': 'error: %s' % str(e)[:40]})

# Write
with open(os.path.join(OUT, 'silver_override_metrics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['target', 'model', 'n_pos', 'prevalence',
        'P@3', 'P@5', 'P@10', 'enrich_P@5', 'AUROC', 'AUPRC'])
    w.writeheader(); w.writerows(results)

print('\n=== DETECTOR V0 SILVER OVERRIDE ===')
for r in results:
    p5 = r.get('P@5', '?')
    if isinstance(p5, float):
        print('%s %s: P@3=%.2f P@5=%.2f P@10=%.2f enrich=%.1fx AUROC=%.3f' % (
            r['target'], r['model'], r['P@3'], r['P@5'], r['P@10'],
            r['enrich_P@5'], r['AUROC']))
    else:
        print('%s %s: n_pos=%d %s' % (r['target'], r['model'], r['n_pos'], p5))
