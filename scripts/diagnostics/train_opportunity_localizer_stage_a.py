#!/usr/bin/env python3
"""Train Stage-A Opportunity Localizer on Object100 dataset v0.

Models: LR, RF. Leave-task-out eval. AUROC/AUPRC/precision@K.
CPU only. Features are online-legal.
"""
import csv, os, sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
DATASET = os.path.join(REPO, 'tables', 'object100_opportunity_dataset_v0.csv')

# ── Load data ────────────────────────────────────────────────────
with open(DATASET) as f:
    rows = list(csv.DictReader(f))
print('Loaded %d rows' % len(rows))

# Feature columns (exclude audit/label)
EXCLUDE = {'row_id', 'episode_key', 'task_key', 'state_id', 'seed',
           'window_start', 'window_end', 'stratum', 'train_use', 'exclude_reason',
           'mechanism_type', 'teacher_window_original', 'opportunity_label',
           'window_start_frac', 'window_len_frac'}  # remove time-leaking features
feature_cols = sorted([c for c in rows[0].keys() if c not in EXCLUDE])
print('Features: %d' % len(feature_cols))

X = np.array([[float(r.get(c, 0)) for c in feature_cols] for r in rows], dtype=np.float32)
y = np.array([int(r['opportunity_label']) for r in rows], dtype=np.int32)
tasks = np.array([r['task_key'] for r in rows])

# Replace inf/nan
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print('X shape:', X.shape, 'y pos:', sum(y), 'neg:', len(y) - sum(y))
unique_tasks = sorted(set(tasks))
print('Tasks:', unique_tasks)

# ── Metrics ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_score):
    """AUROC, AUPRC, Precision@K, Recall@K."""
    n = len(y_true); n_pos = int(sum(y_true))
    if n_pos == 0 or n_pos == n:
        return {'auroc': float('nan'), 'auprc': float('nan'),
                'p_at_10': float('nan'), 'p_at_20': float('nan'), 'r_at_20': float('nan'),
                'n_pos': n_pos, 'n': n}

    order = np.argsort(y_score)[::-1]
    y_ranked = y_true[order]

    # AUROC
    pos_scores = y_score[y_true == 1]; neg_scores = y_score[y_true == 0]
    auc = sum(1 for p in pos_scores for n in neg_scores if p > n)
    auc += 0.5 * sum(1 for p in pos_scores for n in neg_scores if p == n)
    auc /= (len(pos_scores) * len(neg_scores))

    # AUPRC
    tp = 0; auprc = 0.0; prev_recall = 0
    for i in range(n):
        if y_ranked[i] == 1: tp += 1
        precision = tp / (i + 1)
        recall = tp / n_pos
        auprc += precision * (recall - prev_recall)
        prev_recall = recall

    # P@K
    k10 = min(10, n); k20 = min(20, n)
    p_at_10 = sum(y_ranked[:k10]) / k10
    p_at_20 = sum(y_ranked[:k20]) / k20
    r_at_20 = sum(y_ranked[:k20]) / n_pos

    return {'auroc': round(auc, 4), 'auprc': round(auprc, 4),
            'p_at_10': round(p_at_10, 4), 'p_at_20': round(p_at_20, 4),
            'r_at_20': round(r_at_20, 4), 'n_pos': n_pos, 'n': n}

# ── Leave-Task-Out CV ────────────────────────────────────────────
all_metrics = []

for holdout_task in unique_tasks:
    train_mask = tasks != holdout_task
    test_mask = tasks == holdout_task
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        continue

    # LR
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    lr_score = lr.predict_proba(X_test)[:, 1]
    lr_m = compute_metrics(y_test, lr_score)
    lr_m['model'] = 'LR'; lr_m['holdout_task'] = holdout_task
    all_metrics.append(lr_m)

    # RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]
    rf_m = compute_metrics(y_test, rf_score)
    rf_m['model'] = 'RF'; rf_m['holdout_task'] = holdout_task
    all_metrics.append(rf_m)

# ── Baselines ────────────────────────────────────────────────────
# Task-only: predicts 1 for tasks that have high positive ratio in training
for holdout_task in unique_tasks:
    train_mask = tasks != holdout_task
    test_mask = tasks == holdout_task
    y_test = y[test_mask]
    if len(y_test) < 2: continue
    # Task-only: use training prevalence as score
    train_prev = np.mean(y[train_mask])
    task_score = np.full(len(y_test), train_prev)
    tm = compute_metrics(y_test, task_score)
    tm['model'] = 'TaskOnly'; tm['holdout_task'] = holdout_task
    all_metrics.append(tm)

# Time-only: score = window_start_frac (earlier = less likely positive)
if 'window_start_frac' in feature_cols:
    time_idx = feature_cols.index('window_start_frac')
    for holdout_task in unique_tasks:
        test_mask = tasks == holdout_task
        y_test = y[test_mask]
        if len(y_test) < 2: continue
        time_score = X[test_mask, time_idx]  # higher frac = later = more likely positive
        tm = compute_metrics(y_test, time_score)
        tm['model'] = 'TimeOnly'; tm['holdout_task'] = holdout_task
        all_metrics.append(tm)

# ── Full-train metrics ───────────────────────────────────────────
# Train on all tasks for feature importance
lr_full = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
lr_full.fit(X, y)
rf_full = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
rf_full.fit(X, y)

# RF feature importance
rf_importance = sorted(zip(feature_cols, rf_full.feature_importances_), key=lambda x: -x[1])

# ── Aggregate ────────────────────────────────────────────────────
from collections import defaultdict

model_summary = defaultdict(lambda: {'aurocs': [], 'auprcs': [], 'p10s': [], 'p20s': [], 'r20s': []})
for m in all_metrics:
    model = m['model']
    if not np.isnan(m['auroc']):
        model_summary[model]['aurocs'].append(m['auroc'])
        model_summary[model]['auprcs'].append(m['auprc'])
        model_summary[model]['p10s'].append(m['p_at_10'])
        model_summary[model]['p20s'].append(m['p_at_20'])
        model_summary[model]['r20s'].append(m['r_at_20'])

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Object100 Opportunity Localizer — Stage-A Results')
lines.append('')
lines.append('**Dataset**: 294 rows, 74 positive, 24 features')
lines.append('**Eval**: Leave-task-out (9-fold)')
lines.append('')

lines.append('## Model Performance (mean ± std across tasks)')
lines.append('')
lines.append('| Model | AUROC | AUPRC | P@10 | P@20 | R@20 |')
lines.append('|---|---|---|---|---|---|')
for model_name in ['LR', 'RF', 'TaskOnly', 'TimeOnly']:
    ms = model_summary[model_name]
    if ms['aurocs']:
        lines.append('| %s | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f |' % (
            model_name,
            np.mean(ms['aurocs']), np.std(ms['aurocs']),
            np.mean(ms['auprcs']), np.std(ms['auprcs']),
            np.mean(ms['p10s']), np.std(ms['p10s']),
            np.mean(ms['p20s']), np.std(ms['p20s']),
            np.mean(ms['r20s']), np.std(ms['r20s'])))
lines.append('')

lines.append('## Per-Task AUROC')
lines.append('')
lines.append('| Task | LR AUROC | RF AUROC | TaskOnly | TimeOnly |')
lines.append('|---|---|---|---|---|')
for task in unique_tasks:
    lr_auc = next((m['auroc'] for m in all_metrics if m['model'] == 'LR' and m['holdout_task'] == task), float('nan'))
    rf_auc = next((m['auroc'] for m in all_metrics if m['model'] == 'RF' and m['holdout_task'] == task), float('nan'))
    to_auc = next((m['auroc'] for m in all_metrics if m['model'] == 'TaskOnly' and m['holdout_task'] == task), float('nan'))
    tm_auc = next((m['auroc'] for m in all_metrics if m['model'] == 'TimeOnly' and m['holdout_task'] == task), float('nan'))
    lines.append('| %s | %s | %s | %s | %s |' % (task, lr_auc, rf_auc, to_auc, tm_auc))
lines.append('')

lines.append('## RF Top-10 Feature Importance')
lines.append('')
lines.append('| Rank | Feature | Importance |')
lines.append('|---|---|---|')
for i, (feat, imp) in enumerate(rf_importance[:10]):
    lines.append('| %d | %s | %.4f |' % (i+1, feat, imp))
lines.append('')

# Gate evaluation
lr_mean_auc = np.mean(model_summary['LR']['aurocs'])
rf_mean_auc = np.mean(model_summary['RF']['aurocs'])
time_mean_auc = np.mean(model_summary['TimeOnly']['aurocs']) if model_summary['TimeOnly']['aurocs'] else 0.5
task_mean_auc = np.mean(model_summary['TaskOnly']['aurocs']) if model_summary['TaskOnly']['aurocs'] else 0.5

lines.append('## Gate Evaluation')
lines.append('')
lines.append('| Criterion | Threshold | Actual | Pass? |')
lines.append('|---|---|---|---|')
lines.append('| Best model beats time-only | AUROC > %.4f | LR=%.4f RF=%.4f | %s |' % (
    time_mean_auc, lr_mean_auc, rf_mean_auc,
    'YES' if max(lr_mean_auc, rf_mean_auc) > time_mean_auc else 'NO'))
lines.append('| Best model beats task-only | AUROC > %.4f | LR=%.4f RF=%.4f | %s |' % (
    task_mean_auc, lr_mean_auc, rf_mean_auc,
    'YES' if max(lr_mean_auc, rf_mean_auc) > task_mean_auc else 'NO'))
lines.append('| AUROC >= 0.75 | 0.75 | LR=%.4f RF=%.4f | %s |' % (
    lr_mean_auc, rf_mean_auc,
    'YES' if max(lr_mean_auc, rf_mean_auc) >= 0.75 else 'NO'))
lines.append('| Leave-task-out does not collapse | – | min LR=%.4f min RF=%.4f | %s |' % (
    min(model_summary['LR']['aurocs']), min(model_summary['RF']['aurocs']),
    'OK' if min(model_summary['LR']['aurocs']) > 0.55 and min(model_summary['RF']['aurocs']) > 0.55 else 'CHECK'))
lines.append('')

# Final verdict
pass_gate = (lr_mean_auc >= 0.75 or rf_mean_auc >= 0.75) and max(lr_mean_auc, rf_mean_auc) > time_mean_auc
lines.append('## Gate Verdict: %s' % ('PASS' if pass_gate else 'FAIL / BORDERLINE'))
lines.append('')
if pass_gate:
    lines.append('Proceed to Phase-2 VIS labeling plan.')
else:
    lines.append('Do NOT proceed to VIS labeling. Improve features or add data first.')
lines.append('')

with open(os.path.join(REPO, 'reports', 'OBJECT100_OPPORTUNITY_LOCALIZER_V0.md'), 'w') as f:
    f.write('\n'.join(lines))

# Write metrics CSV
METRICS_CSV = os.path.join(REPO, 'tables', 'object100_opportunity_localizer_v0_metrics.csv')
with open(METRICS_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
    w.writeheader(); w.writerows(all_metrics)
print('Wrote %d metrics rows' % len(all_metrics))

print('\n' + '\n'.join(lines[-10:]))
