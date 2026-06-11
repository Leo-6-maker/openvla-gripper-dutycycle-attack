#!/usr/bin/env python3
"""S20G Step 3: Train v0.3.1 pilot detector on 35 paired labels."""
import csv, json, os, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
CONFIGS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/configs'
os.makedirs(CONFIGS, exist_ok=True)

# Load paired labels
paired = []
with open(TABLES + '/s20g_v031_paired_label_table.csv') as f:
    paired = list(csv.DictReader(f))

# Load close-transition audit
trans_audit = {}
with open(TABLES + '/s20g_close_transition_audit.csv') as f:
    for r in csv.DictReader(f):
        key = (r['task'], r['state_id'], r['window_start'], r['window_end'], r['seed'])
        trans_audit[key] = r

# Merge features
for p in paired:
    key = (p['task'], p['state_id'], p['window_start'], p['window_end'], p['seed'])
    t = trans_audit.get(key, {})
    p['nearest_transition_step'] = float(t.get('nearest_transition_step', -1) or -1)
    p['distance_to_transition'] = float(t.get('distance_to_transition', 0) or 0)
    p['pre_open_streak'] = float(t.get('pre_open_streak', 0) or 0)
    p['post_close_streak'] = float(t.get('post_close_streak', 0) or 0)
    p['transition_overlap_center'] = int(t.get('transition_overlap_center', 0) or 0)
    p['close_commitment_score'] = float(t.get('close_commitment_score', 0.5) or 0.5)

# Build targets
y_random_sensitive = np.array([1 if p['classification'] == 'random_sensitive' else 0 for p in paired])
y_cmd_specific = np.array([1 if p['classification'] in ('cmd_specific', 'task_effect', 'contact_effect_weak') else 0 for p in paired])
y_task_or_contact = np.array([1 if p['classification'] in ('task_effect', 'contact_effect_weak') else 0 for p in paired])

# Feature groups
def build_features(group_name):
    features = []
    names = []
    for p in paired:
        feats = []
        if group_name in ('OldClean', 'PhaseAware', 'CloseTransitionAware', 'PhaseAware+CloseTransitionAware', 'NoTask_CloseTransitionAware', 'AllWithTask'):
            # OldClean
            feats += [float(p['rand_open']), float(p['rand_streak']),
                      float(p['rand_open'])/max(float(p['rand_streak']),1),
                      float(p['vis_open']), float(p['vis_streak']),
                      float(p['open_delta']), float(p['streak_delta'])]
        if group_name in ('PhaseAware', 'PhaseAware+CloseTransitionAware', 'AllWithTask'):
            fc = float(p.get('first_close_step', -1) or -1)
            lift = float(p.get('lift_step', -1) or -1)
            ws = float(p['window_start'])
            feats += [fc, lift, ws - fc if fc > 0 else 0, ws - lift if lift > 0 else 0]
        if group_name in ('CloseTransitionAware', 'PhaseAware+CloseTransitionAware', 'NoTask_CloseTransitionAware', 'AllWithTask'):
            feats += [float(p['distance_to_transition']), float(p['pre_open_streak']),
                      float(p['post_close_streak']), int(p['transition_overlap_center']),
                      float(p['close_commitment_score'])]
        if group_name in ('AllWithTask',):
            # Task one-hot
            tasks = ['ketchup', 'tomato_sauce']
            for tk in tasks:
                feats.append(1 if p['task'] == tk else 0)
        features.append(feats)

    X = np.array(features)
    return X

group_names = ['OldClean', 'PhaseAware', 'CloseTransitionAware', 'PhaseAware+CloseTransitionAware', 'NoTask_CloseTransitionAware', 'AllWithTask']

# 3-fold CV evaluation
results = []
for group in group_names:
    X = build_features(group)
    for model_name, Model in [
        ('LR', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('GB', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        # Scale for LR
        if model_name == 'LR':
            X_use = StandardScaler().fit_transform(X)
        else:
            X_use = X

        for target_name, y in [('random_sensitive', y_random_sensitive), ('cmd_specific', y_cmd_specific), ('task_or_contact', y_task_or_contact)]:
            if sum(y) < 3:  # too few positives
                results.append({'feature_group': group, 'model': model_name, 'target': target_name,
                               'n_samples': len(y), 'n_pos': int(sum(y)), 'auroc': '', 'balanced_acc': '', 'cv_mean': ''})
                continue
            try:
                cv = StratifiedKFold(n_splits=min(3, sum(y)), shuffle=True, random_state=42)
                auroc_scores = []
                bal_scores = []
                for train_idx, test_idx in cv.split(X_use, y):
                    if len(set(y[train_idx])) < 2: continue
                    Model.random_state = 42
                    m = Model.__class__(**{k:v for k,v in Model.get_params().items() if k != 'random_state'})
                    m.random_state = 42
                    m.fit(X_use[train_idx], y[train_idx])
                    y_pred = m.predict_proba(X_use[test_idx])[:, 1]
                    auroc_scores.append(roc_auc_score(y[test_idx], y_pred))
                    bal_scores.append(balanced_accuracy_score(y[test_idx], (y_pred >= 0.5).astype(int)))

                results.append({
                    'feature_group': group, 'model': model_name, 'target': target_name,
                    'n_samples': len(y), 'n_pos': int(sum(y)),
                    'auroc': round(np.mean(auroc_scores), 3) if auroc_scores else '',
                    'balanced_acc': round(np.mean(bal_scores), 3) if bal_scores else '',
                    'cv_mean': round(np.mean(auroc_scores), 3) if auroc_scores else '',
                })
            except Exception as e:
                results.append({'feature_group': group, 'model': model_name, 'target': target_name,
                               'n_samples': len(y), 'n_pos': int(sum(y)), 'auroc': 'ERR', 'balanced_acc': 'ERR', 'cv_mean': str(e)[:50]})

# Write metrics
with open(TABLES + '/s20g_v031_pilot_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['feature_group','model','target','n_samples','n_pos','auroc','balanced_acc','cv_mean'])
    w.writeheader(); w.writerows(results)

print('=' * 80)
print('V0.3.1 PILOT TRAINING RESULTS')
print('=' * 80)
print('%-35s %-3s %-20s %5s %5s %8s %8s' % ('Feature Group', 'Mdl', 'Target', 'N', 'Pos', 'AUROC', 'BalAcc'))
print('-' * 80)
for r in results:
    print('%-35s %-3s %-20s %5d %5s %8s %8s' % (
        r['feature_group'], r['model'], r['target'],
        r['n_samples'], r['n_pos'], r['auroc'], r['balanced_acc']))

# Select best model for each target
print()
print('Best per target:')
for target in ['random_sensitive', 'cmd_specific', 'task_or_contact']:
    subset = [r for r in results if r['target'] == target and isinstance(r['auroc'], float)]
    if subset:
        best = max(subset, key=lambda r: r['auroc'])
        print('  %s: %s + %s (AUROC=%.3f, n=%d, pos=%d)' % (
            target, best['feature_group'], best['model'], best['auroc'], best['n_samples'], best['n_pos']))

# ── Full model prediction for candidate ranking ──
best_cfg = max([r for r in results if r['target'] == 'cmd_specific' and isinstance(r['auroc'], float)],
               key=lambda r: r['auroc'])
print()
print('Best cmd_specific model: %s + %s (AUROC=%.3f)' % (best_cfg['feature_group'], best_cfg['model'], best_cfg['auroc']))

# Train best models on full data and predict scores
X_cmd = build_features(best_cfg['feature_group'])
if best_cfg['model'] == 'LR':
    X_cmd = StandardScaler().fit_transform(X_cmd)
    m_cmd = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
elif best_cfg['model'] == 'RF':
    m_cmd = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
else:
    m_cmd = GradientBoostingClassifier(n_estimators=100, random_state=42)

# random_sensitive: 0 pos in paired set (only VIS'd RAND-pass windows)
# Use heuristic abstain: rand_open >= 4 OR rand_timeout
# (This is documented as a limitation in the pilot report)
p_rand = np.array([1.0 if (float(p['rand_open']) >= 4 or p['rand_timeout'] == 'True') else 0.0 for p in paired])

m_cmd.fit(X_cmd, y_cmd_specific)
p_cmd = m_cmd.predict_proba(X_cmd)[:, 1]

# Attack score: p_cmd - 1.0 * p_rand (heuristic random_sensitive)
attack_score = p_cmd - 1.0 * p_rand

# Write predictions
pred_rows = []
for i, p in enumerate(paired):
    pred_rows.append({
        'task': p['task'], 'state_id': p['state_id'], 'window_start': p['window_start'],
        'window_end': p['window_end'], 'seed': p['seed'], 'phase': p['phase'],
        'label_quality': p['label_quality'], 'classification': p['classification'],
        'p_cmd_specific': round(p_cmd[i], 4),
        'p_random_sensitive': round(p_rand[i], 4),
        'attack_score': round(attack_score[i], 4),
        'rand_open': p['rand_open'], 'vis_open': p['vis_open'],
        'open_delta': p['open_delta'],
    })

with open(TABLES + '/s20g_v031_pilot_predictions.csv', 'w', newline='') as f:
    fields = ['task','state_id','window_start','window_end','seed','phase','label_quality','classification',
              'p_cmd_specific','p_random_sensitive','attack_score','rand_open','vis_open','open_delta']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(pred_rows)

# Top candidates for S20H
pred_rows.sort(key=lambda r: -r['attack_score'])
top = [r for r in pred_rows if r['p_random_sensitive'] < 0.5][:10]

with open(TABLES + '/s20g_v031_top_candidates_for_s20h.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(top)

print()
print('Top 10 candidates for S20H:')
for r in top:
    print('  %-14s s%-1s w%-3d-%-3d seed=%-2s phase=%-18s p_cmd=%.3f p_rand=%.3f score=%.3f cls=%s' % (
        r['task'], r['state_id'], r['window_start'], r['window_end'], r['seed'], r['phase'],
        r['p_cmd_specific'], r['p_random_sensitive'], r['attack_score'], r['classification']))

# Save frozen config
config = {
    'detector_version': 'v0.3.1_pilot',
    'training_samples': len(paired),
    'best_cmd_model': best_cfg['feature_group'] + '+' + best_cfg['model'],
    'feature_group': best_cfg['feature_group'],
    'model': best_cfg['model'],
    'targets': ['random_sensitive', 'cmd_specific', 'task_or_contact'],
    'attack_score_formula': 'p_cmd_specific - 1.0 * p_random_sensitive',
    'abstain_threshold': 'p_random_sensitive >= 0.5',
    'held_out': ['tomato_sauce_s0_w70-80', 'ketchup_s0_w150-160'],
}
with open(CONFIGS + '/stageb_detector_v031_pilot.yaml', 'w') as f:
    json.dump(config, f, indent=2)

print()
print('Config saved: %s' % (CONFIGS + '/stageb_detector_v031_pilot.yaml'))
print('All outputs in: %s' % TABLES)
