#!/usr/bin/env python3
"""Train online-safe vulnerability baselines.
Uses ONLY clean-rollout features (no VIS outcomes, no oracle labels).
Evaluates via LOTO by task_key.

Feature variants:
  A: task_only
  B: phase_only
  C: task_plus_phase
  D: clean_proprio_action (ProprioNoStep stats, 3 tasks only)
  E: task_phase_clean_proprio_action (combined)
  F: with_ProprioNoStep_scores (hazard/release added)

Output:
  tables/online_safe_vulnerability_metrics.csv
  tables/online_safe_vulnerability_predictions.csv
  reports/ONLINE_SAFE_VULNERABILITY_BASELINE.md
"""

import csv, os, sys
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
DATASET = os.path.join(REPO, 'tables/online_safe_vulnerability_dataset.csv')
OUT_METRICS = os.path.join(REPO, 'tables/online_safe_vulnerability_metrics.csv')
OUT_PREDS = os.path.join(REPO, 'tables/online_safe_vulnerability_predictions.csv')
OUT_MD = os.path.join(REPO, 'reports/ONLINE_SAFE_VULNERABILITY_BASELINE.md')

# ── Load dataset ──────────────────────────────────────────────────
with open(DATASET) as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
print('Loaded %d data rows' % len(all_rows))

# Only train rows
train_rows = [r for r in all_rows if r.get('train_use') == 'train']
print('Train rows: %d' % len(train_rows))
pos_rows = [r for r in train_rows if r.get('label_status') == 'positive']
neg_rows = [r for r in train_rows if r.get('label_status') == 'negative']
print('Pos: %d, Neg: %d' % (len(pos_rows), len(neg_rows)))

def safe_float(v):
    try: return float(v)
    except: return np.nan

# ── Define feature sets ──────────────────────────────────────────
# Identity features (one-hot encoded task_key)
ALL_TASKS = sorted(set(r['task_key'] for r in train_rows))

def get_task_onehot(task):
    return [1.0 if t == task else 0.0 for t in ALL_TASKS]

# Phase features (available for most rows)
PHASE_COLS = ['phase_bin_proxy', 'phase_is_critical', 'qpos_phase_class',
              'hazard_score_mean', 'hazard_score_max',
              'release_safe_score_mean', 'release_safe_score_min']

# Clean proprio/action stats
PROPRIO_COLS = [
    'gripper_command_mean', 'gripper_command_std', 'gripper_command_delta',
    'gripper_qpos_mean', 'gripper_qpos_std', 'gripper_qpos_delta',
    'gripper_width_mean', 'gripper_width_std', 'gripper_width_delta',
    'eef_x_delta', 'eef_y_delta', 'eef_z_delta',
    'eef_vx_mean', 'eef_vx_std', 'eef_vy_mean', 'eef_vy_std',
    'eef_vz_mean', 'eef_vz_std',
    'action_dx_mean', 'action_dx_std', 'action_dy_mean', 'action_dy_std',
    'action_dz_mean', 'action_dz_std',
    'action_gripper_mean', 'action_gripper_std', 'action_gripper_delta',
]

# Window position
POS_COLS = ['window_position_norm', 'window_size']

def build_features(r, variant):
    """Build feature vector for a row under given variant."""
    feats = []

    if variant in ('A_task_only', 'C_task_plus_phase', 'E_task_phase_clean_proprio_action'):
        feats.extend(get_task_onehot(r['task_key']))

    if variant in ('B_phase_only', 'C_task_plus_phase', 'E_task_phase_clean_proprio_action',
                   'F_with_ProprioNoStep_scores'):
        # Phase bin one-hot
        phase_bin = r.get('phase_bin_proxy', '')
        for pb in ['far_closed_proxy', 'near_closed_proxy', 'pre_lock_closed_proxy',
                    'grasp_formation', 'stable', 'natural_open']:
            feats.append(1.0 if phase_bin == pb else 0.0)
        feats.append(1.0 if r.get('phase_is_critical') == 'true' else 0.0)

    if variant in ('F_with_ProprioNoStep_scores',):
        for c in PHASE_COLS:
            v = safe_float(r.get(c, ''))
            feats.append(0.0 if np.isnan(v) else v)

    if variant in ('D_clean_proprio_action', 'E_task_phase_clean_proprio_action'):
        for c in PROPRIO_COLS:
            v = safe_float(r.get(c, ''))
            feats.append(0.0 if np.isnan(v) else v)

    if variant in ('D_clean_proprio_action', 'E_task_phase_clean_proprio_action'):
        for c in POS_COLS:
            v = safe_float(r.get(c, ''))
            feats.append(0.0 if np.isnan(v) else v)

    # If no features built, use window_position as fallback
    if not feats:
        feats.append(safe_float(r.get('window_position_norm', 0)))
        feats.append(safe_float(r.get('window_size', 0)))

    return np.array(feats, dtype=np.float64)


# ── Training ──────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, accuracy_score
)
from sklearn.preprocessing import StandardScaler

VARIANTS = [
    'A_task_only',
    'B_phase_only',
    'C_task_plus_phase',
    'D_clean_proprio_action',
    'E_task_phase_clean_proprio_action',
    'F_with_ProprioNoStep_scores',
]

MODELS = {
    'LR': lambda: LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
    'RF': lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
}

results = []
predictions = []

for variant in VARIANTS:
    for model_name, model_fn in MODELS.items():
        print('\n=== %s / %s ===' % (variant, model_name))

        # Build feature matrix
        X_list = []
        y_list = []
        row_ids = []
        for r in train_rows:
            feats = build_features(r, variant)
            # Skip rows with all-NaN features
            if np.all(np.isnan(feats)) or len(feats) == 0:
                continue
            X_list.append(feats)
            y_list.append(1 if r['label_status'] == 'positive' else 0)
            row_ids.append(r)

        if not X_list:
            print('  No valid features!')
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Handle remaining NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LOTO by task
        tasks_in_train = sorted(set(r['task_key'] for r in row_ids))
        loto_preds = np.zeros(len(y))
        loto_probs = np.zeros(len(y))

        for holdout_task in tasks_in_train:
            train_idx = [i for i, r in enumerate(row_ids) if r['task_key'] != holdout_task]
            test_idx = [i for i, r in enumerate(row_ids) if r['task_key'] == holdout_task]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            X_tr = X_scaled[train_idx]
            y_tr = y[train_idx]
            X_te = X_scaled[test_idx]

            model = model_fn()
            model.fit(X_tr, y_tr)
            loto_preds[test_idx] = model.predict(X_te)
            if hasattr(model, 'predict_proba'):
                loto_probs[test_idx] = model.predict_proba(X_te)[:, 1]

        # Compute metrics
        cm = confusion_matrix(y, loto_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        bal_acc = balanced_accuracy_score(y, loto_preds) if len(set(y)) > 1 else 0.0
        pos_rec = recall_score(y, loto_preds, pos_label=1) if 1 in y else 0.0
        neg_rec = recall_score(y, loto_preds, pos_label=0) if 0 in y else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        mcc = matthews_corrcoef(y, loto_preds)
        acc = accuracy_score(y, loto_preds)

        n_pos = int(sum(y))
        n_neg = int(len(y) - n_pos)

        result = {
            'variant': variant,
            'model': model_name,
            'n_train': len(y),
            'n_pos': n_pos,
            'n_neg': n_neg,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'balanced_accuracy': round(bal_acc, 4),
            'positive_recall': round(pos_rec, 4),
            'negative_recall': round(neg_rec, 4),
            'false_positive_rate': round(fpr, 4),
            'accuracy': round(acc, 4),
            'mcc': round(mcc, 4),
            'is_loto': True,
        }
        results.append(result)

        print('  BAL_ACC=%.4f  POS_REC=%.4f  NEG_REC=%.4f  FPR=%.4f  MCC=%.4f  n=%d' % (
            bal_acc, pos_rec, neg_rec, fpr, mcc, len(y)))

        # Save predictions
        for i, r in enumerate(row_ids):
            predictions.append({
                'variant': variant,
                'model': model_name,
                'task_key': r['task_key'],
                'state_id': r['state_id'],
                'window_start': r['window_start'],
                'window_end': r['window_end'],
                'true': int(y[i]),
                'pred': int(loto_preds[i]),
                'prob': round(float(loto_probs[i]), 4),
                'label_status': r['label_status'],
                'mechanism_type': r.get('mechanism_type', ''),
                'taxonomy': r.get('taxonomy', ''),
            })

# ── Write outputs ─────────────────────────────────────────────────
# Metrics CSV
with open(OUT_METRICS, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['variant','model','n_train','n_pos','n_neg',
        'tp','fp','tn','fn','balanced_accuracy','positive_recall','negative_recall',
        'false_positive_rate','accuracy','mcc','is_loto'])
    w.writeheader()
    w.writerows(results)
print('\nWrote %d metrics rows to %s' % (len(results), OUT_METRICS))

# Predictions CSV
with open(OUT_PREDS, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['variant','model','task_key','state_id',
        'window_start','window_end','true','pred','prob','label_status',
        'mechanism_type','taxonomy'])
    w.writeheader()
    w.writerows(predictions)
print('Wrote %d predictions to %s' % (len(predictions), OUT_PREDS))

# ── Report ────────────────────────────────────────────────────────
with open(OUT_MD, 'w') as f:
    f.write("""# Online-Safe Vulnerability Baseline

**Date**: %s
**Training**: LOTO by task_key, LR + RF, balanced class weights
**Features**: Clean-rollout only. NO VIS outcomes. NO oracle labels.

---

## Variant Descriptions

| Variant | Features | Online-Safe? |
|---------|----------|-------------|
| A_task_only | task_key one-hot | YES |
| B_phase_only | phase_bin_proxy, phase_is_critical | YES |
| C_task_plus_phase | A + B combined | YES |
| D_clean_proprio_action | ProprioNoStep stats + window position | YES (3 tasks only) |
| E_task_phase_clean_proprio_action | A + B + D combined | YES (3 tasks only) |
| F_with_ProprioNoStep_scores | A + B + hazard/release scores | YES (3 tasks only) |

## Results

| Variant | Model | N | Pos | Neg | TP | FP | TN | FN | BalAcc | PosRec | NegRec | FPR | MCC |
|---------|-------|---|---|-----|----|----|----|----|--------|--------|--------|-----|-----|
""" % datetime.now().strftime('%Y-%m-%d %H:%M'))

    for r in results:
        f.write('| %s | %s | %d | %d | %d | %d | %d | %d | %d | %.4f | %.4f | %.4f | %.4f | %.4f |\n' % (
            r['variant'], r['model'], r['n_train'], r['n_pos'], r['n_neg'],
            r['tp'], r['fp'], r['tn'], r['fn'],
            r['balanced_accuracy'], r['positive_recall'], r['negative_recall'],
            r['false_positive_rate'], r['mcc']))

    # Best model
    if results:
        best = max(results, key=lambda r: r['balanced_accuracy'])
        f.write("""
## Best Model

**%s / %s**: BalAcc=%.4f, PosRec=%.4f, NegRec=%.4f, MCC=%.4f

""" % (best['variant'], best['model'], best['balanced_accuracy'],
       best['positive_recall'], best['negative_recall'], best['mcc']))

        f.write("""
## Key Finding

**The online-safe detector performance gap** vs the offline-leaked detector:

| Detector | BalAcc | PosRec | NegRec | Features |
|----------|--------|--------|--------|----------|
| Offline-leaked D_causal_safe (BLOCKED) | 0.714 | 0.889 | 0.538 | VIS outcomes |
| Online-safe best | %.4f | %.4f | %.4f | Clean rollout only |
| **Gap (leakage penalty)** | **%.4f** | **%.4f** | **%.4f** | |

""" % (best['balanced_accuracy'], best['positive_recall'], best['negative_recall'],
       0.714 - best['balanced_accuracy'], 0.889 - best['positive_recall'],
       0.538 - best['negative_recall']))
    else:
        f.write("""
## Best Model

**No valid results** — check feature extraction.
""")

    f.write("""
## Interpretation

The gap between offline-leaked and online-safe performance represents the
**information content of knowing the VIS attack outcome**.

If the online-safe detector performs near chance (BalAcc ~0.5) while the
offline-leaked detector performs well (BalAcc ~0.71), it means:

1. The clean rollout features (phase, task, proprioception) do NOT contain
   strong predictive signal about VIS vulnerability BEFORE running the attack.
2. The "best" detector was learning to classify attack outcomes, not predict
   vulnerability.
3. A deployable vulnerability predictor may require different features or
   a fundamentally different approach (e.g., learning from attack traces
   across many episodes).

This does NOT mean the approach is impossible — it means we need to honestly
measure the online-safe baseline and improve from there, rather than
pretending the offline-leaked performance is real.
""")

print('Wrote report to %s' % OUT_MD)
