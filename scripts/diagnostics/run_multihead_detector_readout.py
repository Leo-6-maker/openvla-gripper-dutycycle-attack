#!/usr/bin/env python3
"""Multi-head detector readout on 72-pair RC1a pool (14cfabe).

Heads:
  A: cmd_specific vs negative_clean (exclude abstain_any, unstable)
  B: vis_specific_phys strict vs negative_clean (exclude shared_qpos, rand_phys, unstable)
  C: abstain_any vs negative_clean (exclude cmd_specific, vis_specific_phys, unstable)

Feature groups: TaskOnly, CleanNoTask*, CleanWithTask*, Clean+Task+Timing
Metrics: P@K, enrichment@K, AUROC, AUPRC per head, per-task breakdown
"""
import csv, os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict, Counter

LABELS_CSV = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
OUT_DIR = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/multihead_readout_14cfabe'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load labels ──
rows = []
with open(LABELS_CSV, 'r') as f:
    for r in csv.DictReader(f):
        rows.append(r)
print('Loaded %d pairs' % len(rows))

# ── Head definitions ──
# Head A: cmd_specific
pos_a = [r for r in rows if r['cmd_specific'] == '1' and r['unstable_or_edge'] == '0'
         and r['abstain_any'] == '0']
neg_a = [r for r in rows if r['negative_clean'] == '1' and r['unstable_or_edge'] == '0']
pool_a = pos_a + neg_a
print('\nHead A (cmd_specific): pos=%d neg=%d total=%d' % (len(pos_a), len(neg_a), len(pool_a)))

# Head B: vis_specific_phys STRICT (NOT shared_qpos, NOT rand_phys)
pos_b = [r for r in rows if r['vis_specific_phys'] == '1' and r['shared_qpos_response'] == '0'
         and r['rand_phys_confound'] == '0' and r['unstable_or_edge'] == '0'
         and r['cmd_specific'] == '0']
neg_b = [r for r in rows if r['negative_clean'] == '1' and r['unstable_or_edge'] == '0']
pool_b = pos_b + neg_b
print('Head B (vis_specific_phys strict): pos=%d neg=%d total=%d' % (len(pos_b), len(neg_b), len(pool_b)))

# Head C: abstain_any
pos_c = [r for r in rows if r['abstain_any'] == '1' and r['unstable_or_edge'] == '0']
neg_c = [r for r in rows if r['negative_clean'] == '1' and r['unstable_or_edge'] == '0'
         and r['cmd_specific'] == '0' and r['vis_specific_phys'] == '0']
pool_c = pos_c + neg_c
print('Head C (abstain_any): pos=%d neg=%d total=%d' % (len(pos_c), len(neg_c), len(pool_c)))


def extract_features(pool):
    """Build feature matrix from label rows."""
    tasks = sorted(set(r['task_key'] for r in pool))
    task_enc = OneHotEncoder(sparse_output=False)
    task_idx = np.array([tasks.index(r['task_key']) for r in pool]).reshape(-1, 1)
    task_oh = task_enc.fit_transform(task_idx)

    stratum_enc = OneHotEncoder(sparse_output=False)
    stratum_map = {'high_opportunity': 0, 'medium_opportunity': 1, 'hard_negative_or_idle': 2}
    stratum_idx = np.array([stratum_map.get(r.get('stratum', 'medium_opportunity'), 1)
                            for r in pool]).reshape(-1, 1)
    stratum_oh = stratum_enc.fit_transform(stratum_idx)

    def f(r, field, default=0.0):
        try: return float(r.get(field, default) or default)
        except: return default

    X_clean = np.column_stack([
        [f(r, 'clean_open_count') for r in pool],
        [f(r, 'clean_open_frac') for r in pool],
        [f(r, 'raw_gripper_mean') for r in pool],
        [f(r, 'raw_gripper_max') for r in pool],
        [f(r, 'qpos_pre') for r in pool],
        [f(r, 'qpos_mean') for r in pool],
    ])

    ws_arr = np.array([int(r['window_start']) for r in pool])
    we_arr = np.array([int(r['window_end']) for r in pool])
    wc_arr = (ws_arr + we_arr) / 2.0
    max_step_arr = np.array([f(r, 'actual_max_step', 299) for r in pool])
    rel_timing = wc_arr / np.maximum(max_step_arr, 1)

    ss = StandardScaler()
    X_clean_scaled = ss.fit_transform(X_clean)

    feature_groups = {
        'TaskOnly': task_oh,
        'CleanNoTaskNoTiming': X_clean_scaled,
        'CleanNoTaskWithTiming': np.column_stack([X_clean_scaled, wc_arr, rel_timing]),
        'Clean+Task': np.column_stack([X_clean_scaled, task_oh]),
        'Clean+Task+Timing': np.column_stack([X_clean_scaled, task_oh, wc_arr, rel_timing]),
    }
    # Groups
    groups = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r['seed']) for r in pool])
    return feature_groups, groups, tasks


def run_head(head_name, pool, y):
    """Run all feature groups for one head."""
    print('\n' + '=' * 70)
    print('HEAD: %s (N=%d, pos=%d neg=%d)' %
          (head_name, len(pool), sum(y), len(y) - sum(y)))
    print('=' * 70)

    feature_groups, groups, tasks = extract_features(pool)
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        print('  SKIP: only %d groups' % len(set(groups)))
        return {}

    results = {}
    for fg_name, X in feature_groups.items():
        n_features = X.shape[1]
        model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        gkf = GroupKFold(n_splits=n_splits)
        probas = cross_val_predict(model, X, y, groups=groups, cv=gkf, method='predict_proba')[:, 1]

        # AUROC, AUPRC
        auroc = roc_auc_score(y, probas)
        auprc = average_precision_score(y, probas)

        # P@K
        n_pos = sum(y)
        k_vals = [min(5, n_pos), min(10, n_pos), min(n_pos, max(5, n_pos // 2))]
        p_at_k = {}
        enrich_at_k = {}
        order = np.argsort(-probas)
        base_rate = n_pos / len(y) if len(y) > 0 else 0
        for k in k_vals:
            if k == 0: continue
            top_k = order[:k]
            n_hit = sum(y[i] for i in top_k)
            p = n_hit / k if k > 0 else 0
            p_at_k[k] = round(p, 3)
            enrich_at_k[k] = round(p / base_rate, 2) if base_rate > 0 else 0

        results[fg_name] = {
            'n_features': n_features, 'auroc': round(auroc, 3),
            'auprc': round(auprc, 3), 'p_at_k': p_at_k, 'enrich_at_k': enrich_at_k,
        }

        k_str = ' '.join('P@%d=%.2f' % (k, p_at_k[k]) for k in sorted(p_at_k))
        e_str = ' '.join('E@%d=%.1fx' % (k, enrich_at_k[k]) for k in sorted(enrich_at_k))
        print('  %-25s AUROC=%.3f AUPRC=%.3f  %s  %s' %
              (fg_name, auroc, auprc, k_str, e_str))

    # ── Per-task AUROC ──
    print('  Per-task AUROC:')
    for tk in tasks:
        mask = np.array([r['task_key'] == tk for r in pool])
        if sum(mask) < 5 or len(set(y[mask])) < 2:
            continue
        n_tk = sum(mask)
        n_pos_tk = sum(y[mask])
        for fg_name in ['TaskOnly', 'CleanNoTaskNoTiming', 'Clean+Task+Timing']:
            X = feature_groups[fg_name]
            try:
                auroc_tk = roc_auc_score(y[mask], probas[mask])
            except:
                auroc_tk = float('nan')
            if fg_name == 'Clean+Task+Timing':
                print('    %-20s N=%2d pos=%2d AUROC=%.3f' %
                      (tk, n_tk, n_pos_tk, auroc_tk))

    return results


# ── Run heads ──
all_results = {}

y_a = np.array([1 if r['cmd_specific'] == '1' else 0 for r in pool_a])
all_results['A_cmd_specific'] = run_head('A: cmd_specific', pool_a, y_a)

y_b = np.array([1 if r['vis_specific_phys'] == '1' else 0 for r in pool_b])
all_results['B_vis_specific_phys'] = run_head('B: vis_specific_phys strict', pool_b, y_b)

y_c = np.array([1 if r['abstain_any'] == '1' else 0 for r in pool_c])
all_results['C_abstain_any'] = run_head('C: abstain_any', pool_c, y_c)

# ── Save summary ──
summary_path = os.path.join(OUT_DIR, 'readout_summary.csv')
with open(summary_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['head', 'feature_group', 'n_features', 'n_samples', 'n_pos', 'n_neg',
                'auroc', 'auprc', 'p_at_5', 'p_at_10', 'enrich_5', 'enrich_10'])
    for head_key, head_results in all_results.items():
        n_pos = sum(y_a) if 'A_' in head_key else (sum(y_b) if 'B_' in head_key else sum(y_c))
        pool_n = len(pool_a) if 'A_' in head_key else (len(pool_b) if 'B_' in head_key else len(pool_c))
        for fg, res in head_results.items():
            p5 = res['p_at_k'].get(5, res['p_at_k'].get(min(res['p_at_k'].keys()), 0)) if res['p_at_k'] else 0
            p10 = res['p_at_k'].get(10, res['p_at_k'].get(min(res['p_at_k'].keys()), 0)) if res['p_at_k'] else 0
            e5 = res['enrich_at_k'].get(5, res['enrich_at_k'].get(min(res['enrich_at_k'].keys()), 0)) if res['enrich_at_k'] else 0
            e10 = res['enrich_at_k'].get(10, res['enrich_at_k'].get(min(res['enrich_at_k'].keys()), 0)) if res['enrich_at_k'] else 0
            w.writerow([head_key, fg, res['n_features'], pool_n, n_pos, pool_n - n_pos,
                        res['auroc'], res['auprc'], p5, p10, e5, e10])

print('\nSummary saved: %s' % summary_path)
print('Output dir: %s' % OUT_DIR)
