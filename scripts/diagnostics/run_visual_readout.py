#!/usr/bin/env python3
"""Visual sidecar readout: compare TaskOnly vs VisualOnly vs Visual+Clean vs Clean+Task.

P2 fixes: fold-specific StandardScaler, per-FG per-task AUROC tracking.

Usage:
  python scripts/diagnostics/run_visual_readout.py \
    --labels .../all_labels_rc1a_14cfabe_72pairs.csv \
    --features .../visual_sidecar_14cfabe_72pairs/feature_index.csv \
    --embeddings .../visual_sidecar_14cfabe_72pairs/embeddings/
"""
import csv, os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--features', required=True)
ap.add_argument('--embeddings', required=True)
ap.add_argument('--out', default='/tmp/visual_readout.csv')
args = ap.parse_args()

# Load labels
labels = {}
with open(args.labels, 'r') as f:
    for r in csv.DictReader(f):
        labels[r['pair_id']] = r

# Load feature index + embeddings
feature_rows = []
with open(args.features, 'r') as f:
    for r in csv.DictReader(f):
        pid = r['pair_id']
        if pid not in labels:
            continue
        embs = {}
        for pos in ['start', 'center', 'end']:
            key = 'emb_%s_file' % pos
            if r.get(key):
                fpath = os.path.join(args.embeddings, r[key])
                try:
                    emb = np.load(fpath)
                    embs[pos] = emb
                except FileNotFoundError:
                    pass
        if 'center' not in embs:
            continue
        r['_emb_start'] = embs.get('start')
        r['_emb_center'] = embs['center']
        r['_emb_end'] = embs.get('end')
        r['_label'] = labels[pid]
        feature_rows.append(r)

print('Matched features: %d/%d' % (len(feature_rows), len(labels)))


def extract_features(rows_for_head):
    tasks = sorted(set(r['_label']['task_key'] for r in rows_for_head))
    task_oh = np.array([[1 if tk == r['_label']['task_key'] else 0 for tk in tasks]
                         for r in rows_for_head], dtype=np.float64)

    def f(r, field, d=0.0):
        try: return float(r['_label'].get(field, d) or d)
        except: return d

    X_clean = np.column_stack([
        [f(r, 'clean_open_count') for r in rows_for_head],
        [f(r, 'clean_open_frac') for r in rows_for_head],
        [f(r, 'raw_gripper_mean') for r in rows_for_head],
        [f(r, 'raw_gripper_max') for r in rows_for_head],
        [f(r, 'qpos_pre') for r in rows_for_head],
        [f(r, 'qpos_mean') for r in rows_for_head],
    ])
    ws_arr = np.array([int(r['_label']['window_start']) for r in rows_for_head])
    we_arr = np.array([int(r['_label']['window_end']) for r in rows_for_head])
    wc_arr = (ws_arr + we_arr) / 2.0
    max_step_arr = np.array([f(r, 'actual_max_step', 299) for r in rows_for_head])
    rel_timing = wc_arr / np.maximum(max_step_arr, 1)

    X_visual = np.stack([r['_emb_center'] for r in rows_for_head])

    groups = np.array(['%s_%s_%s' % (r['_label']['task_key'], r['_label']['state_id'],
                                      r['_label']['seed']) for r in rows_for_head])

    return {
        'TaskOnly': task_oh,
        'CleanNoTaskNoTiming': X_clean,
        'CleanNoTaskWithTiming': np.column_stack([X_clean, wc_arr, rel_timing]),
        'VisualOnly': X_visual,
        'Visual+CleanNoTask': np.column_stack([X_visual, X_clean]),
        'Visual+Task': np.column_stack([X_visual, task_oh]),
        'Visual+Clean+Task': np.column_stack([X_visual, X_clean, task_oh]),
        'Visual+Clean+Task+Timing': np.column_stack([X_visual, X_clean, task_oh, wc_arr, rel_timing]),
    }, groups, tasks


def run_head(head_name, rows_for_head, y):
    print('\n' + '=' * 70)
    print('HEAD: %s (N=%d, pos=%d)' % (head_name, len(rows_for_head), sum(y)))
    print('=' * 70)

    fgs, groups, tasks = extract_features(rows_for_head)
    n_splits = min(5, len(set(groups)))
    if n_splits < 2:
        print('  SKIP: only %d groups' % len(set(groups)))
        return {}

    results = {}
    all_probas = {}

    for fg_name, X in fgs.items():
        probas_full = np.full(len(y), np.nan)
        gkf = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            # P2: scaler fit on TRAIN only, transform both
            ss = StandardScaler()
            X_train_s = ss.fit_transform(X_train)
            X_test_s = ss.transform(X_test)
            model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            model.fit(X_train_s, y_train)
            probas_full[test_idx] = model.predict_proba(X_test_s)[:, 1]

        if np.any(np.isnan(probas_full)):
            print('  %-28s SKIP: NaN probas' % fg_name)
            continue

        probas = probas_full
        auroc = roc_auc_score(y, probas)
        auprc = average_precision_score(y, probas)
        n_pos = sum(y)
        k5 = min(5, n_pos); k10 = min(10, n_pos)
        order = np.argsort(-probas)
        p5 = sum(y[i] for i in order[:k5]) / k5 if k5 > 0 else 0
        p10 = sum(y[i] for i in order[:k10]) / k10 if k10 > 0 else 0
        base = n_pos / len(y) if len(y) > 0 else 0
        e5 = p5 / base if base > 0 else 0; e10 = p10 / base if base > 0 else 0

        results[fg_name] = {'auroc': round(auroc, 3), 'auprc': round(auprc, 3),
                            'p5': round(p5, 2), 'p10': round(p10, 2),
                            'e5': round(e5, 1), 'e10': round(e10, 1),
                            'n_feat': X.shape[1]}
        all_probas[fg_name] = probas

        print('  %-28s AUROC=%.3f AUPRC=%.3f P@5=%.2f P@10=%.2f E@5=%.1fx E@10=%.1fx (dim=%d)' %
              (fg_name, auroc, auprc, p5, p10, e5, e10, X.shape[1]))

    # P2: Per-task AUROC tracked per feature group
    print('  Per-task AUROC:')
    for fg_name, probas in all_probas.items():
        print('    --- %s (AUROC=%.3f) ---' % (fg_name, results[fg_name]['auroc']))
        for tk in tasks:
            mask = np.array([r['_label']['task_key'] == tk for r in rows_for_head])
            if sum(mask) < 3 or len(set(y[mask])) < 2:
                continue
            y_tk = y[mask]; p_tk = probas[mask]
            try:
                auroc_tk = roc_auc_score(y_tk, p_tk)
                print('      %-20s N=%2d pos=%2d AUROC=%.3f' %
                      (tk, sum(mask), sum(y_tk), auroc_tk))
            except:
                print('      %-20s N=%2d pos=%2d AUROC=FAIL' % (tk, sum(mask), sum(y_tk)))

    return results


def build_head(rows, pos_cond):
    pos = [r for r in rows if pos_cond(r['_label'])
           and r['_label'].get('unstable_or_edge', '0') == '0']
    neg = [r for r in rows if r['_label'].get('negative_clean', '0') == '1'
           and r['_label'].get('unstable_or_edge', '0') == '0']
    pool = pos + neg
    y = np.array([1 if pos_cond(r['_label']) else 0 for r in pool])
    return pool, y

pool_a, y_a = build_head(feature_rows,
    lambda l: l.get('cmd_specific', '0') == '1' and l.get('abstain_any', '0') == '0')
pool_c, y_c = build_head(feature_rows,
    lambda l: l.get('abstain_any', '0') == '1')
pool_d, y_d = build_head(feature_rows,
    lambda l: l.get('shared_qpos_response', '0') == '1')

all_results = {}
if len(pool_a) >= 10 and sum(y_a) >= 2:
    all_results['A_cmd_specific'] = run_head('A: cmd_specific', pool_a, y_a)
if len(pool_c) >= 10 and sum(y_c) >= 2:
    all_results['C_abstain_any'] = run_head('C: abstain_any', pool_c, y_c)
if len(pool_d) >= 10 and sum(y_d) >= 2:
    all_results['D_shared_qpos'] = run_head('D: shared_qpos', pool_d, y_d)

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['head', 'feature_group', 'n_features', 'auroc', 'auprc',
                'p_at_5', 'p_at_10', 'enrich_5', 'enrich_10'])
    for hk, hr in all_results.items():
        for fg, res in hr.items():
            w.writerow([hk, fg, res['n_feat'], res['auroc'], res['auprc'],
                        res['p5'], res['p10'], res['e5'], res['e10']])

print('\nOutput: %s' % args.out)
