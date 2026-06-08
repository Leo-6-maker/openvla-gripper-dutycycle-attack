#!/usr/bin/env python3
"""Visual sidecar readout: compare TaskOnly vs VisualOnly vs Visual+Clean vs Clean+Task.

Loads 72-pair labels + OpenVLA SigLIP vision backbone embeddings (2176-dim),
runs GroupKFold cross-validation across feature groups for multi-head targets.
CPU-only (embeddings pre-computed).

Usage:
  python scripts/diagnostics/run_visual_readout.py \
    --labels /path/to/all_labels_rc1a_14cfabe_72pairs.csv \
    --features /path/to/visual_sidecar_14cfabe_72pairs/feature_index.csv \
    --embeddings /path/to/visual_sidecar_14cfabe_72pairs/embeddings/
"""
import csv, os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument('--labels', required=True)
ap.add_argument('--features', required=True, help='feature_index.csv from extraction')
ap.add_argument('--embeddings', required=True, help='directory of .npy embedding files')
ap.add_argument('--out', default='/tmp/visual_readout.csv')
args = ap.parse_args()

# ── Load labels ──
labels = {}
with open(args.labels, 'r') as f:
    for r in csv.DictReader(f):
        labels[r['pair_id']] = r
print('Labels: %d' % len(labels))

# ── Load feature index + embeddings ──
feature_rows = []
with open(args.features, 'r') as f:
    for r in csv.DictReader(f):
        pid = r['pair_id']
        if pid not in labels:
            continue
        # Load DINOv2 embeddings
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


# ── Build feature matrices ──
def extract_features(rows_for_head):
    """Build all feature group matrices."""
    tasks = sorted(set(r['_label']['task_key'] for r in rows_for_head))
    task_enc = OneHotEncoder(sparse_output=False)
    task_oh = task_enc.fit_transform(
        np.array([tasks.index(r['_label']['task_key']) for r in rows_for_head]).reshape(-1, 1))

    # Clean proprio
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

    ss = StandardScaler()
    X_clean_scaled = ss.fit_transform(X_clean)

    # Visual (DINOv2 center embedding)
    X_visual = np.stack([r['_emb_center'] for r in rows_for_head])
    X_visual = StandardScaler().fit_transform(X_visual)

    # Visual start+center+end concatenated
    X_viz_triple = []
    for r in rows_for_head:
        parts = []
        for pos in ['start', 'center', 'end']:
            emb = r.get('_emb_%s' % pos)
            parts.append(emb if emb is not None else np.zeros(2176))
        X_viz_triple.append(np.concatenate(parts))
    X_viz_triple = np.stack(X_viz_triple)
    X_viz_triple = StandardScaler().fit_transform(X_viz_triple)

    groups = np.array(['%s_%s_%s' % (r['_label']['task_key'], r['_label']['state_id'],
                                      r['_label']['seed']) for r in rows_for_head])

    return {
        'TaskOnly': task_oh,
        'CleanNoTaskNoTiming': X_clean_scaled,
        'CleanNoTaskWithTiming': np.column_stack([X_clean_scaled, wc_arr, rel_timing]),
        'VisualOnly': X_visual,
        'VisualTriple': X_viz_triple,
        'Visual+CleanNoTask': np.column_stack([X_visual, X_clean_scaled]),
        'Visual+Task': np.column_stack([X_visual, task_oh]),
        'Visual+Clean+Task': np.column_stack([X_visual, X_clean_scaled, task_oh]),
        'Visual+Clean+Task+Timing': np.column_stack([X_visual, X_clean_scaled, task_oh, wc_arr, rel_timing]),
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
    for fg_name, X in fgs.items():
        model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        gkf = GroupKFold(n_splits=n_splits)
        probas = cross_val_predict(model, X, y, groups=groups, cv=gkf, method='predict_proba')[:, 1]

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

        print('  %-28s AUROC=%.3f AUPRC=%.3f P@5=%.2f P@10=%.2f E@5=%.1fx E@10=%.1fx (dim=%d)' %
              (fg_name, auroc, auprc, p5, p10, e5, e10, X.shape[1]))

    # Per-task breakdown for best visual model
    best_fg = max(results.keys(), key=lambda k: results[k]['auroc'])
    print('  Per-task AUROC (best=%s):' % best_fg)
    for tk in tasks:
        mask = np.array([r['_label']['task_key'] == tk for r in rows_for_head])
        if sum(mask) < 3 or len(set(y[mask])) < 2:
            continue
        probas_tk = probas[mask]; y_tk = y[mask]
        try:
            auroc_tk = roc_auc_score(y_tk, probas_tk)
            print('    %-20s N=%2d pos=%2d AUROC=%.3f' %
                  (tk, sum(mask), sum(y_tk), auroc_tk))
        except:
            pass

    return results


# ── Build heads ──
def build_head(rows, pos_cond):
    """pos_cond: lambda r -> True if positive"""
    pos = [r for r in rows if pos_cond(r['_label']) and r['_label'].get('unstable_or_edge', '0') == '0']
    neg = [r for r in rows if r['_label'].get('negative_clean', '0') == '1'
           and r['_label'].get('unstable_or_edge', '0') == '0']
    pool = pos + neg
    y = np.array([1 if pos_cond(r['_label']) else 0 for r in pool])
    return pool, y

# Head A: cmd_specific (exclude abstain)
pool_a, y_a = build_head(feature_rows,
                         lambda l: l.get('cmd_specific', '0') == '1' and l.get('abstain_any', '0') == '0')

# Head C: abstain_any
pool_c, y_c = build_head(feature_rows,
                         lambda l: l.get('abstain_any', '0') == '1')

# Head D: shared_qpos (diagnostic)
pool_d, y_d = build_head(feature_rows,
                         lambda l: l.get('shared_qpos_response', '0') == '1')

all_results = {}
if len(pool_a) >= 10 and sum(y_a) >= 2:
    all_results['A_cmd_specific'] = run_head('A: cmd_specific', pool_a, y_a)
else:
    print('\nHead A SKIP: pool=%d pos=%d' % (len(pool_a), sum(y_a)))

if len(pool_c) >= 10 and sum(y_c) >= 2:
    all_results['C_abstain_any'] = run_head('C: abstain_any', pool_c, y_c)
else:
    print('\nHead C SKIP: pool=%d pos=%d' % (len(pool_c), sum(y_c)))

if len(pool_d) >= 10 and sum(y_d) >= 2:
    all_results['D_shared_qpos'] = run_head('D: shared_qpos (diag)', pool_d, y_d)
else:
    print('\nHead D SKIP: pool=%d pos=%d' % (len(pool_d), sum(y_d)))

# ── Save ──
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['head', 'feature_group', 'n_features', 'auroc', 'auprc', 'p_at_5', 'p_at_10', 'enrich_5', 'enrich_10'])
    for hk, hr in all_results.items():
        for fg, res in hr.items():
            w.writerow([hk, fg, res['n_feat'], res['auroc'], res['auprc'], res['p5'], res['p10'], res['e5'], res['e10']])

print('\nOutput: %s' % args.out)
