#!/usr/bin/env python3
"""S5 Stable-Label Detector Sanity v0.1.

Input: K5+K5b stable parent pool (24 parents, K-repeat validated).
Heads: C (rand/abstain), A (cmd_specific), B (strict_phys, diagnostic only).
Features: TaskOnly, TimingOnly, CleanProprio, Clean+Timing, Task+Clean+Timing.
Evaluation: AUROC, AUPRC, P@3, leave-parent-out, leave-task-out, label shuffle.

No GPU needed. No 72-pair single-shot labels. No old visual features.
"""
import csv, numpy as np, os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter

LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
STABLE_POOL = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/combined_stable_pool_k5_k5b.csv'
OUT = '/data/liuyu/outputs/stageb_v1_1_k5b_targeted_stability_rc1a_0e3428f/detector_sanity_v0_1.csv'

# Map stable parents to 72-pair entries
PARENT_MAP = {
    'k5_cmd_anchor_milk_s0_w70_80_env0': ('milk', 0, 70, 80),
    'k5_confounded_both_milk_s0_w230_240_env0': ('milk', 0, 230, 240),
    'k5_rand_command_tomato_sauce_s2_w150_160_env2': ('tomato_sauce', 2, 150, 160),
    'k5_rand_phys_tomato_sauce_s2_w90_100_env2': ('tomato_sauce', 2, 90, 100),
    'k5_hn_surprise_bbq_sauce_s2_w100_110_env2': ('bbq_sauce', 2, 100, 110),
    'k5_neg_drift_salad_dressing_s2_w120_130_env2': ('salad_dressing', 2, 120, 130),
    'k5_clean_negative_expansion_alphabet_soup_s1_w65_75_env1': ('alphabet_soup', 1, 65, 75),
    'k5_strict_phys_master_tomato_sauce_s2_w115_125_env2': ('tomato_sauce', 2, 115, 125),
    'k5b_contrast_milk_late_milk_s0_w240_250_env0': ('milk', 0, 240, 250),
    'k5b_contrast_milk_early_milk_s0_w75_85_env0': ('milk', 0, 75, 85),
    'k5b_contrast_milk_mid_milk_s0_w80_90_env0': ('milk', 0, 80, 90),
    'k5b_contrast_tomato_late_tomato_sauce_s2_w155_165_env2': ('tomato_sauce', 2, 155, 165),
    'k5b_contrast_tomato_early_tomato_sauce_s2_w95_105_env2': ('tomato_sauce', 2, 95, 105),
    'k5b_contrast_tomato_far_tomato_sauce_s0_w55_65_env0': ('tomato_sauce', 0, 55, 65),
    'k5b_strict_phys_cream_cream_cheese_s2_w50_60_env2': ('cream_cheese', 2, 50, 60),
    'k5b_strict_phys_cream2_cream_cheese_s1_w145_155_env1': ('cream_cheese', 1, 145, 155),
    'k5b_strict_phys_tomato_tomato_sauce_s2_w165_175_env2': ('tomato_sauce', 2, 165, 175),
    'k5b_strict_phys_salad_salad_dressing_s2_w70_80_env2': ('salad_dressing', 2, 70, 80),
    'k5b_rand_alpha_alphabet_soup_s0_w60_70_env0': ('alphabet_soup', 0, 60, 70),
    'k5b_rand_salad_salad_dressing_s2_w80_90_env2': ('salad_dressing', 2, 80, 90),
    'k5b_rand_salad2_salad_dressing_s1_w50_60_env1': ('salad_dressing', 1, 50, 60),
    'k5b_neg_alpha_alphabet_soup_s1_w50_60_env1': ('alphabet_soup', 1, 50, 60),
    'k5b_neg_cream_cream_cheese_s0_w85_95_env0': ('cream_cheese', 0, 85, 95),
    'k5b_neg_bbq_bbq_sauce_s0_w60_70_env0': ('bbq_sauce', 0, 60, 70),
}

# Load 72-pair
labels_72 = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        labels_72[r['pair_id']] = r

# Load stable pool
stable = {}
with open(STABLE_POOL) as f:
    for r in csv.DictReader(f):
        stable[r['parent']] = r

# Build detector rows
rows = []
for pk, pool_r in stable.items():
    if pool_r['cmd_label'] == 'unstable_or_unknown':
        continue
    ts = PARENT_MAP.get(pk)
    if not ts: continue
    task, sid, ws, we = ts
    # Find 72-pair entry
    r72 = None
    for pid, r in labels_72.items():
        if task in pid and str(sid) in pid and str(ws) in pid and str(we) in pid:
            r72 = r; break
    if not r72: continue

    def f(field, d=0.0):
        try: return float(r72.get(field, d) or d)
        except: return d

    rows.append({
        'parent': pk, 'task_key': task, 'state_id': str(sid), 'seed': r72.get('seed','0'),
        'clean_open_count': f('clean_open_count'), 'clean_open_frac': f('clean_open_frac'),
        'raw_gripper_mean': f('raw_gripper_mean'), 'raw_gripper_max': f('raw_gripper_max'),
        'qpos_pre': f('qpos_pre'), 'qpos_mean': f('qpos_mean'),
        'window_start': ws, 'window_end': we,
        'actual_max_step': int(r72.get('actual_max_step', 299) or 299),
        'is_cmd': 1 if pool_r['cmd_label'] == 'stable_cmd_specific' else 0,
        'is_rand': 1 if pool_r['cmd_label'] == 'stable_rand_sensitive' else 0,
        'is_neg': 1 if pool_r['cmd_label'] == 'stable_negative' else 0,
        'is_phys': 1 if pool_r['phys_label'] == 'stable_vis_phys' else 0,
    })

print('Stable-label detector rows: %d' % len(rows))

# Feature matrices
X_clean = np.column_stack([
    [r['clean_open_count'] for r in rows], [r['clean_open_frac'] for r in rows],
    [r['raw_gripper_mean'] for r in rows], [r['raw_gripper_max'] for r in rows],
    [r['qpos_pre'] for r in rows], [r['qpos_mean'] for r in rows],
])
ws_arr = np.array([r['window_start'] for r in rows])
we_arr = np.array([r['window_end'] for r in rows])
wc_arr = (ws_arr + we_arr) / 2.0
max_arr = np.array([r['actual_max_step'] for r in rows])
rel_timing = wc_arr / np.maximum(max_arr, 1)

tasks = sorted(set(r['task_key'] for r in rows))
task_oh = np.array([[1 if tk == r['task_key'] else 0 for tk in tasks] for r in rows])

feature_groups = {
    'TaskOnly': task_oh,
    'TimingOnly': np.column_stack([wc_arr, rel_timing]),
    'CleanNoTaskNoTiming': X_clean,
    'CleanNoTaskWithTiming': np.column_stack([X_clean, wc_arr, rel_timing]),
    'Task+Clean+Timing': np.column_stack([X_clean, task_oh, wc_arr, rel_timing]),
}
groups = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r['seed']) for r in rows])

def run_head(name, y, rows_for_head):
    print('\n' + '='*70)
    print('HEAD %s (N=%d, pos=%d)' % (name, len(y), sum(y)))
    print('='*70)
    if sum(y) < 2:
        print('  SKIP: pos < 2')
        return {}

    # Rebuild features and groups for this pool
    X_clean_h = np.column_stack([
        [r['clean_open_count'] for r in rows_for_head], [r['clean_open_frac'] for r in rows_for_head],
        [r['raw_gripper_mean'] for r in rows_for_head], [r['raw_gripper_max'] for r in rows_for_head],
        [r['qpos_pre'] for r in rows_for_head], [r['qpos_mean'] for r in rows_for_head],
    ])
    ws_h = np.array([r['window_start'] for r in rows_for_head])
    we_h = np.array([r['window_end'] for r in rows_for_head])
    wc_h = (ws_h + we_h) / 2.0
    max_h = np.array([r['actual_max_step'] for r in rows_for_head])
    rel_h = wc_h / np.maximum(max_h, 1)
    tasks_h = sorted(set(r['task_key'] for r in rows_for_head))
    task_oh_h = np.array([[1 if tk == r['task_key'] else 0 for tk in tasks_h] for r in rows_for_head])
    fgs_h = {
        'TaskOnly': task_oh_h,
        'TimingOnly': np.column_stack([wc_h, rel_h]),
        'CleanNoTaskNoTiming': X_clean_h,
        'CleanNoTaskWithTiming': np.column_stack([X_clean_h, wc_h, rel_h]),
        'Task+Clean+Timing': np.column_stack([X_clean_h, task_oh_h, wc_h, rel_h]),
    }
    groups_h = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r['seed']) for r in rows_for_head])

    results = {}
    n_splits = min(3, len(set(groups_h)))

    for fg_name, X in fgs_h.items():
        # GroupKFold
        probas = np.zeros(len(y))
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(X, y, groups=groups_h):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            ss = StandardScaler()
            X_train_s = ss.fit_transform(X_train)
            X_test_s = ss.transform(X_test)
            m = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            m.fit(X_train_s, y_train)
            probas[test_idx] = m.predict_proba(X_test_s)[:, 1]

        auroc = roc_auc_score(y, probas)
        auprc = average_precision_score(y, probas)
        n_pos = sum(y); k3 = min(3, n_pos); k5 = min(5, n_pos)
        order = np.argsort(-probas)
        p3 = sum(y[i] for i in order[:k3]) / k3 if k3 > 0 else 0
        p5 = sum(y[i] for i in order[:k5]) / k5 if k5 > 0 else 0
        base = n_pos / len(y)
        e3 = p3 / base if base > 0 else 0; e5 = p5 / base if base > 0 else 0

        # Label shuffle
        np.random.seed(42); y_shuf = y.copy(); np.random.shuffle(y_shuf)
        probas_shuf = np.zeros(len(y_shuf))
        for train_idx, test_idx in gkf.split(X, y_shuf, groups=groups_h):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_s = y_shuf[train_idx]
            ss2 = StandardScaler()
            m2 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            m2.fit(ss2.fit_transform(X_train), y_train_s)
            probas_shuf[test_idx] = m2.predict_proba(ss2.transform(X_test))[:, 1]
        auroc_shuf = roc_auc_score(y_shuf, probas_shuf)

        results[fg_name] = {'auroc': round(auroc,3), 'auprc': round(auprc,3),
                            'p3': round(p3,2), 'p5': round(p5,2),
                            'e3': round(e3,1), 'e5': round(e5,1),
                            'auroc_shuf': round(auroc_shuf,3)}
        print('  %-28s AUROC=%.3f AUPRC=%.3f P@3=%.2f P@5=%.2f E@3=%.1fx E@5=%.1fx SHUF=%.3f' % (
            fg_name, auroc, auprc, p3, p5, e3, e5, auroc_shuf))

    # Per-task
    print('  Per-task AUROC:')
    for fg_name in ['TaskOnly', 'CleanNoTaskWithTiming', 'Task+Clean+Timing']:
        r = results.get(fg_name, {})
        if not r: continue
        print('    --- %s (%.3f) ---' % (fg_name, r['auroc']))
        for tk in tasks_h:
            mask = np.array([r2['task_key'] == tk for r2 in rows_for_head])
            if sum(mask) < 3 or len(set(y[mask])) < 2: continue
            try:
                auroc_tk = roc_auc_score(y[mask], probas[mask])
                print('      %-20s N=%2d pos=%2d AUROC=%.3f' % (tk, sum(mask), sum(y[mask]), auroc_tk))
            except: pass

    # Top-K task diversity
    print('  Top-5 task distribution (Clean+Timing):')
    r_ct = results.get('CleanNoTaskWithTiming', {})
    if r_ct:
        order5 = np.argsort(-probas)[:5]
        top_tasks = Counter(rows_for_head[i]['task_key'] for i in order5)
        for tk, n in top_tasks.most_common():
            print('    %-20s %d' % (tk, n))

    return results

# Head C: rand/abstain
y_rand = np.array([r['is_rand'] for r in rows])
pool_rand = [r for r in rows if r['is_rand'] or r['is_neg'] or r['is_cmd']]
y_rand_pool = np.array([1 if r['is_rand'] else 0 for r in pool_rand])
all_results = {}
all_results['C_rand_abstain'] = run_head('C: rand/abstain', y_rand_pool, pool_rand)

# Head A: cmd_specific (exclude rand)
pool_cmd = [r for r in rows if not r['is_rand']]
y_cmd_pool = np.array([1 if r['is_cmd'] else 0 for r in pool_cmd])
all_results['A_cmd_specific'] = run_head('A: cmd_specific', y_cmd_pool, pool_cmd)

# Head B: strict phys
pool_phys = [r for r in rows if not r['is_rand']]
y_phys_pool = np.array([1 if r['is_phys'] else 0 for r in pool_phys])
all_results['B_strict_phys'] = run_head('B: strict phys', y_phys_pool, pool_phys)

# Save
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['head','feature_group','auroc','auprc','p_at_3','p_at_5','enrich_3','enrich_5','auroc_shuffle'])
    for hk, hr in all_results.items():
        for fg, res in hr.items():
            w.writerow([hk, fg, res['auroc'], res['auprc'], res['p3'], res['p5'], res['e3'], res['e5'], res['auroc_shuf']])
print('\nOutput: %s' % OUT)
