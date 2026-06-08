#!/usr/bin/env python3
"""Detector v0 SilverOverride — with P0 fixes (seed in key, one-hot, ablations, grouped split)."""
import csv, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# ── P0-1: candidate lookup with seed ──
cand_lookup = {}
for c in candidates:
    # Check if seed field exists
    seed = c.get('seed', '0')
    key = (c['task_key'], c['state_id'], seed, c['window_start'], c['window_end'])
    cand_lookup[key] = c

# ── P0-2: Silver stability with matched random qpos exclusion ──
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
    rp_r = sum(1 for r in reps if float(r.get('rand_qpos_delta_shifted', 0)) >= 0.01)
    vr = vc / max(n, 1); rr = rc_r / max(n, 1)
    pr_vis = vp / max(n, 1); pr_rand = rp_r / max(n, 1)

    # P0-2: phys requires VIS qpos AND random qpos NOT meeting threshold
    is_cmd = vr >= 0.67 and rr <= 0.33
    is_phys = pr_vis >= 0.67 and pr_rand <= 0.33  # random qpos must NOT respond
    is_rand_cmd = rr >= 0.67
    is_rand_phys = pr_rand >= 0.67
    is_rand = is_rand_cmd or is_rand_phys  # random_sensitive = cmd OR phys response

    if is_cmd:
        silver_override[parent] = {'cmd': 1, 'phys': 1 if is_phys else 0, 'rand': 1 if is_rand else 0, 'tier': 'silver_cmd'}
    elif is_phys:
        silver_override[parent] = {'cmd': 0, 'phys': 1, 'rand': 1 if is_rand else 0, 'tier': 'silver_phys'}
    elif is_rand:
        silver_override[parent] = {'cmd': 0, 'phys': 0, 'rand': 1, 'tier': 'silver_rand'}
    elif vr <= 0.33 and rr <= 0.33:
        silver_override[parent] = {'cmd': 0, 'phys': 0, 'rand': 0, 'tier': 'silver_hard_neg'}
    else:
        silver_override[parent] = {'cmd': -1, 'phys': -1, 'rand': -1, 'tier': 'silver_unstable'}

# ── Build feature table ──
feature_rows = []
for pid, bl in bronze.items():
    seed = bl.get('seed', '0')
    key = (bl['task_key'], bl['state_id'], seed, bl['window_start'], bl['window_end'])
    c = cand_lookup.get(key, {})
    if not c:
        # Fallback: try without seed
        key_ns = (bl['task_key'], bl['state_id'], bl['window_start'], bl['window_end'])
        c = cand_lookup.get(key_ns, {})
        if not c:
            continue
    ov = silver_override.get(pid, {})
    cmd_label = ov.get('cmd', int(bl['cmd_susceptible'])) if ov.get('cmd', -1) >= 0 else -1
    phys_label = ov.get('phys', int(bl['vis_specific_physical_response'])) if ov.get('phys', -1) >= 0 else -1
    rand_label = ov.get('rand', int(bl['random_confounded'])) if ov.get('rand', -1) >= 0 else -1
    if cmd_label < 0:
        continue
    feature_rows.append({
        'pair_id': pid, 'task_key': bl['task_key'], 'state_id': bl['state_id'], 'seed': seed,
        'window_start': int(bl['window_start']), 'window_end': int(bl['window_end']),
        'target_cmd': cmd_label, 'target_phys': phys_label, 'target_rand': rand_label,
        'label_tier': ov.get('tier', 'bronze_only'),
        'clean_open_count': int(c.get('clean_open_count', 0)),
        'clean_open_frac': float(c.get('clean_open_frac', 0)),
        'raw_gripper_mean': float(c.get('raw_gripper_mean', 0)),
        'raw_gripper_max': float(c.get('raw_gripper_max', 0)),
        'qpos_pre': float(c.get('qpos_abs_sum_pre', 0)),
        'qpos_mean': float(c.get('qpos_abs_sum_window_mean', 0)),
        'qpos_max': float(c.get('qpos_abs_sum_window_max', 0)),
        'qpos_slope': float(c.get('qpos_abs_sum_slope', 0)),
        'eef_disp': float(c.get('eef_displacement', 0)),
        'stratum': c.get('candidate_stratum', '?'),
    })

print('Feature rows:', len(feature_rows))
for t in ['target_cmd', 'target_phys', 'target_rand']:
    print('  %s: %d' % (t, sum(1 for r in feature_rows if r[t] == 1)))
print('tiers:', dict(Counter(r['label_tier'] for r in feature_rows)))

# ── P0-3: Feature groups with one-hot task encoding ──
tasks = sorted(set(r['task_key'] for r in feature_rows))
task_encoder = OneHotEncoder(sparse_output=False)
task_onehot = task_encoder.fit_transform(np.array([tasks.index(r['task_key']) for r in feature_rows]).reshape(-1, 1))

stratum_map = {'high_opportunity': 0, 'medium_opportunity': 1, 'hard_negative_or_idle': 2}
stratum_onehot = OneHotEncoder(sparse_output=False).fit_transform(
    np.array([stratum_map.get(r['stratum'], 1) for r in feature_rows]).reshape(-1, 1))

ws_arr = np.array([r['window_start'] for r in feature_rows])
we_arr = np.array([r['window_end'] for r in feature_rows])
wc_arr = (ws_arr + we_arr) / 2.0
rel_timing = wc_arr / np.maximum(np.array([r['qpos_pre'] * 0 + 300 for r in feature_rows]), 1)  # use max_step from candidate?

X_clean = np.column_stack([
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

# Feature groups
feature_groups = {
    'TaskOnly': task_onehot,
    'StratumOnly': stratum_onehot,
    'Task+Stratum': np.column_stack([task_onehot, stratum_onehot]),
    'CleanNoTaskNoTiming': X_clean,
    'CleanNoTaskWithTiming': np.column_stack([X_clean, wc_arr, rel_timing]),
    'AllWithTaskNoTiming': np.column_stack([task_onehot, stratum_onehot, X_clean]),
    'AllWithTaskWithTiming': np.column_stack([task_onehot, stratum_onehot, X_clean, wc_arr, rel_timing]),
}

# ── P0-4: Group by task_state_seed ──
groups = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r.get('seed', '0')) for r in feature_rows])

# ── Evaluate ──
all_results = []
for target_name, y in [
    ('command_susceptible', np.array([r['target_cmd'] for r in feature_rows])),
    ('vis_specific_physical', np.array([r['target_phys'] for r in feature_rows])),
    ('random_sensitive', np.array([r['target_rand'] for r in feature_rows])),
]:
    pos = int(y.sum())
    prev = pos / max(len(y), 1)
    if pos < 3:
        all_results.append({'target': target_name, 'feature_group': 'ALL', 'model': 'ALL',
            'n_pos': pos, 'prevalence': round(prev, 3), 'P@5': 'underpowered'})
        continue

    for fg_name, X_fg in feature_groups.items():
        n_splits = min(5, len(set(groups)))
        for model_name, fn in [
            ('LR', lambda: LogisticRegression(max_iter=1000, class_weight='balanced')),
            ('RF', lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('GB', lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]:
            try:
                y_prob = np.zeros(len(y))
                gkf = GroupKFold(n_splits=n_splits)
                for ti, te in gkf.split(X_fg, y, groups):
                    X_tr, X_te = X_fg[ti], X_fg[te]; y_tr = y[ti]
                    s = StandardScaler(); X_tr_s = s.fit_transform(X_tr); X_te_s = s.transform(X_te)
                    m = fn(); m.fit(X_tr_s, y_tr)
                    y_prob[te] = m.predict_proba(X_te_s)[:, 1]
                order = np.argsort(-y_prob)
                p3 = y[order[:min(3, len(y))]].sum() / min(3, len(y))
                p5 = y[order[:min(5, len(y))]].sum() / min(5, len(y))
                p10 = y[order[:min(10, len(y))]].sum() / min(10, len(y))
                all_results.append({
                    'target': target_name, 'feature_group': fg_name, 'model': model_name,
                    'n_pos': pos, 'prevalence': round(prev, 3),
                    'P@3': round(p3, 3), 'P@5': round(p5, 3), 'P@10': round(p10, 3),
                    'enrich_P@5': round(p5 / max(prev, 0.01), 1),
                    'AUROC': round(roc_auc_score(y, y_prob), 3),
                    'AUPRC': round(average_precision_score(y, y_prob), 3),
                })
            except Exception as e:
                all_results.append({'target': target_name, 'feature_group': fg_name, 'model': model_name,
                    'n_pos': pos, 'prevalence': round(prev, 3), 'P@5': 'error'})

# ── Shuffle baseline ──
np.random.seed(42)
y_shuf = y.copy(); np.random.shuffle(y_shuf)
try:
    y_prob = np.zeros(len(y_shuf))
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for ti, te in gkf.split(X_fg, y_shuf, groups):
        X_tr, X_te = X_fg[ti], X_fg[te]; y_tr = y_shuf[ti]
        s = StandardScaler(); X_tr_s = s.fit_transform(X_tr); X_te_s = s.transform(X_te)
        m = LogisticRegression(max_iter=1000); m.fit(X_tr_s, y_tr)
        y_prob[te] = m.predict_proba(X_te_s)[:, 1]
    all_results.append({'target': target_name, 'feature_group': 'ShuffleLabel', 'model': 'LR',
        'n_pos': pos, 'prevalence': round(prev, 3),
        'P@5': round(y_shuf[np.argsort(-y_prob)[:min(5, len(y))]].sum() / min(5, len(y)), 3),
        'enrich_P@5': round((y_shuf[np.argsort(-y_prob)[:min(5, len(y))]].sum() / min(5, len(y))) / max(prev, 0.01), 1),
        'AUROC': round(roc_auc_score(y_shuf, y_prob), 3)})
except Exception as e:
    pass

# ── Write outputs ──
with open(os.path.join(OUT, 'silver_override_metrics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['target', 'feature_group', 'model', 'n_pos', 'prevalence',
        'P@3', 'P@5', 'P@10', 'enrich_P@5', 'AUROC', 'AUPRC'])
    w.writeheader(); w.writerows(all_results)

with open(os.path.join(OUT, 'silver_override_feature_table.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=feature_rows[0].keys())
    w.writeheader(); w.writerows(feature_rows)

# ── Feature group ablation summary ──
print('\n=== FEATURE GROUP ABLATION ===')
for target_name in ['command_susceptible', 'vis_specific_physical', 'random_sensitive']:
    tr = [r for r in all_results if r['target'] == target_name and r['model'] == 'RF']
    print('\n%s:' % target_name)
    for r in sorted(tr, key=lambda x: -float(x.get('P@5', 0)) if isinstance(x.get('P@5'), float) else 0):
        p5 = r.get('P@5', '?')
        if isinstance(p5, float):
            print('  %-25s P@5=%.2f enrich=%.1fx AUROC=%.3f' % (r['feature_group'], p5, r.get('enrich_P@5', 0), r.get('AUROC', 0)))

with open(os.path.join(OUT, 'feature_group_ablation.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['target', 'feature_group', 'model', 'n_pos', 'prevalence',
        'P@3', 'P@5', 'P@10', 'enrich_P@5', 'AUROC', 'AUPRC'])
    w.writeheader(); w.writerows(all_results)

# ── Split audit ──
split_audit = []
for gi, group_val in enumerate(sorted(set(groups))):
    idx = np.where(groups == group_val)[0]
    split_audit.append({'group_id': gi, 'group_key': group_val, 'n_samples': len(idx),
        'cmd_pos': int(y[idx].sum()) if len(idx) > 0 else 0})

with open(os.path.join(OUT, 'split_audit_fixed.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['group_id', 'group_key', 'n_samples', 'cmd_pos'])
    w.writeheader(); w.writerows(split_audit)

print('\nSplit audit: %d groups, %d samples' % (len(split_audit), sum(s['n_samples'] for s in split_audit)))
print('Done. Outputs in', OUT)
