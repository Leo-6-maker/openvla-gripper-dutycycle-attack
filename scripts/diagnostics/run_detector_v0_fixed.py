#!/usr/bin/env python3
"""Detector v0 — unified fixed framework (P0-A,B,C + P1-A,B,C repaired).

Usage:
  python run_detector_v0_fixed.py --label-tier bronze
  python run_detector_v0_fixed.py --label-tier silver_override
"""
import csv, os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict, Counter

ap = argparse.ArgumentParser()
ap.add_argument('--label-tier', choices=['bronze', 'silver_override', 'rescue_override'], default='silver_override')
ap.add_argument('--bronze-labels', default='/tmp/bronze_labels.csv')
ap.add_argument('--silver-labels', default='/tmp/silver_p1a_labels.csv')
ap.add_argument('--rescue-labels', default='/tmp/rescue_labels.csv')
ap.add_argument('--candidates', default='/data/liuyu/outputs/stageb_v1_1_reachable_window_candidates.csv')
ap.add_argument('--out', default='/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608')
args = ap.parse_args()

BRONZE_LABELS = args.bronze_labels
SILVER_LABELS = args.silver_labels
RESCUE_LABELS = args.rescue_labels
CANDIDATES = args.candidates
OUT = args.out
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
with open(BRONZE_LABELS) as f: bronze = {r['pair_id']: r for r in csv.DictReader(f)}
with open(CANDIDATES) as f: candidates = list(csv.DictReader(f))

# ── P0-A: candidate lookup with seed ──
cand_lookup = {}
missing_seed = 0
for c in candidates:
    seed = c.get('seed', 'MISSING')
    key = (c['task_key'], c['state_id'], seed, c['window_start'], c['window_end'])
    cand_lookup[key] = c
print('Candidate entries: %d' % len(cand_lookup))

# ── Label construction ──
label_rows = []
if args.label_tier == 'bronze':
    for pid, bl in bronze.items():
        seed = bl.get('seed', '0')
        key = (bl['task_key'], bl['state_id'], seed, bl['window_start'], bl['window_end'])
        # P0-B: hard fail if candidate not found
        c = cand_lookup.get(key)
        if c is None:
            print('HARD_FAIL_MISSING_CANDIDATE: %s' % str(key))
            sys.exit(1)
        cmd_any = int(bl['cmd_susceptible'])
        phys = int(bl['vis_specific_physical_response'])
        rand = int(bl['random_confounded'])
        cmd_specific = 1 if (cmd_any == 1 and rand == 0) else 0
        label_rows.append({'pair_id': pid, 'task_key': bl['task_key'], 'state_id': bl['state_id'],
            'seed': seed, 'window_start': int(bl['window_start']), 'window_end': int(bl['window_end']),
            'target_cmd_any': cmd_any, 'target_cmd_specific': cmd_specific,
            'target_phys': phys, 'target_rand': rand,
            'label_tier': 'bronze',
            'actual_max_step': int(c['actual_max_step']),
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

elif args.label_tier == 'silver_override':
    with open(SILVER_LABELS) as f: silver = list(csv.DictReader(f))
    # Group by parent
    sp = defaultdict(list)
    for r in silver:
        parent = r['pair_id'].replace('silver_bronze_', 'bronze_').rsplit('_r', 1)[0]
        sp[parent].append(r)
    # Stability
    silver_override = {}
    for parent, reps in sp.items():
        n = len(reps)
        vc = sum(1 for r in reps if int(r.get('vis_open_count', 0)) >= 6)
        vp = sum(1 for r in reps if float(r.get('vis_qpos_delta_shifted', 0)) >= 0.01)
        rc_r = sum(1 for r in reps if int(r.get('rand_open_count', 0)) >= 6)
        rp_r = sum(1 for r in reps if float(r.get('rand_qpos_delta_shifted', 0)) >= 0.01)
        vr = vc / max(n, 1); rr = rc_r / max(n, 1)
        pr_vis = vp / max(n, 1); pr_rand = rp_r / max(n, 1)
        is_cmd = vr >= 0.67 and rr <= 0.33
        is_phys = pr_vis >= 0.67 and pr_rand <= 0.33
        is_rand = rr >= 0.67 or pr_rand >= 0.67
        if is_cmd:
            silver_override[parent] = {'cmd_any': 1, 'phys': 1 if is_phys else 0, 'rand': 1 if is_rand else 0, 'tier': 'silver_cmd'}
        elif is_phys:
            silver_override[parent] = {'cmd_any': 0, 'phys': 1, 'rand': 1 if is_rand else 0, 'tier': 'silver_phys'}
        elif is_rand:
            silver_override[parent] = {'cmd_any': 0, 'phys': 0, 'rand': 1, 'tier': 'silver_rand'}
        elif vr <= 0.33 and rr <= 0.33:
            silver_override[parent] = {'cmd_any': 0, 'phys': 0, 'rand': 0, 'tier': 'silver_hard_neg'}
        else:
            silver_override[parent] = {'cmd_any': -1, 'phys': -1, 'rand': -1, 'tier': 'silver_unstable'}

    for pid, bl in bronze.items():
        ov = silver_override.get(pid, {})
        cmd_any = ov.get('cmd_any', int(bl['cmd_susceptible'])) if ov.get('cmd_any', -1) >= 0 else -1
        phys = ov.get('phys', int(bl['vis_specific_physical_response'])) if ov.get('phys', -1) >= 0 else -1
        rand = ov.get('rand', int(bl['random_confounded'])) if ov.get('rand', -1) >= 0 else -1
        if cmd_any < 0:
            continue  # unstable
        # P0-C: cmd_specific = cmd_any AND NOT random_sensitive
        cmd_specific = 1 if (cmd_any == 1 and rand == 0) else 0

        seed = bl.get('seed', '0')
        key = (bl['task_key'], bl['state_id'], seed, bl['window_start'], bl['window_end'])
        c = cand_lookup.get(key)
        if c is None:
            print('HARD_FAIL_MISSING_CANDIDATE: %s' % str(key))
            sys.exit(1)
        label_rows.append({'pair_id': pid, 'task_key': bl['task_key'], 'state_id': bl['state_id'],
            'seed': seed, 'window_start': int(bl['window_start']), 'window_end': int(bl['window_end']),
            'target_cmd_any': cmd_any, 'target_cmd_specific': cmd_specific,
            'target_phys': phys, 'target_rand': rand,
            'label_tier': ov.get('tier', 'bronze_only'),
            'actual_max_step': int(c['actual_max_step']),
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

print('Label rows: %d (tier=%s)' % (len(label_rows), args.label_tier))
for t in ['target_cmd_any', 'target_cmd_specific', 'target_phys', 'target_rand']:
    print('  %s: %d' % (t, sum(1 for r in label_rows if r[t] == 1)))
print('tiers:', dict(Counter(r['label_tier'] for r in label_rows)))

if len(label_rows) < 10:
    print('WARNING: < 10 rows — detector underpowered')
    sys.exit(0)

# ── P0-A: One-hot task + stratum ──
tasks = sorted(set(r['task_key'] for r in label_rows))
task_enc = OneHotEncoder(sparse_output=False)
task_oh = task_enc.fit_transform(np.array([tasks.index(r['task_key']) for r in label_rows]).reshape(-1, 1))
stratum_enc = OneHotEncoder(sparse_output=False)
stratum_oh = stratum_enc.fit_transform(np.array([[{'high_opportunity':0,'medium_opportunity':1,'hard_negative_or_idle':2}.get(r['stratum'],1)] for r in label_rows]))

X_clean = np.column_stack([
    [r['clean_open_count'] for r in label_rows],
    [r['clean_open_frac'] for r in label_rows],
    [r['raw_gripper_mean'] for r in label_rows],
    [r['raw_gripper_max'] for r in label_rows],
    [r['qpos_pre'] for r in label_rows],
    [r['qpos_mean'] for r in label_rows],
    [r['qpos_max'] for r in label_rows],
    [r['qpos_slope'] for r in label_rows],
    [r['eef_disp'] for r in label_rows],
])

# P1-A: proper rel_timing from actual_max_step
ws_arr = np.array([r['window_start'] for r in label_rows])
we_arr = np.array([r['window_end'] for r in label_rows])
wc_arr = (ws_arr + we_arr) / 2.0
max_step_arr = np.array([r['actual_max_step'] for r in label_rows])
rel_timing = wc_arr / np.maximum(max_step_arr, 1)

feature_groups = {
    'TaskOnly': task_oh,
    'StratumOnly': stratum_oh,
    'Task+Stratum': np.column_stack([task_oh, stratum_oh]),
    'CleanNoTaskNoTiming': X_clean,
    'CleanNoTaskWithTiming': np.column_stack([X_clean, wc_arr, rel_timing]),
    'AllWithTaskNoTiming': np.column_stack([task_oh, stratum_oh, X_clean]),
    'AllWithTaskWithTiming': np.column_stack([task_oh, stratum_oh, X_clean, wc_arr, rel_timing]),
}

# P0-A: Group by task_state_seed
groups = np.array(['%s_%s_%s' % (r['task_key'], r['state_id'], r['seed']) for r in label_rows])
n_splits = min(5, len(set(groups)))

targets = [
    ('cmd_any', np.array([r['target_cmd_any'] for r in label_rows])),
    ('cmd_specific', np.array([r['target_cmd_specific'] for r in label_rows])),
    ('vis_specific_physical', np.array([r['target_phys'] for r in label_rows])),
    ('random_sensitive', np.array([r['target_rand'] for r in label_rows])),
]

# ── Evaluate ──
all_results = []
for target_name, y in targets:
    pos = int(y.sum())
    prev = pos / max(len(y), 1)
    if pos < 3:
        all_results.append({'target': target_name, 'feature_group': 'ALL', 'model': 'ALL',
            'n_pos': pos, 'prevalence': round(prev, 3), 'P@5': 'underpowered'})
        continue

    for fg_name, X_fg in feature_groups.items():
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
                    'tier': args.label_tier, 'target': target_name, 'feature_group': fg_name, 'model': model_name,
                    'n_pos': pos, 'prevalence': round(prev, 3),
                    'P@3': round(p3, 3), 'P@5': round(p5, 3), 'P@10': round(p10, 3),
                    'enrich_P@5': round(p5 / max(prev, 0.01), 1),
                    'AUROC': round(roc_auc_score(y, y_prob), 3),
                    'AUPRC': round(average_precision_score(y, y_prob), 3),
                })
            except Exception as e:
                all_results.append({'tier': args.label_tier, 'target': target_name, 'feature_group': fg_name,
                    'model': model_name, 'n_pos': pos, 'prevalence': round(prev, 3), 'P@5': 'error'})

    # P1-B: Per-target shuffle baseline
    np.random.seed(42)
    y_shuf = y.copy(); np.random.shuffle(y_shuf)
    try:
        X_fg = feature_groups['AllWithTaskNoTiming']
        y_prob = np.zeros(len(y_shuf))
        gkf = GroupKFold(n_splits=n_splits)
        for ti, te in gkf.split(X_fg, y_shuf, groups):
            X_tr, X_te = X_fg[ti], X_fg[te]; y_tr = y_shuf[ti]
            s = StandardScaler(); X_tr_s = s.fit_transform(X_tr); X_te_s = s.transform(X_te)
            m = LogisticRegression(max_iter=1000); m.fit(X_tr_s, y_tr)
            y_prob[te] = m.predict_proba(X_te_s)[:, 1]
        p5s = y_shuf[np.argsort(-y_prob)[:min(5, len(y))]].sum() / min(5, len(y))
        all_results.append({'tier': args.label_tier, 'target': target_name, 'feature_group': 'ShuffleLabel',
            'model': 'LR', 'n_pos': pos, 'prevalence': round(prev, 3),
            'P@5': round(p5s, 3), 'enrich_P@5': round(p5s / max(prev, 0.01), 1),
            'AUROC': round(roc_auc_score(y_shuf, y_prob), 3)})
    except Exception as e:
        pass

# ── P1-C: Split audit with full target counts ──
split_rows = []
gkf = GroupKFold(n_splits=n_splits)
for fold_i, (ti, te) in enumerate(gkf.split(X_clean, targets[0][1], groups)):
    n_train = len(ti); n_test = len(te)
    for g in sorted(set(groups[te])):
        idx = np.where(groups[te] == g)[0]
        split_rows.append({'fold': fold_i, 'group_key': g, 'n_train': n_train, 'n_test': len(idx),
            'cmd_any_pos': int(targets[0][1][te][idx].sum()),
            'cmd_spec_pos': int(targets[1][1][te][idx].sum()),
            'phys_pos': int(targets[2][1][te][idx].sum()),
            'rand_pos': int(targets[3][1][te][idx].sum())})

# ── Write outputs ──
prefix = 'fixed_%s' % args.label_tier
with open(os.path.join(OUT, '%s_metrics.csv' % prefix), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tier','target','feature_group','model','n_pos','prevalence',
        'P@3','P@5','P@10','enrich_P@5','AUROC','AUPRC'])
    w.writeheader(); w.writerows(all_results)

with open(os.path.join(OUT, '%s_feature_table.csv' % prefix), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=label_rows[0].keys())
    w.writeheader(); w.writerows(label_rows)

with open(os.path.join(OUT, '%s_split_audit.csv' % prefix), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['fold','group_key','n_test','cmd_any_pos','cmd_spec_pos'])
    w.writeheader(); w.writerows(split_rows)

# ── Ablation summary ──
with open(os.path.join(OUT, '%s_feature_group_ablation.csv' % prefix), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tier','target','feature_group','model','n_pos','prevalence',
        'P@3','P@5','P@10','enrich_P@5','AUROC','AUPRC'])
    w.writeheader(); w.writerows(all_results)

print('\n=== ABLATION (%s) ===' % args.label_tier)
for t in ['cmd_any', 'cmd_specific', 'vis_specific_physical', 'random_sensitive']:
    tr = [r for r in all_results if r['target'] == t and r['model'] == 'RF']
    print('\n%s:' % t)
    for r in sorted(tr, key=lambda x: -float(x['P@5']) if isinstance(x['P@5'], float) else 0)[:5]:
        p5 = r['P@5']
        if isinstance(p5, float):
            print('  %-25s P@5=%.2f enrich=%.1fx AUROC=%.3f' % (r['feature_group'], p5, r.get('enrich_P@5',0), r.get('AUROC',0)))

print('\nSplit audit: %d groups' % len(set(s['group_key'] for s in split_rows)))
print('Done. Outputs in', OUT)
