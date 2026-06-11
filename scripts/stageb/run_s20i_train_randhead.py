#!/usr/bin/env python3
"""v0.3.1 randhead: train random_sensitive abstain head, aligned with v0.3 multi-head design.
Uses ALL RAND labels (S20F/S20G/S20H/S20I), clean-only features, GroupKFold by task_state."""
import csv, json, glob, os, numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

TABLES = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables'
CONFIGS = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/configs'
os.makedirs(TABLES, exist_ok=True)
os.makedirs(CONFIGS, exist_ok=True)

# ── Step 1: Collect ALL RAND labels ──
rand_dirs = [
    '/data/liuyu/outputs/stageb_s20f_queues_20260611/output',
    '/data/liuyu/outputs/stageb_s20f_v031_gpu10_extra_20260611',
    '/data/liuyu/outputs/stageb_s20g_v031_visfill_overnight_20260611',
    '/data/liuyu/outputs/stageb_s20h_positive_multiseed_20260612',
    '/data/liuyu/outputs/stageb_s20i_datamax_9h_20260612',
]

rand_rows = []
seen = set()
for d in rand_dirs:
    for f in glob.glob(d + '/summary_*random_linf*.json'):
        s = json.load(open(f))
        key = (s['task'], str(s['state_id']), s['window_start'], s['window_end'], str(s.get('attack_seed','0')))
        if key in seen: continue
        seen.add(key)
        r_open = s['decoded_open_count']; r_streak = s['max_open_streak']
        r_done = s['success_done_any']; r_timeout = s.get('timeout', False)
        r_steps = s['n_steps']

        if r_timeout or not r_done:
            label = 'RANDOM_SENSITIVE'
            target = 1
        elif r_open >= 6 or r_streak >= 6:
            label = 'RANDOM_SENSITIVE'
            target = 1
        elif r_open <= 3 and r_streak <= 3:
            label = 'RAND_STRICT_CLEAN'
            target = 0
        elif r_open <= 5 and r_streak <= 5:
            label = 'RAND_USABLE_CLEAN'
            target = 0
        else:
            label = 'BORDERLINE'
            target = -1

        rand_rows.append({
            'task': s['task'], 'state_id': str(s['state_id']),
            'window_start': s['window_start'], 'window_end': s['window_end'],
            'attack_seed': str(s.get('attack_seed','0')),
            'rand_open': r_open, 'rand_streak': r_streak,
            'rand_done': r_done, 'rand_timeout': r_timeout, 'rand_steps': r_steps,
            'rand_label': label, 'target_random_sensitive': target,
        })

label_counts = Counter(r['rand_label'] for r in rand_rows)
print('RAND master table: %d rows' % len(rand_rows))
print('Labels: RAND_STRICT=%d RAND_USABLE=%d RANDOM_SENSITIVE=%d BORDERLINE=%d' % (
    label_counts.get('RAND_STRICT_CLEAN',0), label_counts.get('RAND_USABLE_CLEAN',0),
    label_counts.get('RANDOM_SENSITIVE',0), label_counts.get('BORDERLINE',0)))

with open(TABLES + '/s20i_rand_label_master.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rand_rows[0].keys()))
    w.writeheader(); w.writerows(rand_rows)

# ── Step 2: Build clean-only features ──
# Load universe for phase + transition
universe = {}
with open('/data/liuyu/outputs/stageb_s20f_v031_repair_20260611/s20f_v031_candidate_universe.csv') as f:
    for r in csv.DictReader(f):
        universe[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']))] = r

# Load close-transition audit where available
trans_audit = {}
for fpath in glob.glob(TABLES + '/s20g_close_transition_audit.csv'):
    with open(fpath) as f:
        for r in csv.DictReader(f):
            trans_audit[(r['task'], r['state_id'], int(r['window_start']), int(r['window_end']), r['seed'])] = r

def build_features(task, sid, ws, we, seed):
    u = universe.get((task, sid, ws, we), {})
    t = trans_audit.get((task, sid, ws, we, str(seed)), {})
    fc = float(u.get('first_close_step', -1) or -1)
    lift = float(u.get('lift_step', -1) or -1)
    dl = float(u.get('done_step', 280) or 280)
    wc = (ws + we) / 2.0
    return {
        'task': task, 'phase': u.get('phase_id', '?'),
        'ws': ws, 'we': we, 'wc': wc, 'rel_timing': wc / max(dl, 1),
        'fc': fc if fc > 0 else -1, 'lift': lift if lift > 0 else -1,
        'ws_minus_fc': ws - fc if fc > 0 else 50,
        'ws_minus_lift': ws - lift if lift > 0 else 50,
        'clean_open_count': float(u.get('clean_open_count', 0)),
        'clean_open_frac': float(u.get('clean_open_frac', 0)),
        'post_grasp_open': float(u.get('post_grasp_open_count', 0)),
        'qpos_mean': float(u.get('qpos_mean', 0)),
        'qpos_slope': float(u.get('qpos_slope', 0)),
        'eef_disp': float(u.get('eef_disp', 0)),
        'distance_to_transition': float(t.get('distance_to_transition', 0) or 0),
        'pre_open_streak': float(t.get('pre_open_streak', 0) or 0),
        'post_close_streak': float(t.get('post_close_streak', 0) or 0),
        'transition_overlap': int(t.get('transition_overlap_center', 0) or 0),
        'close_commitment': float(t.get('close_commitment_score', 0.5) or 0.5),
    }

# Build feature vectors for trainable rows
trainable = [r for r in rand_rows if r['target_random_sensitive'] >= 0]
feat_rows = [build_features(r['task'], r['state_id'], r['window_start'], r['window_end'], r['attack_seed']) for r in trainable]
y = np.array([r['target_random_sensitive'] for r in trainable])
groups = np.array(['%s_%s' % (r['task'], r['state_id']) for r in trainable])
tasks = sorted(set(r['task'] for r in trainable))
phases = ['approach', 'grasp_transition', 'early_transport', 'transport', 'preplace', 'place_or_done']

n_pos = int(sum(y))
print('\nTraining set: %d (pos=%d, neg=%d), groups=%d' % (len(y), n_pos, len(y)-n_pos, len(set(groups))))

# ── Step 3: Train with all feature groups ──
feature_groups = {
    'TaskOnly': lambda f: [1 if f['task'] == tk else 0 for tk in tasks],
    'PhaseOnly': lambda f: [1 if f['phase'] == p else 0 for p in phases],
    'Task+Phase': lambda f: [1 if f['task'] == tk else 0 for tk in tasks] + [1 if f['phase'] == p else 0 for p in phases],
    'CleanNoTaskNoTiming': lambda f: [f['clean_open_count'], f['clean_open_frac'], f['post_grasp_open'], f['qpos_mean'], f['qpos_slope'], f['eef_disp']],
    'CleanNoTaskWithTiming': lambda f: [f['clean_open_count'], f['clean_open_frac'], f['post_grasp_open'], f['qpos_mean'], f['qpos_slope'], f['eef_disp'], f['rel_timing'], f['wc']],
    'TransitionOnly': lambda f: [f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'], f['transition_overlap'], f['close_commitment']],
    'Clean+Transition': lambda f: [f['clean_open_count'], f['clean_open_frac'], f['qpos_mean'], f['eef_disp'], f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'], f['transition_overlap'], f['close_commitment']],
    'Phase+Clean+Transition': lambda f: [f['ws_minus_fc'], f['ws_minus_lift'], f['rel_timing'], f['clean_open_count'], f['clean_open_frac'], f['qpos_mean'], f['eef_disp'], f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'], f['transition_overlap'], f['close_commitment']],
    'AllCleanNoTask': lambda f: [f['fc'], f['lift'], f['ws_minus_fc'], f['ws_minus_lift'], f['rel_timing'], f['clean_open_count'], f['clean_open_frac'], f['post_grasp_open'], f['qpos_mean'], f['qpos_slope'], f['eef_disp'], f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'], f['transition_overlap'], f['close_commitment']],
    'AllCleanWithTask': lambda f: [f['fc'], f['lift'], f['ws_minus_fc'], f['ws_minus_lift'], f['rel_timing'], f['clean_open_count'], f['clean_open_frac'], f['post_grasp_open'], f['qpos_mean'], f['qpos_slope'], f['eef_disp'], f['distance_to_transition'], f['pre_open_streak'], f['post_close_streak'], f['transition_overlap'], f['close_commitment']] + [1 if f['task'] == tk else 0 for tk in tasks],
}

results = []
for fg_name, fg_fn in feature_groups.items():
    X = np.array([fg_fn(f) for f in feat_rows])
    n_splits = max(2, min(5, len(set(groups))))
    gkf = GroupKFold(n_splits=n_splits)

    for model_name, ModelCls, needs_scale in [
        ('LR', LogisticRegression, True),
        ('RF', RandomForestClassifier, False),
        ('GB', GradientBoostingClassifier, False),
    ]:
        auroc_scores = []; auprc_scores = []; prec_scores = []; rec_scores = []
        oof_preds = np.zeros(len(y))

        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr])) < 2: continue
            if needs_scale:
                ss = StandardScaler(); Xtr = ss.fit_transform(X[tr]); Xte = ss.transform(X[te])
            else:
                Xtr = X[tr]; Xte = X[te]

            kw = {'max_iter': 2000, 'class_weight': 'balanced', 'random_state': 42} if model_name == 'LR' else \
                 {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42} if model_name == 'RF' else \
                 {'n_estimators': 100, 'random_state': 42}
            m = ModelCls(**kw)
            m.fit(Xtr, y[tr])
            yp = m.predict_proba(Xte)[:, 1]
            oof_preds[te] = yp
            auroc_scores.append(roc_auc_score(y[te], yp))
            auprc_scores.append(average_precision_score(y[te], yp))
            yh = (yp >= 0.5).astype(int)
            if sum(yh) > 0:
                prec_scores.append(precision_score(y[te], yh, zero_division=0))
                rec_scores.append(recall_score(y[te], yh, zero_division=0))

        eligible_mask = oof_preds <= 0.40
        eligible_prec = 1 - np.mean(y[eligible_mask]) if sum(eligible_mask) > 0 else 0
        abstain_rate = 1 - sum(eligible_mask) / len(y)

        results.append({
            'feature_group': fg_name, 'model': model_name,
            'n': len(y), 'pos': n_pos,
            'auroc': round(np.mean(auroc_scores), 3) if auroc_scores else '',
            'auprc': round(np.mean(auprc_scores), 3) if auprc_scores else '',
            'recall': round(np.mean(rec_scores), 3) if rec_scores else '',
            'eligible_precision': round(eligible_prec, 3),
            'false_clean_rate': round(1 - eligible_prec, 3),
            'abstain_rate': round(abstain_rate, 3),
        })

# Write metrics
with open(TABLES + '/s20i_v031_randhead_metrics.csv', 'w', newline='') as f:
    fields = ['feature_group','model','n','pos','auroc','auprc','recall','eligible_precision','false_clean_rate','abstain_rate']
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(results)

print('\n=== RANDHEAD METRICS ===')
print('%-28s %-3s %5s %4s %7s %7s %7s %7s %7s %7s' % ('Feature Group','Mdl','N','Pos','AUROC','AUPRC','Recall','EligPrec','FalseCln','Abstain'))
print('-' * 100)
for r in results:
    print('%-28s %-3s %5d %4d %7s %7s %7s %7s %7s %7s' % (
        r['feature_group'], r['model'], r['n'], r['pos'],
        r['auroc'], r['auprc'], r['recall'], r['eligible_precision'], r['false_clean_rate'], r['abstain_rate']))

# Find best
best = max([r for r in results if isinstance(r['auroc'], float)], key=lambda r: r['eligible_precision'])
print('\nBest: %s + %s (EligPrec=%.3f, AUROC=%.3f, Abstain=%.3f)' % (
    best['feature_group'], best['model'], best['eligible_precision'], best['auroc'], best['abstain_rate']))

# Save config
with open(CONFIGS + '/stageb_detector_v031_randhead.yaml', 'w') as f:
    json.dump({'detector_version': 'v0.3.1_randhead', 'best_group': best['feature_group'],
               'best_model': best['model'], 'eligible_threshold': 0.40, 'abstain_threshold': 0.50,
               'eligible_strict': 0.25, 'n_train': int(best['n']), 'n_pos': int(best['pos']),
               'auroc': best['auroc'], 'eligible_precision': best['eligible_precision']}, f, indent=2)

print('\nOutputs: %s, %s' % (TABLES, CONFIGS))
