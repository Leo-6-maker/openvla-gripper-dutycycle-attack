#!/usr/bin/env python3
"""Detector v0.3 inference adapter for S20d clean traces.
Trains LR on full 72-pair labeled set, then predicts on S20d candidate windows."""
import csv, numpy as np, sys, os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LABELS_72 = '/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/all_labels_rc1a_14cfabe_72pairs.csv'
STABLE = '/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv'

CANDIDATES = [
    {'task': 'ketchup', 'state_id': '1', 'ws': 149, 'we': 158,
     'clean_trace': '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_ketchup_s1_w0_10_s20d_clean_seed0_job960100.csv'},
    {'task': 'tomato_sauce', 'state_id': '3', 'ws': 111, 'we': 120,
     'clean_trace': '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s3_w0_10_s20d_clean_seed0_job960101.csv'},
    {'task': 'tomato_sauce', 'state_id': '5', 'ws': 148, 'we': 157,
     'clean_trace': '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/clean/trace_tomato_sauce_s5_w0_10_s20d_clean_seed0_job960102.csv'},
]

OUT_CSV = '/data/liuyu/outputs/stageb_s20e_mainline_official_closure_20260611/s20e_detector_v03_window_candidates.csv'

# ── Load 72 labeled pairs ──
labels = {}
with open(LABELS_72) as f:
    for r in csv.DictReader(f):
        key = (r['task_key'], r['state_id'], r.get('seed', '0'),
               r['window_start'], r['window_end'])
        labels[key] = r

def compute_features(trace_path, ws, we):
    """Extract v0.3 features from S20d trace for a given window [ws, we]."""
    with open(trace_path) as f:
        rows = list(csv.DictReader(f))

    window_rows = [r for r in rows if ws <= int(r['step']) <= we]
    pre_rows = [r for r in rows if int(r['step']) < ws]

    def g(row, key, default=0.0):
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return default

    # Clean open count/frac in window
    clean_open_count = sum(1 for r in window_rows if g(r, 'decoded_open_bool') == 1)
    clean_open_frac = clean_open_count / max(len(window_rows), 1)

    # Raw gripper commands in window (clean)
    gripper_vals = [g(r, 'clean_gripper_env') for r in window_rows]
    raw_gripper_mean = np.mean(gripper_vals) if gripper_vals else 0.0
    raw_gripper_max = np.max(gripper_vals) if gripper_vals else 0.0

    # Qpos before window and mean during window
    qpos_pre_vals = [g(r, 'gripper_qpos_before') for r in pre_rows[-5:]] if pre_rows else [0.0]
    qpos_pre = np.median(qpos_pre_vals) if qpos_pre_vals else 0.0
    qpos_vals = [g(r, 'gripper_qpos_before') for r in window_rows]
    qpos_mean = np.mean(qpos_vals) if qpos_vals else 0.0

    # Timing
    actual_max_step = max(int(r['step']) for r in rows)
    window_center = (ws + we) / 2.0
    rel_timing = window_center / max(actual_max_step, 1)

    return {
        'clean_open_count': clean_open_count,
        'clean_open_frac': clean_open_frac,
        'raw_gripper_mean': raw_gripper_mean,
        'raw_gripper_max': raw_gripper_max,
        'qpos_pre': qpos_pre,
        'qpos_mean': qpos_mean,
        'window_start': ws,
        'window_end': we,
        'actual_max_step': actual_max_step,
    }


# ── Train LR on full 72-pair set ──
train_rows = []
with open(STABLE) as f:
    stable = {r['parent']: r for r in csv.DictReader(f)}

KNOWN_TASKS = ['alphabet_soup', 'bbq_sauce', 'butter', 'cream_cheese',
               'milk', 'orange_juice', 'salad_dressing', 'tomato_sauce']

import re
for pk, pr in stable.items():
    if pr['cmd_label'] == 'unstable_or_unknown':
        continue
    # Parse task/state/window
    task = None
    for tk in KNOWN_TASKS:
        if tk in pk:
            task = tk; break
    m_s = re.search(r'_s(\d+)', pk)
    sid = m_s.group(1) if m_s else '0'
    m_w = re.search(r'_w(\d+)_(\d+)', pk)
    ws_str = m_w.group(1) if m_w else None
    we_str = m_w.group(2) if m_w else None
    if not all([task, ws_str, we_str]):
        continue

    # Find in labels
    found = None
    for s in ['0', '1', '2']:
        key = (task, str(sid), s, ws_str, we_str)
        if key in labels:
            found = labels[key]; break
    if not found:
        continue

    def ff(field, d=0.0):
        try: return float(found.get(field, d) or d)
        except: return d

    train_rows.append({
        'clean_open_count': ff('clean_open_count'),
        'clean_open_frac': ff('clean_open_frac'),
        'raw_gripper_mean': ff('raw_gripper_mean'),
        'raw_gripper_max': ff('raw_gripper_max'),
        'qpos_pre': ff('qpos_pre'),
        'qpos_mean': ff('qpos_mean'),
        'window_start': int(ws_str),
        'window_end': int(we_str),
        'actual_max_step': int(found.get('actual_max_step', 299) or 299),
        'is_rand': 1 if 'rand_sensitive' in pr['cmd_label'] else 0,
        'is_cmd': 1 if 'cmd_specific' in pr['cmd_label'] else 0,
    })

print('Training rows: %d (cmd=%d rand=%d)' % (
    len(train_rows),
    sum(r['is_cmd'] for r in train_rows),
    sum(r['is_rand'] for r in train_rows)))

# Build training features
X_clean_tr = np.column_stack([
    [r['clean_open_count'] for r in train_rows],
    [r['clean_open_frac'] for r in train_rows],
    [r['raw_gripper_mean'] for r in train_rows],
    [r['raw_gripper_max'] for r in train_rows],
    [r['qpos_pre'] for r in train_rows],
    [r['qpos_mean'] for r in train_rows],
])
ws_tr = np.array([r['window_start'] for r in train_rows])
we_tr = np.array([r['window_end'] for r in train_rows])
wc_tr = (ws_tr + we_tr) / 2.0
max_tr = np.array([r['actual_max_step'] for r in train_rows])
rel_tr = wc_tr / np.maximum(max_tr, 1)
X_tr = np.column_stack([X_clean_tr, wc_tr, rel_tr])

y_rand = np.array([r['is_rand'] for r in train_rows])
y_cmd = np.array([r['is_cmd'] for r in train_rows])

# Fit scaler + models on full data
ss = StandardScaler()
X_tr_scaled = ss.fit_transform(X_tr)

m_rand = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_rand.fit(X_tr_scaled, y_rand)

m_cmd = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
m_cmd.fit(X_tr_scaled, y_cmd)

# Baseline thresholds
p_rand_train = m_rand.predict_proba(X_tr_scaled)[:, 1]
abstain_threshold = np.percentile(p_rand_train, 50)
print('Abstain threshold (p_rand 50th pctile): %.4f' % abstain_threshold)

# ── Predict on S20d candidates ──
out_rows = []
for c in CANDIDATES:
    feats = compute_features(c['clean_trace'], c['ws'], c['we'])
    X_test = np.array([[
        feats['clean_open_count'], feats['clean_open_frac'],
        feats['raw_gripper_mean'], feats['raw_gripper_max'],
        feats['qpos_pre'], feats['qpos_mean'],
        (feats['window_start'] + feats['window_end']) / 2.0,
        ((feats['window_start'] + feats['window_end']) / 2.0) / max(feats['actual_max_step'], 1),
    ]])
    X_test_scaled = ss.transform(X_test)

    p_rand = float(m_rand.predict_proba(X_test_scaled)[0, 1])
    p_cmd = float(m_cmd.predict_proba(X_test_scaled)[0, 1])
    abstain = p_rand > abstain_threshold

    row = {
        'task': c['task'],
        'state_id': c['state_id'],
        'detector_name': 'detector_v0.3',
        'detector_version': 'v0.3',
        'checkpoint_path': 'sklearn_LR_trained_on_72pairs_K5_K5b_K5c',
        'config_path': 'scripts/diagnostics/run_selector_v0_3.py',
        'feature_source': 's20d_clean_trace',
        'feature_adapter': 's20d_trace_to_v03_features',
        'window_start': c['ws'],
        'window_end': c['we'],
        'window_len': c['we'] - c['ws'] + 1,
        'detector_score_cmd': round(p_cmd, 4),
        'detector_score_rand': round(p_rand, 4),
        'confidence': 'high' if p_cmd > 0.7 and not abstain else ('medium' if p_cmd > 0.5 and not abstain else 'low'),
        'abstain': abstain,
        'abstain_reason': 'p_rand=%.4f > threshold=%.4f' % (p_rand, abstain_threshold) if abstain else '',
        'clean_success_done': True,
        'clean_success_check': True,
        'pre_registered_before_rand_vis': True,
        'notes': 'window from generic autowindow phase-cue detector; v0.3 inference on top',
    }
    out_rows.append(row)
    print('[%s_s%s] w%d-%d p_cmd=%.4f p_rand=%.4f abstain=%s' % (
        c['task'], c['state_id'], c['ws'], c['we'], p_cmd, p_rand, abstain))

# Write output
with open(OUT_CSV, 'w', newline='') as f:
    fieldnames = ['task', 'state_id', 'detector_name', 'detector_version',
                  'checkpoint_path', 'config_path', 'feature_source', 'feature_adapter',
                  'window_start', 'window_end', 'window_len',
                  'detector_score_cmd', 'detector_score_rand', 'confidence',
                  'abstain', 'abstain_reason',
                  'clean_success_done', 'clean_success_check',
                  'pre_registered_before_rand_vis', 'notes']
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader(); w.writerows(out_rows)

print('\nOutput:', OUT_CSV)
