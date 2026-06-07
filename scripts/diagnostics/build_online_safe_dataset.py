#!/usr/bin/env python3
"""Build online-safe vulnerability dataset from clean-rollout features ONLY.
No VIS outcomes. No oracle labels as features. No offline columns.

Feature sources:
  1. ProprioNoStep per-step features (3 tasks: cream_cheese, ketchup, salad_dressing)
  2. Clean trace CSVs (batch3b + milestone_7 runs)
  3. Phase features (detector_phase_features_v1.csv — all 9 tasks)
  4. Window position / temporal features

Output:
  tables/online_safe_vulnerability_dataset.csv
  reports/ONLINE_SAFE_VULNERABILITY_DATASET_AUDIT.md
"""

import csv, os, sys, glob, re
from collections import defaultdict
from datetime import datetime
import numpy as np

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
CODEX = '/data/liuyu/outputs/codex_phase_detector_twostage_20260606/tables'
PROPRIO = '/data/liuyu/outputs/proprionostep_cpu_20260602'
BATCH3B = '/data/liuyu/outputs/nightly_object_batch3b_20260604'
MILESTONE7 = '/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs'
OUT_CSV = os.path.join(REPO, 'tables/online_safe_vulnerability_dataset.csv')
OUT_MD = os.path.join(REPO, 'reports/ONLINE_SAFE_VULNERABILITY_DATASET_AUDIT.md')

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

def make_key(r, tk='task_key', sid='state_id', ws='window_start', we='window_end'):
    return (str(r.get(tk,'')).strip(), str(r.get(sid,'')).strip(),
            str(r.get(ws,'')).strip(), str(r.get(we,'')).strip())

def safe_float(v, default=np.nan):
    try: return float(v)
    except: return default

def compute_window_stats(step_data, ws, we):
    """Compute statistics for a window over step-level data.
    step_data: list of dicts with step-level features.
    Returns dict of {feature_name: {mean, std, min, max, delta, first, last}}"""
    window = [d for d in step_data if ws <= int(d.get('step', -1)) <= we]
    if len(window) < 2:
        return None

    numeric_cols = []
    for k in window[0].keys():
        if k in ('task', 'step', 'task_key'):
            continue
        try:
            float(window[0][k])
            numeric_cols.append(k)
        except:
            pass

    stats = {}
    for col in numeric_cols:
        vals = [float(d[col]) for d in window]
        vals_arr = np.array(vals)
        stats[col + '_mean'] = float(np.mean(vals_arr))
        stats[col + '_std'] = float(np.std(vals_arr))
        stats[col + '_min'] = float(np.min(vals_arr))
        stats[col + '_max'] = float(np.max(vals_arr))
        stats[col + '_delta'] = float(vals[-1] - vals[0]) if len(vals) > 1 else 0.0
        stats[col + '_first'] = float(vals[0])
        stats[col + '_last'] = float(vals[-1])
    return stats


# ── Load labels ───────────────────────────────────────────────────
labels_v2 = read_csv(os.path.join(SHARED, 'object_phase_response_labels_v2.csv'))
labels_by_key = {make_key(r): r for r in labels_v2}
print('Labels v2: %d rows' % len(labels_v2))

# ── Load phase features (all 9 tasks) ─────────────────────────────
phase_features = read_csv(os.path.join(CODEX, 'detector_phase_features_v1.csv'))
phase_by_key = {make_key(r): r for r in phase_features}
print('Phase features: %d rows covering %d keys' % (len(phase_features), len(phase_by_key)))

# ── Load ProprioNoStep per-step features (3 tasks) ─────────────────
proprio_data = {}
for task in ['cream_cheese', 'ketchup', 'salad_dressing']:
    fpath = os.path.join(PROPRIO, 'features_%s.csv' % task)
    if os.path.exists(fpath):
        data = read_csv(fpath)
        # Index by step
        proprio_data[task] = {}
        for d in data:
            step = int(d.get('step', -1))
            if step >= 0:
                proprio_data[task][step] = d
        print('ProprioNoStep %s: %d steps' % (task, len(proprio_data[task])))

# ── Load checkpoint scores (hazard/release per step-window) ───────
checkpoint_scores = read_csv(os.path.join(PROPRIO, 'checkpoint_scores_clean_traces.csv'))
print('Checkpoint scores: %d step-level entries' % len(checkpoint_scores))

# Aggregate checkpoint scores to window-level per (task, ws, we)
checkpoint_by_task_window = defaultdict(lambda: {'hazard': [], 'release': []})
for r in checkpoint_scores:
    tk = str(r.get('task','')).strip()
    ws = str(r.get('window_start','')).strip()
    we = str(r.get('window_end','')).strip()
    hs = safe_float(r.get('hazard_score'))
    rs = safe_float(r.get('release_safe_score'))
    if not np.isnan(hs):
        checkpoint_by_task_window[(tk, ws, we)]['hazard'].append(hs)
    if not np.isnan(rs):
        checkpoint_by_task_window[(tk, ws, we)]['release'].append(rs)

# ── Find clean traces for each label row ──────────────────────────
def find_clean_trace(task, state_id, ws, we):
    """Find a clean trace CSV for the given task/state/window."""
    patterns = [
        # batch3b pattern
        os.path.join(BATCH3B, '%s_s%s' % (task, state_id), 'traces',
                     '%s_s%s_clean_w%s_%s_trace.csv' % (task, state_id, ws, we)),
        # milestone_7 pattern
        os.path.join(MILESTONE7, 'vis_%s_state%s_clean_*_w%s_%s_*_trace.csv' % (task, state_id, ws, we)),
        # batch3b alt pattern
        os.path.join(BATCH3B, '%s_s%s' % (task, state_id), 'traces',
                     'clean*trace*.csv'),
    ]

    # Try exact match first
    exact = patterns[0]
    if os.path.exists(exact):
        return exact

    # Try glob
    for pat in [patterns[1]]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]

    return None


# ── Build online-safe dataset ─────────────────────────────────────
print('\n=== Building online-safe dataset ===')

# Define ALL output columns with tags
FEATURE_COLS = []

# Identity columns
ID_COLS = ['task_key', 'state_id', 'window_start', 'window_end']
for c in ID_COLS:
    FEATURE_COLS.append((c, 'online_identity'))

# Window position features
POS_COLS = ['window_position_norm', 'window_size']
for c in POS_COLS:
    FEATURE_COLS.append((c, 'online_feature'))

# Phase features (ProprioNoStep + heuristic, available online)
PHASE_COLS = [
    'phase_bin_proxy', 'predicted_phase', 'phase_confidence',
    'phase_is_critical', 'qpos_phase_class',
    'hazard_score_mean', 'hazard_score_max',
    'release_safe_score_mean', 'release_safe_score_min',
]
for c in PHASE_COLS:
    FEATURE_COLS.append((c, 'online_feature'))

# Clean proprio/action statistics (from ProprioNoStep features or clean traces)
PROPRIO_STATS = [
    'gripper_command_mean', 'gripper_command_std', 'gripper_command_delta',
    'gripper_qpos_mean', 'gripper_qpos_std', 'gripper_qpos_delta',
    'gripper_width_mean', 'gripper_width_std', 'gripper_width_delta',
    'eef_x_delta', 'eef_y_delta', 'eef_z_delta',
    'eef_vx_mean', 'eef_vx_std', 'eef_vy_mean', 'eef_vy_std', 'eef_vz_mean', 'eef_vz_std',
    'action_dx_mean', 'action_dx_std', 'action_dy_mean', 'action_dy_std',
    'action_dz_mean', 'action_dz_std',
    'action_gripper_mean', 'action_gripper_std', 'action_gripper_delta',
]
for c in PROPRIO_STATS:
    FEATURE_COLS.append((c, 'online_feature'))

# Label columns (NOT features — for training target only)
LABEL_COLS = [
    'label_status', 'label_vulnerability_ready', 'label_source',
    'train_use', 'taxonomy', 'mechanism_type',
]
for c in LABEL_COLS:
    FEATURE_COLS.append((c, 'label_only'))

# Audit columns
AUDIT_COLS = [
    'clean_trace_source', 'phase_source', 'proprio_available',
    'feature_completeness', 'exclusion_or_uncertain_reason',
]
for c in AUDIT_COLS:
    FEATURE_COLS.append((c, 'offline_audit_only'))

column_tags = {c: tag for c, tag in FEATURE_COLS}
all_cols = [c for c, _ in FEATURE_COLS]

dataset_rows = []

for r in labels_v2:
    key = make_key(r)
    task = r['task_key'].strip()
    sid = r['state_id'].strip()
    ws = int(r['window_start'])
    we = int(r['window_end'])

    row = {c: '' for c in all_cols}

    # Identity
    row['task_key'] = task
    row['state_id'] = sid
    row['window_start'] = str(ws)
    row['window_end'] = str(we)

    # Window position features
    row['window_position_norm'] = str(round(ws / 100.0, 3))  # rough normalization
    row['window_size'] = str(we - ws)

    # Phase features (from detector_phase_features_v1.csv)
    ph = phase_by_key.get(key, {})
    if ph:
        row['phase_bin_proxy'] = ph.get('phase_bin_proxy', '')
        row['predicted_phase'] = ph.get('predicted_phase', '')
        row['phase_confidence'] = ph.get('phase_confidence', '')
        row['phase_is_critical'] = ph.get('phase_is_critical', '')
        row['qpos_phase_class'] = ph.get('qpos_phase_class', '')
        row['phase_source'] = 'detector_phase_features_v1'
    else:
        row['phase_source'] = 'missing'

    # Hazard/release scores from checkpoint aggregation
    ck = checkpoint_by_task_window.get((task, str(ws), str(we)), None)
    if ck and ck['hazard']:
        row['hazard_score_mean'] = str(round(np.mean(ck['hazard']), 10))
        row['hazard_score_max'] = str(round(np.max(ck['hazard']), 10))
    if ck and ck['release']:
        row['release_safe_score_mean'] = str(round(np.mean(ck['release']), 10))
        row['release_safe_score_min'] = str(round(np.min(ck['release']), 10))

    # ProprioNoStep per-step features (3 tasks only)
    proprio_available = False
    if task in proprio_data:
        step_data_list = []
        for step, d in proprio_data[task].items():
            step_data_list.append(d)
        if step_data_list:
            stats = compute_window_stats(step_data_list, ws, we)
            if stats:
                proprio_available = True
                for k, v in stats.items():
                    col_name = k
                    if col_name in row:
                        row[col_name] = str(round(v, 6))
    row['proprio_available'] = 'YES' if proprio_available else 'no'

    # Try to find and read clean trace CSV for richer proprio stats
    clean_trace_path = find_clean_trace(task, sid, ws, we)
    if clean_trace_path and not proprio_available:
        # Read clean trace and compute stats
        try:
            trace_data = []
            with open(clean_trace_path) as f:
                reader = csv.DictReader(f)
                for d in reader:
                    # Try to get step from the data
                    step_val = None
                    for k in ['step', 'Step', 'STEP', 'frame', 'Frame']:
                        if k in d:
                            try:
                                step_val = int(float(d[k]))
                            except:
                                pass
                            break
                    if step_val is not None:
                        trace_data.append({'step': step_val, **d})
            if trace_data:
                stats = compute_window_stats(trace_data, ws, we)
                if stats:
                    for k, v in stats.items():
                        col_name = k
                        if col_name in row:
                            row[col_name] = str(round(v, 6))
                    proprio_available = True
        except Exception as e:
            pass

    row['clean_trace_source'] = clean_trace_path or 'not_found'

    # Feature completeness
    online_feature_cols = [c for c, tag in column_tags.items() if tag == 'online_feature']
    filled = sum(1 for c in online_feature_cols if row.get(c) and row[c] != '')
    row['feature_completeness'] = '%d/%d' % (filled, len(online_feature_cols))

    # Copy labels (target, NOT features)
    row['label_status'] = r.get('label_status', '')
    row['label_vulnerability_ready'] = r.get('label_vulnerability_ready', '')
    row['label_source'] = r.get('label_source', 'gold_v2')
    row['train_use'] = r.get('label_use', '')
    row['taxonomy'] = r.get('taxonomy', '')

    # Mechanism type from audit
    mech_tax = read_csv(os.path.join(REPO, 'tables/vulnerability_mechanism_taxonomy_audit.csv'))
    mech_by_key = {make_key(r): r for r in mech_tax}
    mt = mech_by_key.get(key, {})
    row['mechanism_type'] = mt.get('mechanism_type', '')

    # Exclusion
    if r.get('label_use') == 'ignore':
        row['exclusion_or_uncertain_reason'] = 'label_use=ignore: %s' % r.get('taxonomy', '')
    elif not proprio_available and task not in proprio_data:
        row['exclusion_or_uncertain_reason'] = 'no ProprioNoStep features available'
    else:
        row['exclusion_or_uncertain_reason'] = ''

    dataset_rows.append(row)

# ── Write CSV ─────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_cols)
    w.writeheader()
    w.writerows(dataset_rows)
print('Wrote %d data rows to %s' % (len(dataset_rows), OUT_CSV))

# Write column tag metadata separately
tag_csv = OUT_CSV.replace('.csv', '_column_tags.csv')
with open(tag_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['column_name', 'tag'])
    w.writeheader()
    for col, tag in FEATURE_COLS:
        w.writerow({'column_name': col, 'tag': tag})
print('Wrote column tags to %s' % tag_csv)

# ── Audit report ──────────────────────────────────────────────────
train_rows = [r for r in dataset_rows if r['train_use'] == 'train']
pos_rows = [r for r in train_rows if r['label_status'] == 'positive']
neg_rows = [r for r in train_rows if r['label_status'] == 'negative']

online_cols = [c for c, tag in column_tags.items() if tag == 'online_feature']
label_cols = [c for c, tag in column_tags.items() if tag == 'label_only']

with open(OUT_MD, 'w') as f:
    f.write("""# Online-Safe Vulnerability Dataset Audit

**Date**: %s
**Purpose**: Rebuild features from clean-time data only. No VIS outcomes, no oracle labels as features.

---

## Column Classification

| Category | Count | Description |
|----------|-------|-------------|
| online_identity | %d | Task/state/window keys |
| online_feature | %d | Available at deployment from clean rollout only |
| label_only | %d | Training target — NOT input features |
| offline_audit_only | %d | Metadata — NOT input features |

""" % (datetime.now().strftime('%Y-%m-%d %H:%M'),
       sum(1 for _, t in FEATURE_COLS if t == 'online_identity'),
       sum(1 for _, t in FEATURE_COLS if t == 'online_feature'),
       sum(1 for _, t in FEATURE_COLS if t == 'label_only'),
       sum(1 for _, t in FEATURE_COLS if t == 'offline_audit_only')))

    f.write("""
## Online Feature Groups

| Group | Features | Source | Task Coverage |
|-------|----------|--------|---------------|
| Window position | window_position_norm, window_size | derived | 9/9 tasks |
| Phase heuristic | phase_bin_proxy, qpos_phase_class, phase_is_critical | heuristic | 9/9 tasks |
| Phase detector | predicted_phase, phase_confidence | ProprioNoStep | 3/9 tasks |
| Hazard scores | hazard_score_mean/max, release_safe_score_mean/min | ProprioNoStep checkpoint | 3/9 tasks |
| Proprio stats | gripper_command/qpos/width/eef/action stats | ProprioNoStep features | 3/9 tasks |
| Clean trace stats | Same stats from clean trace CSV | batch3b / milestone_7 runs | variable |

""")

    f.write("""
## Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows | %d |
| Train rows (label_use=train) | %d |
| Positive rows | %d |
| Negative rows | %d |
| Rows with ProprioNoStep features (3 tasks) | %d |
| Rows with clean trace found | %d |
| Rows with phase features | %d |

""" % (len(dataset_rows), len(train_rows), len(pos_rows), len(neg_rows),
       sum(1 for r in dataset_rows if r['proprio_available'] == 'YES'),
       sum(1 for r in dataset_rows if 'not_found' not in r.get('clean_trace_source', '')),
       sum(1 for r in dataset_rows if r['phase_source'] != 'missing')))

    # Column listing
    f.write("""
## Full Column Inventory

| Column | Tag | Description |
|--------|-----|-------------|
""")
    for col, tag in FEATURE_COLS:
        f.write('| %s | %s | |\n' % (col, tag))

    # Forbidden columns check
    f.write("""
## Forbidden Columns Audit

The following columns from the OLD dataset are explicitly EXCLUDED:

| Column | Reason |
|--------|--------|
| qpos_opening_delta | VIS outcome (B) |
| vis_open_count | VIS outcome (B) |
| action_bridge_confounded | VIS outcome (B) |
| label_action_bridge | Oracle (C) |
| label_physical_response | Oracle (C) |
| label_task_failure | Oracle (C) |
| qpos_label | VIS outcome (B) |
| done | VIS outcome (B) |
| task_failure | VIS outcome (B) |
| mechanism_type as feature | Oracle (C) |
| taxonomy as feature | Oracle (C) |
| label_source as feature | Oracle (C) |
| provenance_status | VIS metadata (B) |
| 3R result | VIS outcome (B) |

**All confirmed EXCLUDED from the online-safe dataset.**
""")

    # Feature completeness audit
    f.write("""
## Feature Completeness per Row

| Task | State | Window | Proprio? | Phase? | Clean Trace? | Online Features Filled |
|------|-------|--------|----------|--------|-------------|----------------------|
""")
    for r in dataset_rows:
        f.write('| %s | %s | [%s,%s] | %s | %s | %s | %s |\n' % (
            r['task_key'], r['state_id'], r['window_start'], r['window_end'],
            r['proprio_available'],
            'YES' if r['phase_source'] != 'missing' else 'no',
            'YES' if 'not_found' not in r.get('clean_trace_source', '') else 'no',
            r['feature_completeness']))

print('Wrote report to %s' % OUT_MD)
print()
print('=== Dataset Summary ===')
print('Total: %d rows' % len(dataset_rows))
print('Train: %d (pos=%d, neg=%d)' % (len(train_rows), len(pos_rows), len(neg_rows)))
print('With ProprioNoStep: %d' % sum(1 for r in dataset_rows if r['proprio_available'] == 'YES'))
print('With phase features: %d' % sum(1 for r in dataset_rows if r['phase_source'] != 'missing'))
print('Online feature columns: %d' % len(online_cols))
print('Label columns (target only): %d' % len(label_cols))
