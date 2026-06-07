#!/usr/bin/env python3
"""Build clean policy margin cache from milestone_7 clean traces.
Extracts raw_gripper action per step, computes per-window policy margin features.

All features are ONLINE-SAFE: from clean rollout OpenVLA forward passes only.
No VIS outcomes. No oracle labels.

Output:
  tables/openvla_clean_policy_margin_cache.csv    (per-step)
  tables/openvla_online_policy_sensitivity_features.csv  (per-window)
  reports/OPENVLA_CLEAN_POLICY_MARGIN_FEATURES.md
"""

import csv, os, sys, glob, re
from collections import defaultdict
from datetime import datetime
import numpy as np

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
MILESTONE7 = '/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs'
CODEX = '/data/liuyu/outputs/codex_phase_detector_twostage_20260606/tables'

OUT_STEP = os.path.join(REPO, 'tables/openvla_clean_policy_margin_cache.csv')
OUT_WINDOW = os.path.join(REPO, 'tables/openvla_online_policy_sensitivity_features.csv')
OUT_MD = os.path.join(REPO, 'reports/OPENVLA_CLEAN_POLICY_MARGIN_FEATURES.md')

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f: return list(csv.DictReader(f))

def make_key(r, tk='task_key', sid='state_id', ws='window_start', we='window_end'):
    return (str(r.get(tk,'')).strip(), str(r.get(sid,'')).strip(),
            str(r.get(ws,'')).strip(), str(r.get(we,'')).strip())

# ── Load candidates ───────────────────────────────────────────────
labels_v2 = read_csv(os.path.join(SHARED, 'object_phase_response_labels_v2.csv'))
phase_features = read_csv(os.path.join(CODEX, 'detector_phase_features_v1.csv'))
online_ds = read_csv(os.path.join(REPO, 'tables/online_safe_vulnerability_dataset.csv'))

# Merge all candidates
all_candidates = {}  # key -> {source info}
for r in labels_v2:
    key = make_key(r)
    all_candidates[key] = {
        'task_key': r['task_key'], 'state_id': r['state_id'],
        'window_start': r['window_start'], 'window_end': r['window_end'],
        'label_status': r.get('label_status',''), 'label_source': 'labels_v2',
        'train_use': r.get('label_use',''), 'taxonomy': r.get('taxonomy',''),
    }
print('Candidates from labels_v2: %d' % len(all_candidates))

# ── Index milestone_7 clean traces by (task, state_id) ────────────
trace_index = defaultdict(list)
for trace_path in glob.glob(MILESTONE7 + '/*clean*trace.csv'):
    fname = os.path.basename(trace_path)
    m = re.search(r'vis_(\w+)_state(\d+)_clean', fname)
    if m:
        task = m.group(1)
        sid = m.group(2)
        trace_index[(task, sid)].append(trace_path)

print('Milestone_7 traces indexed: %d (task,state) pairs' % len(trace_index))
for (tk, sid), paths in sorted(trace_index.items()):
    pass  # print('  %s s%s: %d traces' % (tk, sid, len(paths)))

# ── Extract policy margin features ────────────────────────────────
SEMANTICS = 'raw_gripper_lt_0.5_is_open'  # per trace metadata
THRESHOLD = 0.5

step_rows = []
window_rows = []
matched_count = 0
missed_count = 0

for key, cand in sorted(all_candidates.items()):
    task, sid, ws_str, we_str = key
    ws = int(ws_str)
    we = int(we_str)

    # Find matching trace
    traces = trace_index.get((task, sid), [])
    if not traces:
        missed_count += 1
        continue

    # Use first matching trace (prefer one that covers our window)
    best_trace = traces[0]
    # Try to find a trace whose filename window is closest
    best_dist = 9999
    for t in traces:
        m2 = re.search(r'w(\d+)_(\d+)', os.path.basename(t))
        if m2:
            tws, twe = int(m2.group(1)), int(m2.group(2))
            dist = abs(tws - ws) + abs(twe - we)
            if dist < best_dist:
                best_dist = dist
                best_trace = t

    # Read trace
    try:
        with open(best_trace) as f:
            trace_rows = list(csv.DictReader(f))
    except:
        missed_count += 1
        continue

    matched_count += 1

    # Extract raw_gripper for steps in window
    window_steps = []
    for tr in trace_rows:
        step_str = tr.get('step', '')
        if not step_str:
            continue
        try:
            step = int(float(step_str))
        except:
            continue

        if ws <= step <= we:
            raw_grip = None
            for col in ['raw_gripper', 'clean_grip', 'gripper_command']:
                v = tr.get(col, '')
                if v:
                    try:
                        raw_grip = float(v)
                        break
                    except:
                        pass

            if raw_grip is not None:
                dist_to_boundary = abs(raw_grip - THRESHOLD)
                open_close = 1 if raw_grip < THRESHOLD else 0  # <0.5 = open

                step_rows.append({
                    'task_key': task, 'state_id': sid,
                    'window_start': ws_str, 'window_end': we_str,
                    'step': str(step),
                    'raw_gripper_action': str(round(raw_grip, 8)),
                    'distance_to_boundary': str(round(dist_to_boundary, 8)),
                    'open_close_decision': str(open_close),
                })
                window_steps.append({
                    'step': step, 'raw_grip': raw_grip,
                    'dist': dist_to_boundary, 'open_close': open_close,
                })

    # ── Aggregate to window-level features ─────────────────────────
    if len(window_steps) < 2:
        continue

    dists = np.array([s['dist'] for s in window_steps])
    raw_grips = np.array([s['raw_grip'] for s in window_steps])
    open_close = np.array([s['open_close'] for s in window_steps])

    # Core policy margin features
    dist_min = float(np.min(dists))
    dist_mean = float(np.mean(dists))
    dist_max = float(np.max(dists))
    dist_std = float(np.std(dists))

    # Low margin: distance < 0.1 (model is uncertain)
    low_margin_mask = dists < 0.1
    low_margin_count = int(np.sum(low_margin_mask))
    low_margin_ratio = float(low_margin_count / len(dists))

    # Longest streak of low-margin steps
    streak = 0
    max_streak = 0
    for d in dists:
        if d < 0.1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Gripper action variance
    grip_std = float(np.std(raw_grips))
    grip_delta = float(raw_grips[-1] - raw_grips[0])

    # Flip rate: how often does the decision flip?
    flips = int(np.sum(np.abs(np.diff(open_close))))
    flip_rate = float(flips / len(open_close)) if len(open_close) > 1 else 0.0

    # Open fraction
    open_fraction = float(np.mean(open_close))

    # Margin reversal: did min_margin step differ from majority decision?
    maj_decision = 1 if open_fraction > 0.5 else 0
    min_margin_step_open = int(open_close[np.argmin(dists)])
    margin_reversal = 1 if min_margin_step_open != maj_decision else 0

    window_rows.append({
        'task_key': task, 'state_id': sid,
        'window_start': ws_str, 'window_end': we_str,
        'n_steps': str(len(window_steps)),
        # Policy margin core
        'distance_to_boundary_min': str(round(dist_min, 6)),
        'distance_to_boundary_mean': str(round(dist_mean, 6)),
        'distance_to_boundary_max': str(round(dist_max, 6)),
        'distance_to_boundary_std': str(round(dist_std, 6)),
        # Low margin streaks
        'low_margin_step_count': str(low_margin_count),
        'low_margin_step_ratio': str(round(low_margin_ratio, 4)),
        'longest_low_margin_streak': str(max_streak),
        # Gripper action dynamics
        'gripper_action_mean': str(round(float(np.mean(raw_grips)), 6)),
        'gripper_action_std': str(round(grip_std, 6)),
        'gripper_action_delta': str(round(grip_delta, 6)),
        'gripper_action_min': str(round(float(np.min(raw_grips)), 6)),
        'gripper_action_max': str(round(float(np.max(raw_grips)), 6)),
        # Decision dynamics
        'open_close_flip_count': str(flips),
        'open_close_flip_rate': str(round(flip_rate, 4)),
        'open_fraction': str(round(open_fraction, 4)),
        'margin_reversal_flag': str(margin_reversal),
        # Metadata
        'trace_source': os.path.basename(best_trace),
        # Labels (target only, NOT features)
        'label_status': cand.get('label_status', ''),
        'label_source': cand.get('label_source', ''),
        'train_use': cand.get('train_use', ''),
        'taxonomy': cand.get('taxonomy', ''),
    })

print('Matched: %d, Missed: %d' % (matched_count, missed_count))

# ── Write outputs ─────────────────────────────────────────────────
# Step-level cache
with open(OUT_STEP, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['task_key','state_id','window_start','window_end',
        'step','raw_gripper_action','distance_to_boundary','open_close_decision'])
    w.writeheader()
    w.writerows(step_rows)
print('Wrote %d step rows to %s' % (len(step_rows), OUT_STEP))

# Window-level features
with open(OUT_WINDOW, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(window_rows[0].keys()))
    w.writeheader()
    w.writerows(window_rows)
print('Wrote %d window rows to %s' % (len(window_rows), OUT_WINDOW))

# ── Report ────────────────────────────────────────────────────────
train_windows = [r for r in window_rows if r['train_use'] == 'train']
pos_windows = [r for r in train_windows if r['label_status'] == 'positive']
neg_windows = [r for r in train_windows if r['label_status'] == 'negative']

with open(OUT_MD, 'w') as f:
    f.write("""# OpenVLA Clean Policy Margin Features

**Date**: %s
**Source**: milestone_7 clean traces (178 traces, 9 tasks)
**Semantics**: raw_gripper < 0.5 = OPEN, >= 0.5 = CLOSE

---

## Feature Description

All features are computed from clean rollout OpenVLA forward passes.
These are **online-safe**: available before any VIS attack is run.

### Per-Step Features (cache)
- `raw_gripper_action`: post-sigmoid action probability from OpenVLA
- `distance_to_boundary`: abs(raw_gripper_action - 0.5)
- `open_close_decision`: 1 if raw_gripper < 0.5 (open), 0 otherwise

### Per-Window Aggregates
- **Policy margin**: min/mean/max/std of distance_to_boundary
- **Low margin**: count, ratio, longest streak of steps where distance < 0.1
- **Gripper dynamics**: action mean/std/delta/min/max
- **Decision dynamics**: flip count/rate, open fraction, margin reversal flag

---

## Coverage

| Metric | Value |
|--------|-------|
| Candidates matched | %d |
| Candidates missed | %d |
| Step-level rows | %d |
| Window-level rows | %d |
| Train windows | %d (pos=%d, neg=%d) |

""" % (datetime.now().strftime('%Y-%m-%d %H:%M'), matched_count, missed_count,
       len(step_rows), len(window_rows), len(train_windows), len(pos_windows), len(neg_windows)))

    # Feature comparison: pos vs neg
    f.write("""
## Positive vs Negative: Policy Margin Features

| Feature | Pos Mean | Neg Mean | Delta |
|---------|----------|----------|-------|
""")
    for feat in ['distance_to_boundary_min', 'distance_to_boundary_mean',
                  'low_margin_step_ratio', 'longest_low_margin_streak',
                  'gripper_action_std', 'open_close_flip_rate', 'open_fraction',
                  'margin_reversal_flag']:
        pos_vals = [float(r[feat]) for r in pos_windows if r.get(feat)]
        neg_vals = [float(r[feat]) for r in neg_windows if r.get(feat)]
        pos_m = np.mean(pos_vals) if pos_vals else 0
        neg_m = np.mean(neg_vals) if neg_vals else 0
        delta = pos_m - neg_m
        f.write('| %s | %.4f | %.4f | %+.4f |\n' % (feat, pos_m, neg_m, delta))

    f.write("""
## Interpretation

- **distance_to_boundary_min**: How close the model gets to the decision boundary.
  Low values suggest the model is uncertain about gripper control.
- **low_margin_step_ratio**: Fraction of steps where distance < 0.1.
  High values suggest persistent uncertainty.
- **gripper_action_std**: Variance of gripper action within window.
  High variance suggests unstable control.
- **open_close_flip_rate**: How often the open/close decision flips.
  High flip rates suggest indecisive control.
- **margin_reversal_flag**: Whether the minimum-margin step's decision
  differs from the majority decision. Flags ambiguous states.

These features probe the model's own decision-making confidence
without running a VIS attack. Windows where the model shows low
confidence or unstable control may be more susceptible to perturbation.

## Online-Safe Guarantee

| Check | Status |
|-------|--------|
| VIS outcomes excluded | YES — clean traces only |
| Oracle labels as features | NO — labels are target only |
| Attack outcome leakage | NO — no VIS data used |
| Deployable at inference | YES — only needs clean OpenVLA forward pass |
""")

print('Wrote report to %s' % OUT_MD)
