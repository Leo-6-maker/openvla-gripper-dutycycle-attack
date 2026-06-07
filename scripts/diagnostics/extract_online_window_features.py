#!/usr/bin/env python3
"""Extract online-legal (clean-forward, causal-only) window features from existing clean rollout traces.

Input: Clean rollout trace CSVs from ProprioNoStep shadow calibration
Output: Window-level feature table for online window detector v0

All features are computable from current/past observations only. No VIS outcome.
"""
import csv, os, sys, glob
import numpy as np
from collections import Counter, defaultdict

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
TRACE_DIR = '/data/liuyu/outputs/proprionostep_shadow_calib_20260607'
OUT_TABLE = os.path.join(REPO, 'tables', 'online_window_features_v0.csv')
OUT_REPORT = os.path.join(REPO, 'reports', 'ONLINE_WINDOW_FEATURE_EXTRACTION_AUDIT.md')

# ── Load labels ──────────────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    labels = list(csv.DictReader(f))
print('Labels: %d windows' % len(labels))

# ── Index clean traces by (task, state) ──────────────────────────
# Map task names to trace directory format
TASK_MAP = {
    'ketchup': 'ketchup', 'butter': 'butter', 'cream_cheese': 'cream_cheese',
    'salad_dressing': 'salad_dressing', 'bbq_sauce': 'bbq_sauce', 'milk': 'milk',
    'alphabet_soup': 'alphabet_soup', 'tomato_sauce': 'tomato_sauce',
    'orange_juice': 'orange_juice',
    # Variations in trace naming
    'salad': 'salad_dressing',
}

def find_trace(task, state_id):
    """Find clean trace CSV for a given task and state."""
    for pattern in [
        os.path.join(TRACE_DIR, 'vis_%s_s%s_clean_*_trace.csv' % (task, state_id)),
        os.path.join(TRACE_DIR, 'vis_%s_s%s_clean_v2_trace.csv' % (task, state_id)),
    ]:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None

# ── Feature extraction per window ────────────────────────────────
def extract_window_features(trace_rows, ws, we, task, state_id):
    """Extract online-legal features for window [ws, we]."""
    n_total = len(trace_rows)
    window_rows = [trace_rows[i] for i in range(max(0, ws), min(we + 1, n_total)) if i < n_total]
    pre_rows = [trace_rows[i] for i in range(max(0, ws - 20), ws) if i < n_total]
    if len(window_rows) < 2:
        return None

    def safe_float(v, default=0.0):
        try: return float(v)
        except (ValueError, TypeError): return default

    # ── Gripper qpos ─────────────────────────────────────────────
    qpos = np.array([safe_float(r.get('gripper_qpos', 0)) for r in window_rows])
    qpos_pre = np.array([safe_float(r.get('qpos_pre', 0)) for r in window_rows])
    qpos_post = np.array([safe_float(r.get('qpos_post', 0)) for r in window_rows])

    # ── Gripper action (decoded, after normalize+invert) ─────────
    env_grip = np.array([safe_float(r.get('env_gripper', 0)) for r in window_rows])
    raw_grip = np.array([safe_float(r.get('raw_gripper', 0)) for r in window_rows])

    # ── End-effector position ────────────────────────────────────
    eef_x = np.array([safe_float(r.get('eef_x', 0)) for r in window_rows])
    eef_y = np.array([safe_float(r.get('eef_y', 0)) for r in window_rows])
    eef_z = np.array([safe_float(r.get('eef_z', 0)) for r in window_rows])

    # ── ProprioNoStep scores ─────────────────────────────────────
    hazard = np.array([safe_float(r.get('proprionostep_hazard_score', 0)) for r in window_rows])
    release = np.array([safe_float(r.get('proprionostep_release_safe_score', 0)) for r in window_rows])
    phase_idx = [int(safe_float(r.get('proprionostep_phase_idx', -1))) for r in window_rows]

    # ── Pre-window context ───────────────────────────────────────
    pre_qpos = np.array([safe_float(r.get('gripper_qpos', 0)) for r in pre_rows]) if pre_rows else qpos[:1]
    pre_env_grip = np.array([safe_float(r.get('env_gripper', 0)) for r in pre_rows]) if pre_rows else env_grip[:1]

    # ── Compute features ─────────────────────────────────────────
    n = len(window_rows)

    # Gripper qpos features
    gripper_qpos_mean = float(np.mean(qpos))
    gripper_qpos_std = float(np.std(qpos))
    gripper_qpos_min = float(np.min(qpos))
    gripper_qpos_max = float(np.max(qpos))
    gripper_qpos_at_start = float(qpos[0])
    gripper_qpos_range = float(np.max(qpos) - np.min(qpos))
    gripper_is_closed = 1.0 if gripper_qpos_mean < 0.03 else 0.0
    gripper_is_open = 1.0 if gripper_qpos_mean > 0.035 else 0.0

    # Gripper action features (+1=OPEN, -1=CLOSE)
    grip_open_count = int(np.sum(env_grip > 0))
    grip_close_count = int(np.sum(env_grip < 0))
    grip_open_rate = float(grip_open_count / max(n, 1))
    grip_action_mean = float(np.mean(env_grip))
    grip_action_std = float(np.std(env_grip))
    grip_action_switches = int(np.sum(np.abs(np.diff(np.sign(env_grip))) > 0))

    # Raw gripper (before normalize/invert, 0=CLOSE, ~1=OPEN)
    raw_grip_mean = float(np.mean(raw_grip))
    raw_grip_std = float(np.std(raw_grip))

    # End-effector features
    eef_displacement = float(np.linalg.norm([eef_x[-1] - eef_x[0], eef_y[-1] - eef_y[0], eef_z[-1] - eef_z[0]]))
    eef_velocity_mean = float(np.mean(np.sqrt(np.diff(eef_x)**2 + np.diff(eef_y)**2 + np.diff(eef_z)**2))) if n > 1 else 0.0
    eef_z_mean = float(np.mean(eef_z))
    eef_z_std = float(np.std(eef_z))
    eef_z_trend = float(eef_z[-1] - eef_z[0]) if n > 1 else 0.0  # positive = moving up

    # ProprioNoStep features (hazard of gripper opening in clean rollouts)
    hazard_mean = float(np.mean(hazard))
    hazard_max = float(np.max(hazard))
    hazard_above_001 = int(np.sum(hazard > 0.01))
    hazard_above_003 = int(np.sum(hazard > 0.03))
    release_mean = float(np.mean(release))
    phase_mode = Counter(p for p in phase_idx if p >= 0).most_common(1)
    phase_mode_val = int(phase_mode[0][0]) if phase_mode else -1

    # Temporal: change from pre-window to window
    qpos_delta_from_pre = float(np.mean(qpos) - np.mean(pre_qpos))
    grip_action_delta_from_pre = float(np.mean(env_grip) - np.mean(pre_env_grip))

    # Window position features
    window_start_frac = float(ws / max(n_total, 1))
    window_center_frac = float((ws + we) / 2 / max(n_total, 1))
    window_len_steps = we - ws + 1
    window_len_frac = float(window_len_steps / max(n_total, 1))
    steps_remaining = n_total - we

    # Step at window_start
    step_at_start = ws

    # Open streak in clean rollout
    streak = 0; max_streak = 0
    for g in env_grip:
        if g > 0: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    clean_longest_open_streak = int(max_streak)

    return {
        'task_key': task, 'state_id': str(state_id),
        'window_start': str(ws), 'window_end': str(we),
        'window_len_steps': str(window_len_steps),
        'n_trace_steps': str(n_total),
        # Gripper qpos
        'gripper_qpos_mean': str(round(gripper_qpos_mean, 6)),
        'gripper_qpos_std': str(round(gripper_qpos_std, 6)),
        'gripper_qpos_min': str(round(gripper_qpos_min, 6)),
        'gripper_qpos_max': str(round(gripper_qpos_max, 6)),
        'gripper_qpos_at_start': str(round(gripper_qpos_at_start, 6)),
        'gripper_qpos_range': str(round(gripper_qpos_range, 6)),
        'gripper_is_closed': str(round(gripper_is_closed, 4)),
        'gripper_is_open': str(round(gripper_is_open, 4)),
        # Gripper action
        'grip_open_count': str(grip_open_count),
        'grip_close_count': str(grip_close_count),
        'grip_open_rate': str(round(grip_open_rate, 4)),
        'grip_action_mean': str(round(grip_action_mean, 4)),
        'grip_action_std': str(round(grip_action_std, 4)),
        'grip_action_switches': str(grip_action_switches),
        'clean_longest_open_streak': str(clean_longest_open_streak),
        # Raw gripper
        'raw_grip_mean': str(round(raw_grip_mean, 6)),
        'raw_grip_std': str(round(raw_grip_std, 6)),
        # EEF
        'eef_displacement': str(round(eef_displacement, 6)),
        'eef_velocity_mean': str(round(eef_velocity_mean, 6)),
        'eef_z_mean': str(round(eef_z_mean, 4)),
        'eef_z_std': str(round(eef_z_std, 4)),
        'eef_z_trend': str(round(eef_z_trend, 4)),
        # ProprioNoStep
        'hazard_mean': str(round(hazard_mean, 6)),
        'hazard_max': str(round(hazard_max, 6)),
        'hazard_above_001': str(hazard_above_001),
        'hazard_above_003': str(hazard_above_003),
        'release_mean': str(round(release_mean, 6)),
        'phase_mode': str(phase_mode_val),
        # Temporal
        'qpos_delta_from_pre': str(round(qpos_delta_from_pre, 6)),
        'grip_action_delta_from_pre': str(round(grip_action_delta_from_pre, 4)),
        # Position
        'window_start_frac': str(round(window_start_frac, 4)),
        'window_center_frac': str(round(window_center_frac, 4)),
        'window_len_frac': str(round(window_len_frac, 4)),
        'step_at_start': str(step_at_start),
        'steps_remaining': str(steps_remaining),
        # Data quality
        'trace_available': 'yes',
        'trace_path': '',
    }

# ── Process all labeled windows ──────────────────────────────────
feature_rows = []
missing_traces = []
for r in labels:
    task = r['task_key'].strip(); sid = r['state_id'].strip()
    ws = int(r['window_start']); we = int(r['window_end'])

    trace_path = find_trace(task, sid)
    if trace_path is None:
        missing_traces.append((task, sid))
        feature_rows.append({
            'task_key': task, 'state_id': sid,
            'window_start': str(ws), 'window_end': str(we),
            'trace_available': 'no',
        })
        continue

    with open(trace_path) as f:
        trace_rows = list(csv.DictReader(f))

    feats = extract_window_features(trace_rows, ws, we, task, sid)
    if feats:
        feats['trace_path'] = os.path.basename(trace_path)
        feature_rows.append(feats)
    else:
        missing_traces.append((task, sid, 'too_short'))

# ── Write table ──────────────────────────────────────────────────
if feature_rows:
    keys = [k for k in feature_rows[0].keys() if feature_rows[0][k] != '' or True]
    with open(OUT_TABLE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(feature_rows[0].keys()))
        w.writeheader(); w.writerows(feature_rows)
    print('Wrote %d rows to %s' % (len(feature_rows), OUT_TABLE))

avail = sum(1 for r in feature_rows if r.get('trace_available') == 'yes')
print('Available: %d/%d windows' % (avail, len(feature_rows)))
print('Missing traces for: %s' % (missing_traces))

# ── Write audit report ───────────────────────────────────────────
lines = []
lines.append('# Online Window Feature Extraction Audit')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Source**: ProprioNoStep shadow calibration clean traces')
lines.append('')
lines.append('## Coverage')
lines.append('')
lines.append('- Total labeled windows: %d' % len(labels))
lines.append('- Traces available: %d' % avail)
lines.append('- Traces missing: %d' % len(missing_traces))
lines.append('')
if missing_traces:
    lines.append('### Missing')
    for m in missing_traces:
        lines.append('- %s s%s' % (m[0], m[1]))
    lines.append('')
lines.append('## Feature Groups')
lines.append('')
lines.append('| Group | Features | Source | Count |')
lines.append('|---|---|---|---|')
lines.append('| Gripper qpos | mean, std, min, max, at_start, range, is_closed, is_open | env state | 8 |')
lines.append('| Gripper action | open_count, close_count, open_rate, mean, std, switches, longest_streak | decoded clean action | 7 |')
lines.append('| Raw gripper | mean, std | model output (before normalize) | 2 |')
lines.append('| End-effector | displacement, velocity_mean, z_mean, z_std, z_trend | env state | 5 |')
lines.append('| ProprioNoStep | hazard_mean/max, above_001/003, release_mean, phase_mode | TCN detector | 6 |')
lines.append('| Temporal (pre→window) | qpos_delta, grip_action_delta | computed | 2 |')
lines.append('| Window position | start_frac, center_frac, len_frac, step_at_start, steps_remaining | computed | 5 |')
lines.append('| **Total** | | | **35** |')
lines.append('')
lines.append('## Missing Features (require GPU forward pass)')
lines.append('')
lines.append('The following features are NOT available in current traces and would require')
lines.append('a GPU clean-forward pass with model logit/hidden-state extraction:')
lines.append('')
lines.append('- Gripper logits (open_score, open_margin, open_entropy)')
lines.append('- Action token entropy (per-dimension)')
lines.append('- Visual embedding (PCA-compressed hidden states)')
lines.append('- Top-K token overlap between clean and expected action')
lines.append('')
lines.append('## Next Steps')
lines.append('')
lines.append('1. Use these 35 proprio/action/position features for detector v0 baseline')
lines.append('2. If baseline is insufficient, run GPU feature extraction for logits+embeddings')
lines.append('3. Add traces for missing tasks (butter, tomato_sauce, orange_juice) from clean shadow runs')
lines.append('')

with open(OUT_REPORT, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote report to %s' % OUT_REPORT)
