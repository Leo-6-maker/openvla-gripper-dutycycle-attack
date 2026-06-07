#!/usr/bin/env python3
"""Active Probe v1 Re-analysis: Use existing v0b step features + VIS labels
to evaluate command_susceptible and physical_bridge predictability.

v0b step features are LOGIT-level (open_token_count), not decoded gripper.
This analysis treats open_token_count as a rough proxy for probe_open_count.
"""
import csv, json, os, sys, glob
import numpy as np
from collections import defaultdict

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_TABLE = REPO + '/tables/active_probe_v1_reanalysis_features.csv'
OUT_REPORT = REPO + '/reports/ACTIVE_PROBE_V1_REANALYSIS.md'
os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)

# ── Load VIS labels ─────────────────────────────────────────────
with open(SHARED + '/object_phase_response_labels_v2.csv') as f:
    labels = list(csv.DictReader(f))
print('Loaded %d labels' % len(labels))

# Index by (task, state, ws, we)
label_by_key = {}
for r in labels:
    k = (r['task_key'].strip(), r['state_id'].strip(),
         int(r['window_start']), int(r['window_end']))
    label_by_key[k] = r

# ── Load v0b step features ──────────────────────────────────────
step_files = glob.glob(REPO + '/tables/active_probe_v0b_step_features_shard_*.csv')
all_steps = []
for sf in sorted(step_files):
    with open(sf) as f:
        rows = list(csv.DictReader(f))
    print('  %s: %d steps' % (sf.split('/')[-1], len(rows)))
    all_steps.extend(rows)
print('Total steps: %d' % len(all_steps))

# ── Aggregate step → window ─────────────────────────────────────
def safe_float(v, default=0.0):
    try: return float(v)
    except (ValueError, TypeError): return default

def safe_int(v, default=0):
    try: return int(v)
    except (ValueError, TypeError): return default

window_data = defaultdict(lambda: {
    'steps': [],
    'open_score_gains': [],
    'open_token_counts': [],
    'close_token_counts': [],
    'grad_norms': [],
    'token_flips': [],
    'probe_errors': [],
})

for r in all_steps:
    key = (r['task_key'].strip(), r['state_id'].strip(),
           safe_int(r['window_start']), safe_int(r['window_end']))
    window_data[key]['steps'].append(safe_int(r['step']))
    window_data[key]['open_score_gains'].append(safe_float(r['max_open_score_gain']))
    window_data[key]['open_token_counts'].append(safe_int(r.get('open_token_count', 0)))
    window_data[key]['close_token_counts'].append(safe_int(r.get('close_token_count', 0)))
    window_data[key]['grad_norms'].append(safe_float(r['max_grad_norm']))
    window_data[key]['token_flips'].append(safe_int(r.get('any_token_flip', 0)))
    window_data[key]['probe_errors'].append(safe_int(r.get('any_probe_error', 0)))

# ── Build window-level features ─────────────────────────────────
rows_out = []
for key, d in sorted(window_data.items()):
    label = label_by_key.get(key, {})
    if not label:
        continue

    n = len(d['steps'])
    og = np.array(d['open_score_gains'])
    otc = np.array(d['open_token_counts'])
    ctc = np.array(d['close_token_counts'])
    tflip = np.array(d['token_flips'])
    perr = np.array(d['probe_errors'])

    # Probe open count proxy: open_token_count (logit-level, not decoded)
    # "probe_open_count" = sum of open_token_count across probe steps
    # "probe_open_rate" = mean(open_token_count / (open+close))
    total_tokens = otc + ctc + 1e-9
    probe_open_rate_per_step = otc / total_tokens
    # Longest open streak: count consecutive steps where otc > ctc
    open_dominant = (otc > ctc).astype(int)
    streak = 0; max_streak = 0
    for v in open_dominant:
        if v: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0

    # VIS labels
    vis_open_count = safe_int(label.get('vis_open_count', 0))
    qpos_delta = safe_float(label.get('qpos_opening_delta', 0))
    qpos_label = label.get('qpos_label', 'none').strip()
    phys_resp = safe_float(label.get('label_physical_response', 0))
    label_status = label.get('label_status', '').strip()
    taxonomy = label.get('taxonomy', '').strip()
    mechanism = label.get('mechanism_type', '').strip()

    # command_susceptible_label: VIS trace open_count >= threshold
    cmd_sus_k1 = 1 if vis_open_count >= 1 else 0
    cmd_sus_k3 = 1 if vis_open_count >= 3 else 0
    cmd_sus_k6 = 1 if vis_open_count >= 6 else 0
    cmd_sus_k10 = 1 if vis_open_count >= 10 else 0

    # physical_bridge_label
    phys_bridge = 1 if phys_resp >= 1.0 else 0
    phys_any = 1 if phys_resp >= 0.5 else 0

    row = {
        'task_key': key[0], 'state_id': key[1],
        'window_start': str(key[2]), 'window_end': str(key[3]),
        'n_probe_steps': str(n),
        # Probe features (logit-level proxy)
        'probe_open_score_gain_max': str(round(np.max(og), 6)),
        'probe_open_score_gain_mean': str(round(np.mean(og), 6)),
        'probe_open_token_count_total': str(int(np.sum(otc))),
        'probe_open_token_rate': str(round(float(np.mean(probe_open_rate_per_step)), 6)),
        'probe_open_dominant_streak': str(max_streak),
        'probe_open_dominant_count': str(int(np.sum(open_dominant))),
        'probe_token_flip_rate': str(round(float(np.mean(tflip)), 6)),
        'probe_error_rate': str(round(float(np.mean(perr)), 6)),
        # VIS trace labels
        'vis_open_count': str(vis_open_count),
        'qpos_opening_delta': str(round(qpos_delta, 6)),
        'qpos_label': qpos_label,
        'physical_response_score': str(phys_resp),
        # Derived labels
        'command_susceptible_k1': str(cmd_sus_k1),
        'command_susceptible_k3': str(cmd_sus_k3),
        'command_susceptible_k6': str(cmd_sus_k6),
        'command_susceptible_k10': str(cmd_sus_k10),
        'physical_bridge': str(phys_bridge),
        'physical_any': str(phys_any),
        # Metadata
        'label_status': label_status,
        'taxonomy': taxonomy,
        'mechanism_type': mechanism if mechanism else '',
    }
    rows_out.append(row)

# ── Write features ──────────────────────────────────────────────
if rows_out:
    with open(OUT_TABLE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print('Wrote %d windows to %s' % (len(rows_out), OUT_TABLE))
else:
    print('FATAL: no windows matched labels')
    sys.exit(1)

# ── Evaluate ────────────────────────────────────────────────────
def compute_metrics(y_true, y_score):
    """AUROC, AUPRC, precision@K."""
    if len(set(y_true)) < 2:
        return {'auroc': float('nan'), 'auprc': float('nan'),
                'p_at_1': float('nan'), 'p_at_3': float('nan'),
                'p_at_5': float('nan'), 'n_pos': int(sum(y_true)), 'n_total': len(y_true)}
    order = np.argsort(y_score)[::-1]
    y_ranked = np.array(y_true)[order]

    # AUROC: pairwise comparison
    pos = [y_score[i] for i in range(len(y_score)) if y_true[i] == 1]
    neg = [y_score[i] for i in range(len(y_score)) if y_true[i] == 0]
    auc = 0
    for p in pos:
        auc += sum(1 for n in neg if p > n) + 0.5 * sum(1 for n in neg if p == n)
    auc /= (len(pos) * len(neg))

    # AUPRC
    precisions = []; recalls = []; tp = 0
    total_pos = sum(y_true)
    for i, idx in enumerate(order):
        if y_ranked[i] == 1: tp += 1
        precisions.append(tp / (i + 1))
        recalls.append(tp / total_pos)
    auprc = 0
    for i in range(1, len(precisions)):
        auprc += (recalls[i] - recalls[i-1]) * precisions[i]
    auprc += recalls[0] * precisions[0]

    # Precision@K
    n_pos = sum(y_true)
    p_at_1 = y_ranked[0] if len(y_ranked) >= 1 else float('nan')
    p_at_3 = sum(y_ranked[:3]) / 3 if len(y_ranked) >= 3 else float('nan')
    p_at_5 = sum(y_ranked[:5]) / 5 if len(y_ranked) >= 5 else float('nan')

    return {'auroc': round(auc, 4), 'auprc': round(auprc, 4),
            'p_at_1': round(p_at_1, 4), 'p_at_3': round(p_at_3, 4),
            'p_at_5': round(p_at_5, 4), 'n_pos': n_pos, 'n_total': len(y_true)}

# Score columns to test
score_cols = [
    ('probe_open_score_gain_max', 'open_score_gain_max'),
    ('probe_open_score_gain_mean', 'open_score_gain_mean'),
    ('probe_open_token_count_total', 'open_token_count_total'),
    ('probe_open_token_rate', 'open_token_rate'),
    ('probe_open_dominant_streak', 'open_dominant_streak'),
    ('probe_open_dominant_count', 'open_dominant_count'),
    ('probe_token_flip_rate', 'token_flip_rate'),
]

# Label columns to predict
label_cols = [
    ('command_susceptible_k1', 'cmd_sus_k1'),
    ('command_susceptible_k3', 'cmd_sus_k3'),
    ('command_susceptible_k6', 'cmd_sus_k6'),
    ('command_susceptible_k10', 'cmd_sus_k10'),
    ('physical_bridge', 'phys_bridge'),
    ('physical_any', 'phys_any'),
]

all_metrics = []
for score_name, score_label in score_cols:
    # Normalize scores for stability (except counts)
    raw_scores = np.array([float(r[score_name]) for r in rows_out])
    if raw_scores.max() - raw_scores.min() > 0:
        scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    else:
        scores = raw_scores

    for label_name, label_short in label_cols:
        y_true = [int(r[label_name]) for r in rows_out]
        m = compute_metrics(y_true, scores)
        all_metrics.append({
            'score': score_label, 'label': label_short,
            'auroc': m['auroc'], 'auprc': m['auprc'],
            'p_at_1': m['p_at_1'], 'p_at_3': m['p_at_3'], 'p_at_5': m['p_at_5'],
            'n_pos': m['n_pos'], 'n_total': m['n_total'],
        })

# ── Write report ────────────────────────────────────────────────
lines = []
lines.append('# Active Probe V1 Re-analysis Report')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Input**: v0b step features (logit-level proxy, NOT decoded gripper)')
lines.append('**Windows**: %d' % len(rows_out))
lines.append('')
lines.append('## Data Summary')
lines.append('')
lines.append('| Label | N Pos | N Total |')
lines.append('|---|---|---|')
for ln, ls in label_cols:
    y = [int(r[ln]) for r in rows_out]
    lines.append('| %s | %d | %d |' % (ls, sum(y), len(y)))
lines.append('')

# AUROC/AUPRC table
lines.append('## AUROC / AUPRC')
lines.append('')
lines.append('| Score | Label | AUROC | AUPRC | P@1 | P@3 | P@5 |')
lines.append('|---|---|---|---|---|---|---|')
for m in all_metrics:
    lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
        m['score'], m['label'], m['auroc'], m['auprc'],
        m['p_at_1'], m['p_at_3'], m['p_at_5']))
lines.append('')

# Best per label
lines.append('## Best per Label')
lines.append('')
for ln, ls in label_cols:
    subset = [m for m in all_metrics if m['label'] == ls]
    subset.sort(key=lambda m: (0 if not np.isnan(m['auroc']) else 1, -abs(m['auroc'] - 0.5)))
    if subset:
        best = subset[0]
        lines.append('- **%s**: best AUROC=%s (score=%s, n_pos=%d)' % (
            ls, best['auroc'], best['score'], best['n_pos']))

lines.append('')
lines.append('## Conclusion')
lines.append('')
lines.append('These results use LOGIT-LEVEL open_token_count as a proxy for decoded gripper actions.')
lines.append('v0b step features do NOT contain actual decoded gripper actions.')
lines.append('Step 2 (active_probe_v1_temporal.py) is needed for true decoded gripper streak analysis.')
lines.append('')

report_text = '\n'.join(lines)
with open(OUT_REPORT, 'w') as f:
    f.write(report_text)
print(report_text)
print('Wrote report to %s' % OUT_REPORT)
