#!/usr/bin/env python3
"""Full31 metrics: frozen score definitions, strict label policy, ceiling-excluded."""
import csv, os, sys
import numpy as np
from collections import defaultdict

OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'

# ── Read merged window features ──────────────────────────────────
def load_all(name_pattern):
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT_DIR, name_pattern))):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows

rows = load_all('window_features_full31_shard*.csv')
if not rows:
    print('FATAL: no full31 files found')
    sys.exit(1)
print(f'Loaded {len(rows)} windows')

# ── Attach VIS labels ────────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    labels = { (r['task_key'].strip(), r['state_id'].strip(),
                int(r['window_start']), int(r['window_end'])): r
              for r in csv.DictReader(f) }

# Load mechanism labels if available
mech_path = os.path.join(REPO, 'tables', 'vulnerability_mechanism_taxonomy_audit.csv')
mech_map = {}
if os.path.exists(mech_path):
    with open(mech_path) as f:
        for r in csv.DictReader(f):
            k = (r['task_key'].strip(), r['state_id'].strip(),
                 r['window_start'].strip(), r['window_end'].strip())
            mech_map[k] = r.get('mechanism_type', '')

# ── Compute all frozen features ──────────────────────────────────
for row in rows:
    key = (row['task_key'], row['state_id'],
           int(row['window_start']), int(row['window_end']))
    lbl = labels.get(key, {})

    t = int(row['targeted_open_count'])
    c = int(row['clean_open_count'])
    r = int(row['random_open_count'])
    ts = int(row['targeted_longest_open_streak'])
    cs = int(row['clean_longest_open_streak'])
    rs = int(row['random_longest_open_streak'])
    n = int(row['n_probe_frames'])

    # Primary
    row['tmc_count'] = t - c
    row['tmc_streak'] = ts - cs
    # Secondary
    row['targeted_open_rate'] = round(t / max(n, 1), 4)
    row['random_open_rate'] = round(r / max(n, 1), 4)
    row['clean_open_rate_val'] = round(c / max(n, 1), 4)
    row['tmr_count'] = int(row['targeted_minus_random_open_count'])
    row['tmr_streak'] = int(row['targeted_minus_random_streak'])
    row['ceiling_flag'] = 1 if c >= 0.8 * n else 0

    # VIS labels
    vis_oc = int(lbl.get('vis_open_count', 0) or 0)
    qpos_delta = float(lbl.get('qpos_opening_delta', 0) or 0)
    qpos_label = lbl.get('qpos_label', '?').strip()
    phys_resp = float(lbl.get('label_physical_response', 0) or 0)
    label_status = lbl.get('label_status', '?').strip()
    taxonomy = lbl.get('taxonomy', '?').strip()
    mech = mech_map.get(key, '')

    row['vis_open_count'] = vis_oc
    row['qpos_delta'] = qpos_delta
    row['qpos_label'] = qpos_label
    row['phys_resp'] = phys_resp
    row['label_status'] = label_status
    row['taxonomy_full'] = taxonomy
    row['mechanism_type'] = mech if mech else ''

    # ── Frozen label policy ──────────────────────────────────────
    # physical_bridge_positive: phys_resp >= 1.0
    row['label_phys_bridge'] = 1 if phys_resp >= 1.0 else 0

    # command_susceptible_positive: VIS trace open_count >= 6
    row['label_cmd_sus'] = 1 if vis_oc >= 6 else 0

    # negative_clean: phys_resp == 0, NOT polluted, NOT uncertain
    is_polluted = 'polluted' in taxonomy.lower()
    is_uncertain = 'uncertain' in taxonomy.lower() or phys_resp == 0.5
    row['label_negative_clean'] = 1 if (phys_resp == 0 and not is_polluted and not is_uncertain) else 0

    # ignore_polluted
    row['label_ignore_polluted'] = 1 if (is_polluted or label_status == 'ignore') else 0

    # uncertain
    row['label_uncertain'] = 1 if (is_uncertain or phys_resp == 0.5) else 0

    # ceiling
    row['label_ceiling'] = row['ceiling_flag']

# ── Write merged features ────────────────────────────────────────
MERGED = os.path.join(OUT_DIR, 'window_features_full31_merged.csv')
all_keys = ['task_key','state_id','window_start','window_end',
            'label_status','taxonomy_full','mechanism_type',
            'n_probe_frames',
            'clean_open_count','clean_open_rate','clean_longest_open_streak',
            'targeted_open_count','targeted_open_rate','targeted_longest_open_streak',
            'random_open_count','random_open_rate','random_longest_open_streak',
            'tmc_count','tmr_count','tmc_streak','tmr_streak',
            'ceiling_flag',
            'gripper_delta_vs_clean','targeted_error_rate',
            'vis_open_count','qpos_delta','qpos_label',
            'label_phys_bridge','label_cmd_sus','label_negative_clean',
            'label_ignore_polluted','label_uncertain','label_ceiling']
with open(MERGED, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
print(f'Wrote merged: {MERGED}')

# ── Metrics ──────────────────────────────────────────────────────
def compute_metrics(y_true, y_score):
    """Returns AUROC, AUPRC, precision@K, top20% hit rate."""
    if len(set(y_true)) < 2 or sum(y_true) == 0:
        return {'auroc': float('nan'), 'auprc': float('nan'),
                'p_at_3': float('nan'), 'p_at_5': float('nan'),
                'p_at_top20pct': float('nan'), 'n_pos': int(sum(y_true)), 'n_total': len(y_true)}

    order = np.argsort(y_score)[::-1]
    y_ranked = np.array(y_true)[order]
    n_pos = sum(y_true)
    n_total = len(y_true)
    top_k_20 = max(1, int(n_total * 0.2))

    # AUROC
    pos_scores = [y_score[i] for i in range(n_total) if y_true[i] == 1]
    neg_scores = [y_score[i] for i in range(n_total) if y_true[i] == 0]
    auc = sum(1 for p in pos_scores for n in neg_scores if p > n)
    auc += 0.5 * sum(1 for p in pos_scores for n in neg_scores if p == n)
    auc /= (len(pos_scores) * len(neg_scores))

    # AUPRC
    tp = 0; auprc = 0.0; prev_recall = 0
    for i in range(n_total):
        if y_ranked[i] == 1: tp += 1
        precision = tp / (i + 1)
        recall = tp / max(n_pos, 1)
        auprc += precision * (recall - prev_recall)
        prev_recall = recall

    # Precision@K
    p_at_3 = sum(y_ranked[:3]) / 3 if n_total >= 3 else float('nan')
    p_at_5 = sum(y_ranked[:5]) / 5 if n_total >= 5 else float('nan')
    p_at_top20 = sum(y_ranked[:top_k_20]) / top_k_20

    # Top 20% hit rate (= recall in top 20%)
    top20_hit = sum(y_ranked[:top_k_20]) / max(n_pos, 1)

    return {'auroc': round(auc, 4), 'auprc': round(auprc, 4),
            'p_at_3': round(p_at_3, 4), 'p_at_5': round(p_at_5, 4),
            'p_at_top20pct': round(p_at_top20, 4),
            'top20_hit_rate': round(top20_hit, 4),
            'n_pos': n_pos, 'n_total': n_total}

# Score columns to evaluate
SCORES = {
    'tmc_count': 'targeted_minus_clean_open_count',
    'tmc_streak': 'targeted_minus_clean_streak',
    'targeted_open_rate': 'targeted_open_rate',
    'targeted_longest_open_streak': 'targeted_longest_open_streak',
    'tmr_count': 'targeted_minus_random_open_count',
    'tmr_streak': 'targeted_minus_random_streak',
}

# Label columns (frozen policy)
LABELS = {
    'label_cmd_sus': 'command_susceptible_positive',
    'label_phys_bridge': 'physical_bridge_positive',
    'label_negative_clean': 'negative_clean',
}

# Evaluate full set
print('\n=== FULL SET (all non-ceiling) ===')
full_no_ceiling = [r for r in rows if not r['ceiling_flag']]
print(f'n={len(full_no_ceiling)} (excluded {len(rows) - len(full_no_ceiling)} ceiling)')

all_metrics = []
for score_name, score_label in SCORES.items():
    raw = np.array([float(r.get(score_name, 0)) for r in full_no_ceiling])
    for label_name, label_str in LABELS.items():
        y = [int(r[label_name]) for r in full_no_ceiling]
        m = compute_metrics(y, raw)
        m['score'] = score_label; m['label'] = label_str
        m['subset'] = 'no_ceiling'
        all_metrics.append(m)

# Also evaluate ceiling-only subset
ceiling_only = [r for r in rows if r['ceiling_flag']]
if ceiling_only:
    print(f'\n=== CEILING ONLY (n={len(ceiling_only)}) ===')
    for score_name, score_label in SCORES.items():
        raw = np.array([float(r.get(score_name, 0)) for r in ceiling_only])
        # Only raw scores matter for ceiling (delta-to-clean is invalid)
        for label_name, label_str in LABELS.items():
            y = [int(r[label_name]) for r in ceiling_only]
            m = compute_metrics(y, raw)
            m['score'] = score_label; m['label'] = label_str
            m['subset'] = 'ceiling_only'
            all_metrics.append(m)

# ── Disagreement Review Queue ────────────────────────────────────
print('\n=== DISAGREEMENT REVIEW QUEUE ===')
queue = []

for row in rows:
    # High probe score but negative/polluted label
    if row['tmc_count'] >= 3 and row['label_phys_bridge'] == 0 and not row['ceiling_flag']:
        row['disagreement_type'] = 'HIGH_PROBE_NEGATIVE_LABEL'
        row['audit_action'] = 'Verify VIS trace; check if probe false positive or label error'
        queue.append(dict(row))

    # Positive label but low probe score (and not ceiling)
    if row['tmc_count'] <= 0 and row['label_phys_bridge'] == 1 and not row['ceiling_flag']:
        row['disagreement_type'] = 'POSITIVE_LABEL_LOW_PROBE'
        row['audit_action'] = 'Check if probe objective/reachability matches VIS; retry with higher PGD steps'
        queue.append(dict(row))

    # Ceiling positives
    if row['ceiling_flag'] and row['label_phys_bridge'] == 1:
        row['disagreement_type'] = 'CEILING_POSITIVE'
        row['audit_action'] = 'Clean model already opens; probe delta-to-clean invalid. Use raw targeted_open_rate instead.'
        queue.append(dict(row))

print(f'Queue: {len(queue)} entries')
for q in sorted(queue, key=lambda r: -r['tmc_count']):
    print(f'  [{q["disagreement_type"]}] {q["task_key"]} s{q["state_id"]} [{q["window_start"]},{q["window_end"]}] '
          f't-c={q["tmc_count"]} c_rate={q["clean_open_rate_val"]} t_rate={q["targeted_open_rate"]} '
          f'label={q["label_status"]} tax={q["taxonomy_full"][:30]} → {q["audit_action"]}')

# Write queue CSV
QUEUE_CSV = os.path.join(REPO, 'tables', 'active_probe_v1_disagreement_review_queue.csv')
if queue:
    qkeys = ['task_key','state_id','window_start','window_end',
             'disagreement_type','audit_action',
             'clean_open_count','clean_open_rate_val','clean_longest_open_streak',
             'targeted_open_count','targeted_open_rate','targeted_longest_open_streak',
             'random_open_count','random_open_rate','random_longest_open_streak',
             'tmc_count','tmr_count','tmc_streak','tmr_streak',
             'ceiling_flag','gripper_delta_vs_clean',
             'label_status','taxonomy_full','mechanism_type',
             'vis_open_count','qpos_delta','qpos_label',
             'label_phys_bridge','label_cmd_sus']
    with open(QUEUE_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=qkeys, extrasaction='ignore')
        w.writeheader(); w.writerows(queue)
    print(f'Wrote queue: {QUEUE_CSV}')

# ── Metrics CSV ──────────────────────────────────────────────────
METRICS_CSV = os.path.join(REPO, 'tables', 'active_probe_v1_temporal_metrics_full31.csv')
if all_metrics:
    with open(METRICS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)
    print(f'Wrote metrics: {METRICS_CSV}')

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Active Probe V1 Temporal — Full31 Readout')
lines.append('')
lines.append(f'**Date**: 2026-06-07')
lines.append(f'**Windows**: {len(rows)} total')
lines.append(f'**Ceiling excluded**: {len(full_no_ceiling)}')
lines.append(f'**Method**: PGD3 prefix_locked_gripper_open_margin, eps=6/255, 10 frames/window')
lines.append('')

# Label distribution
lines.append('## Label Distribution')
lines.append('')
for ln, ls in LABELS.items():
    y = [int(r[ln]) for r in rows]
    lines.append(f'- {ls}: {sum(y)}/{len(rows)} positive')
ceiling_n = sum(1 for r in rows if r['ceiling_flag'])
lines.append(f'- ceiling_flag: {ceiling_n}/{len(rows)}')
lines.append('')

# Metrics (no_ceiling subset)
lines.append('## Metrics — Ceiling-Excluded Subset (n=%d)' % len(full_no_ceiling))
lines.append('')
lines.append('| Score | Label | AUROC | AUPRC | P@3 | P@5 | P@Top20% | Top20%Hit |')
lines.append('|---|---|---|---|---|---|---|---|')
for m in all_metrics:
    if m['subset'] == 'no_ceiling':
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            m['score'], m['label'], m['auroc'], m['auprc'],
            m['p_at_3'], m['p_at_5'], m['p_at_top20pct'], m['top20_hit_rate']))
lines.append('')

# Best per label
lines.append('## Best Score per Label (no_ceiling)')
lines.append('')
for ln, ls in LABELS.items():
    subset = [m for m in all_metrics if m['label'] == ls and m['subset'] == 'no_ceiling' and not np.isnan(m['auroc'])]
    if subset:
        subset.sort(key=lambda m: -m['auroc'])
        best = subset[0]
        lines.append('- **%s**: best AUROC=%s (score=%s), P@3=%s, Top20%%Hit=%s' % (
            ls, best['auroc'], best['score'], best['p_at_3'], best['top20_hit_rate']))

lines.append('')
lines.append('## Gate Verdict')
lines.append('')

# Check gate criteria
cmd_best = [m for m in all_metrics if m['label'] == 'command_susceptible_positive' and m['subset'] == 'no_ceiling']
cmd_best.sort(key=lambda m: -m['auroc'])
phys_best = [m for m in all_metrics if m['label'] == 'physical_bridge_positive' and m['subset'] == 'no_ceiling']
phys_best.sort(key=lambda m: -m['auroc'])

if cmd_best and not np.isnan(cmd_best[0]['auroc']):
    best_cmd_auc = cmd_best[0]['auroc']
    lines.append('- command_susceptible AUROC: %s (score=%s)' % (best_cmd_auc, cmd_best[0]['score']))
    if best_cmd_auc >= 0.65:
        lines.append('- **PASS**: command_susceptible AUROC >= 0.65')
    else:
        lines.append('- **BELOW 0.65**')

if phys_best and not np.isnan(phys_best[0]['auroc']):
    lines.append('- physical_bridge AUROC: %s (score=%s)' % (phys_best[0]['auroc'], phys_best[0]['score']))

# Top-K precision check
if cmd_best:
    rand_base = sum(r['label_cmd_sus'] for r in full_no_ceiling) / max(len(full_no_ceiling), 1)
    lines.append('- P@3: %s (random baseline: ~%.2f)' % (cmd_best[0]['p_at_3'], rand_base))

lines.append('')
lines.append(f'## Disagreement Queue ({len(queue)} entries)')
lines.append('')
for q in sorted(queue, key=lambda r: -r['tmc_count']):
    lines.append(f'- [{q["disagreement_type"]}] **{q["task_key"]} s{q["state_id"]} [{q["window_start"]},{q["window_end"]}]** — '
                 f't-c={q["tmc_count"]}, clean_rate={q["clean_open_rate_val"]}, targeted_rate={q["targeted_open_rate"]}, '
                 f'label={q["label_status"]}, tax={q["taxonomy_full"][:25]}')
lines.append('')

report_text = '\n'.join(lines)
REPORT_PATH = os.path.join(REPO, 'reports', 'ACTIVE_PROBE_V1_TEMPORAL_FULL31_READOUT.md')
with open(REPORT_PATH, 'w') as f:
    f.write(report_text)
print(f'\nWrote report: {REPORT_PATH}')
print('\n=== DONE ===')
