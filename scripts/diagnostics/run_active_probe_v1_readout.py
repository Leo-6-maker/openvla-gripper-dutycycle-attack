#!/usr/bin/env python3
"""Merge 12-window pilot results and write comprehensive readout."""
import csv, os, sys, json
import numpy as np
from collections import defaultdict

OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'

# ── Read all 12 windows ──────────────────────────────────────────
all_rows = []
for s in ['0', '1', '2']:
    f = os.path.join(OUT_DIR, f'window_features_pilot_shard{s}.csv')
    with open(f) as fh:
        all_rows.extend(list(csv.DictReader(fh)))
print(f'Loaded {len(all_rows)} windows')

# ── Attach VIS labels ────────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    labels = { (r['task_key'].strip(), r['state_id'].strip(),
                int(r['window_start']), int(r['window_end'])): r
              for r in csv.DictReader(f) }

# ── Compute metrics ──────────────────────────────────────────────
for row in all_rows:
    key = (row['task_key'], row['state_id'],
           int(row['window_start']), int(row['window_end']))
    lbl = labels.get(key, {})
    row['vis_open_count'] = lbl.get('vis_open_count', '?')
    row['qpos_opening_delta'] = lbl.get('qpos_opening_delta', '?')
    row['qpos_label'] = lbl.get('qpos_label', '?')
    row['phys_resp'] = lbl.get('label_physical_response', '?')
    row['mechanism'] = lbl.get('mechanism_type', '')
    # Derived
    t = int(row['targeted_open_count'])
    c = int(row['clean_open_count'])
    r = int(row['random_open_count'])
    row['tmc_count'] = t - c  # targeted minus clean
    row['tmr_count'] = int(row['targeted_minus_random_open_count'])
    row['tmc_streak'] = int(row['targeted_longest_open_streak']) - int(row['clean_longest_open_streak'])
    row['tmr_streak'] = int(row['targeted_minus_random_streak'])
    vis_oc = int(lbl.get('vis_open_count', 0) or 0)
    row['cmd_sus_vis_k6'] = 1 if vis_oc >= 6 else 0
    row['phys_bridge'] = 1 if float(lbl.get('label_physical_response', 0) or 0) >= 1.0 else 0
    row['phys_any'] = 1 if float(lbl.get('label_physical_response', 0) or 0) >= 0.5 else 0

# ── Write merged features ────────────────────────────────────────
MERGED = os.path.join(OUT_DIR, 'window_features_pilot_merged.csv')
keys = ['task_key','state_id','window_start','window_end','label_status','taxonomy',
        'mechanism','n_probe_frames',
        'clean_open_count','clean_open_rate','clean_longest_open_streak',
        'targeted_open_count','targeted_open_rate','targeted_longest_open_streak',
        'random_open_count','random_open_rate','random_longest_open_streak',
        'targeted_minus_random_open_count','targeted_minus_random_streak',
        'gripper_delta_vs_clean','targeted_error_rate',
        'tmc_count','tmr_count','tmc_streak','tmr_streak',
        'vis_open_count','qpos_opening_delta','qpos_label','phys_resp',
        'cmd_sus_vis_k6','phys_bridge','phys_any']
with open(MERGED, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader(); w.writerows(all_rows)
print(f'Wrote {MERGED}')

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Active Probe V1 Temporal Pilot — 12-Window Readout')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Method**: PGD3 prefix_locked_gripper_open_margin, eps=6/255')
lines.append('**Frames per window**: up to 10 evenly spaced')
lines.append('**GPU pairs**: 0,1 / 2,6 / 4,5 (3-way parallel)')
lines.append('')

# Summary table
lines.append('## Per-Window Results')
lines.append('')
lines.append('Sorted by targeted-minus-clean (t−c) open count:')
lines.append('')
lines.append('| Window | Label | Tax | Clean | Targeted | Random | t−c | t−r | VisOpen | Phys |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for row in sorted(all_rows, key=lambda r: -r['tmc_count']):
    lines.append('| %s s%s [%s,%s] | %s | %s | %s | %s | %s | %+d | %+d | %s | %s |' % (
        row['task_key'], row['state_id'], row['window_start'], row['window_end'],
        row['label_status'], row['taxonomy'][:25],
        row['clean_open_count'], row['targeted_open_count'], row['random_open_count'],
        row['tmc_count'], row['tmr_count'],
        row['vis_open_count'], row['phys_resp']))
lines.append('')

# Strong induction analysis
strong = [r for r in all_rows if r['tmc_count'] >= 3]
weak = [r for r in all_rows if r['tmc_count'] <= 1]
lines.append('## Strong Induction (t−c >= 3)')
lines.append('')
lines.append(f'{len(strong)}/12 windows:')
for r in sorted(strong, key=lambda r: -r['tmc_count']):
    lines.append('- **%s s%s [%s,%s]** t−c=%+d t−r=%+d — label=%s taxonomy=%s vis_open=%s phys=%s' % (
        r['task_key'], r['state_id'], r['window_start'], r['window_end'],
        r['tmc_count'], r['tmr_count'],
        r['label_status'], r['taxonomy'][:30],
        r['vis_open_count'], r['phys_resp']))

lines.append('')
lines.append('Of these 5:')
lines.append('- 2/4 physical_bridge positives (butter s0, cream_cheese s4) → RECALL=50%')
lines.append('- 1 negative/no_action_bridge (ketchup s4) → probe vs VIS DISAGREEMENT')
lines.append('- 2 ignore/polluted (salad_dressing s0, butter s3) → VIS labels UNRELIABLE')

lines.append('')
lines.append('## Weak/No Induction (t−c <= 1)')
lines.append('')
lines.append(f'{len(weak)}/12 windows:')
for r in sorted(weak, key=lambda r: r['tmc_count']):
    note = ''
    c = int(r['clean_open_count']); n = int(r['n_probe_frames'])
    if c >= 0.9 * n: note = ' [CEILING: clean already open]'
    lines.append('- **%s s%s [%s,%s]** t−c=%+d — label=%s taxonomy=%s%s' % (
        r['task_key'], r['state_id'], r['window_start'], r['window_end'],
        r['tmc_count'], r['label_status'], r['taxonomy'][:30], note))

lines.append('')
lines.append('## Gate Evaluation')
lines.append('')

# Per-user gate criteria
# 1. targeted_minus_random_streak separation
pos_tmr = [r['tmr_streak'] for r in all_rows if r['phys_bridge'] == 1]
neg_tmr = [r['tmr_streak'] for r in all_rows if r['phys_bridge'] == 0]
lines.append('### 1. targeted_minus_random_streak separation')
lines.append(f'- Positives (n={len(pos_tmr)}): mean t-r streak = {np.mean(pos_tmr):.1f}, values = {pos_tmr}')
lines.append(f'- Negatives (n={len(neg_tmr)}): mean t-r streak = {np.mean(neg_tmr):.1f}, values = {neg_tmr}')
sep = abs(np.mean(pos_tmr) - np.mean(neg_tmr))
verdict = "BORDERLINE" if sep >= 1 else "WEAK"
lines.append(f'- Separation: {sep:.1f} → {verdict}')

# 2. Rough AUROC estimate
def fast_auroc(y_true, y_score):
    if len(set(y_true)) < 2: return float('nan')
    pos = [s for i, s in enumerate(y_score) if y_true[i] == 1]
    neg = [s for i, s in enumerate(y_score) if y_true[i] == 0]
    auc = sum(1 for p in pos for n in neg if p > n) + 0.5*sum(1 for p in pos for n in neg if p == n)
    return auc / (len(pos) * len(neg))

for score_name in ['tmc_count', 'tmr_count', 'tmc_streak', 'tmr_streak']:
    scores = [r[score_name] for r in all_rows]
    for label_name in ['phys_bridge', 'cmd_sus_vis_k6']:
        y = [r[label_name] for r in all_rows]
        if sum(y) >= 2 and sum(y) < len(y):
            auc = fast_auroc(y, scores)
            lines.append(f'- AUROC({score_name} → {label_name}): {auc:.4f} (n_pos={sum(y)})')

lines.append('')
lines.append('### 3. precision@topK')
top3 = sorted(all_rows, key=lambda r: -r['tmc_count'])[:3]
top3_pos = sum(1 for r in top3 if r['phys_bridge'] == 1)
lines.append(f'- Top-3 by t-c: {top3_pos}/3 physical_bridge positives = {top3_pos/3:.0%}')
top3_names = [(r["task_key"] + " s" + r["state_id"], r["tmc_count"]) for r in top3]
lines.append(f'- Top-3 windows: {top3_names}')

top4 = sorted(all_rows, key=lambda r: -r['tmc_count'])[:4]
top4_pos = sum(1 for r in top4 if r['phys_bridge'] == 1)
lines.append(f'- Top-4 by t−c: {top4_pos}/4 physical_bridge positives = {top4_pos/4:.0%}')

lines.append('')
lines.append('## Gate Verdict: BORDERLINE — PROCEED WITH CAVEATS')
lines.append('')
lines.append('**Evidence FOR proceeding:**')
lines.append('- 2/4 physical_bridge positives show clear probe signal (t−c >= 4)')
lines.append('- 2 "ignore/polluted" windows show very strong signal → may be real positives VIS missed')
lines.append('- Clean/non-susceptible negatives correctly show no signal')
lines.append('- Signal is above random baseline')
lines.append('')
lines.append('**Evidence AGAINST:**')
lines.append('- 2/4 positives missed (1 ceiling, 1 unclear)')
lines.append('- 1 labeled negative shows signal (ketchup s4 — probe vs VIS disagreement)')
lines.append('- Small n=12, AUROC unreliable')
lines.append('- VIS labels are noisy (polluted windows, ceiling effects)')
lines.append('')
lines.append('**Recommendation:**')
lines.append('Proceed to full31 to resolve label noise with larger sample.')
lines.append('Key metric: targeted_minus_clean open count (NOT targeted_minus_random).')
lines.append('Treat "polluted" label + strong probe signal as candidate rediscoveries.')
lines.append('')

report_text = '\n'.join(lines)
REPORT_PATH = os.path.join(REPO, 'reports', 'ACTIVE_PROBE_V1_TEMPORAL_12_READOUT.md')
with open(REPORT_PATH, 'w') as f:
    f.write(report_text)
print(report_text)
print(f'Wrote report to {REPORT_PATH}')
