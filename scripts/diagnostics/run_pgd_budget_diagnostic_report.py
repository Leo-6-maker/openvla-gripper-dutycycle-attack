#!/usr/bin/env python3
"""Merge PGD budget diagnostic CSVs and write final report with decision logic.

Key: compare PGD3_3f vs PGD10_3f on SAME 3 frames. Do NOT compare against old full31.
"""
import csv, os, sys, glob
import numpy as np

OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'

# ── Merge all shard CSVs ─────────────────────────────────────────
rows = []
for f in sorted(glob.glob(os.path.join(OUT_DIR, 'pgd_budget_diagnostic_pgd_budget_shard*.csv'))):
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))
if not rows:
    print('FATAL: no diagnostic CSVs found')
    # List all files in output dir for debugging
    for f in sorted(os.listdir(OUT_DIR)):
        print(' ', f)
    sys.exit(1)
print('Merged %d diagnostic rows' % len(rows))

# ── Attach VIS labels ───────────────────────────────────────────
with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    labels = { (r['task_key'].strip(), r['state_id'].strip(),
                int(r['window_start']), int(r['window_end'])): r
              for r in csv.DictReader(f) }

# ── Attach full31 PGD3 (for reference only, NOT for comparison) ──
with open(os.path.join(OUT_DIR, 'window_features_full31_merged.csv')) as f:
    full31 = { (r['task_key'], r['state_id'],
                int(r['window_start']), int(r['window_end'])): r
              for r in csv.DictReader(f) }

# ── Compute per-row diagnosis ────────────────────────────────────
for row in rows:
    tk = row['task_key']; sid = row['state_id']
    ws = int(row['window_start']); we = int(row['window_end'])
    key = (tk, sid, ws, we)
    lbl = labels.get(key, {})
    f31 = full31.get(key, {})

    # ── Extract metrics (3-frame, same frames) ──────────────────
    c_cnt = int(row.get('clean_open_count', 0))
    r_cnt = int(row.get('random_open_count', 0))
    p3_cnt = int(row.get('targeted_open_count_pgd3', 0))
    p10_cnt = int(row.get('targeted_open_count_pgd10', 0))
    p3_streak = int(row.get('targeted_longest_streak_pgd3', 0))
    p10_streak = int(row.get('targeted_longest_streak_pgd10', 0))
    n_frames = int(row.get('n_frames', 3))

    # ── VIS metrics ──────────────────────────────────────────────
    vis_open = int(lbl.get('vis_open_count', 0) or 0)
    vis_label = lbl.get('label_status', '?').strip()
    taxonomy = lbl.get('taxonomy', '?').strip()
    phys_resp = float(lbl.get('label_physical_response', 0) or 0)
    source_batch = lbl.get('source_batch', '?').strip()

    # VIS budget info
    vis_budgets = {'batch1': 20, 'batch3': 40, 'batch3b': 40}
    vis_pgd = vis_budgets.get(source_batch, 20)

    # ── Compute deltas (PGD3_3f and PGD10_3f, same 3 frames) ───
    p3_mc = p3_cnt - c_cnt   # PGD3 minus clean (3 frames)
    p10_mc = p10_cnt - c_cnt # PGD10 minus clean (3 frames)
    p3_mr = p3_cnt - r_cnt   # PGD3 minus random (3 frames)
    p10_mr = p10_cnt - r_cnt # PGD10 minus random (3 frames)

    row['p3_mc'] = p3_mc; row['p10_mc'] = p10_mc
    row['p3_mr'] = p3_mr; row['p10_mr'] = p10_mr

    # ── Decision per row ─────────────────────────────────────────
    dtype = row.get('disagreement_type', '').strip()
    ceiling = int(row.get('ceiling_flag', 0))

    # For full31 reference (NOT used for decision, just context)
    f31_tmc = int(f31.get('tmc_count', 0)) if f31 else 0
    row['f31_pgd3_tmc_ref'] = f31_tmc

    # Row-level PGD3 decision
    if p3_mc > 0:
        p3_decision = 'PROBE_OPEN'
    elif p3_mc < 0:
        p3_decision = 'PROBE_CLOSE'
    else:
        p3_decision = 'PROBE_NEUTRAL'

    # Row-level PGD10 decision
    if p10_mc > 0:
        p10_decision = 'PROBE_OPEN'
    elif p10_mc < 0:
        p10_decision = 'PROBE_CLOSE'
    else:
        p10_decision = 'PROBE_NEUTRAL'

    row['pgd3_decision'] = p3_decision
    row['pgd10_decision'] = p10_decision

    # ── Status transition ────────────────────────────────────────
    vis_is_open = vis_open >= 6  # VIS label says susceptible
    p3_is_open = p3_mc > 0       # PGD3 says susceptible
    p10_is_open = p10_mc > 0     # PGD10 says susceptible

    if not vis_is_open and not p3_is_open and not p10_is_open:
        transition = 'AGREE_NEGATIVE'
    elif vis_is_open and p3_is_open and p10_is_open:
        transition = 'AGREE_POSITIVE'
    elif not vis_is_open and p3_is_open and not p10_is_open:
        transition = 'REPAIRED: PGD10 fixed FP'
    elif vis_is_open and not p3_is_open and p10_is_open:
        transition = 'REPAIRED: PGD10 fixed FN'
    elif not vis_is_open and p3_is_open and p10_is_open:
        transition = 'PERSISTENT_FP: both PGD3 and PGD10 disagree with VIS'
    elif vis_is_open and not p3_is_open and not p10_is_open:
        transition = 'PERSISTENT_FN: both PGD3 and PGD10 miss VIS signal'
    elif not vis_is_open and not p3_is_open and p10_is_open:
        transition = 'NEW_FP: PGD10 introduces false positive'
    else:
        transition = 'OTHER'

    # Special: ceiling rows
    if ceiling:
        # For ceiling, "repair" means PGD10 OPEN rate >= PGD3 OPEN rate
        if p10_cnt >= p3_cnt:
            transition = 'CEILING: PGD10 >= PGD3 in raw open count'
        else:
            transition = 'CEILING: PGD10 < PGD3 (unexpected)'

    row['status_transition'] = transition
    row['vis_label'] = vis_label
    row['vis_open_count'] = vis_open
    row['vis_pgd_steps'] = vis_pgd
    row['taxonomy'] = taxonomy

    # ── Diagnosis per row ────────────────────────────────────────
    diag_parts = []
    if 'REPAIRED' in transition:
        diag_parts.append('PGD3_UNDERPOWERED: higher budget corrects the error')
    elif 'PERSISTENT' in transition:
        if p10_mc == p3_mc:
            diag_parts.append('BUDGET_INSENSITIVE: PGD10 same as PGD3 — not a budget issue')
        else:
            diag_parts.append('PARTIAL_BUDGET: PGD10 changes signal but does not align with VIS')
        diag_parts.append('NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS')
    elif 'NEW_FP' in transition:
        diag_parts.append('BUDGET_OVERSHOOT: PGD10 creates false positive that PGD3 avoids')
    elif 'CEILING' in transition:
        diag_parts.append('CEILING_ARTIFACT: delta-to-clean invalid; raw targeted_open_rate is primary metric')
    elif 'AGREE' in transition:
        diag_parts.append('ALIGNED: both PGD3 and PGD10 agree with VIS label')

    if source_batch in ('batch3', 'batch3b') and vis_pgd >= 40:
        diag_parts.append('NOTE: VIS uses PGD%d rollout; PGD10 no-env still 4x under budget' % vis_pgd)

    row['diagnosis'] = '; '.join(diag_parts)

# ── Write merged CSV ─────────────────────────────────────────────
DIA_CSV = os.path.join(REPO, 'tables', 'active_probe_v1_pgd_budget_diagnostic.csv')
all_keys = ['candidate_id','task_key','state_id','window_start','window_end',
            'disagreement_type','vis_label','vis_open_count','vis_pgd_steps',
            'taxonomy','ceiling_flag','n_frames',
            'clean_open_count','random_open_count',
            'targeted_open_count_pgd3','targeted_open_count_pgd10',
            'targeted_longest_streak_pgd3','targeted_longest_streak_pgd10',
            'targeted_minus_clean_pgd3','targeted_minus_clean_pgd10',
            'targeted_minus_random_pgd3','targeted_minus_random_pgd10',
            'p3_mc','p10_mc','p3_mr','p10_mr',
            'f31_pgd3_tmc_ref',
            'pgd3_decision','pgd10_decision','status_transition','diagnosis']
with open(DIA_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
print('Wrote %s' % DIA_CSV)

# ── Aggregate analysis ───────────────────────────────────────────
from collections import Counter

trans_counts = Counter(r['status_transition'] for r in rows)
repaired = sum(1 for r in rows if 'REPAIRED' in r['status_transition'])
persistent = sum(1 for r in rows if 'PERSISTENT' in r['status_transition'])
agree = sum(1 for r in rows if 'AGREE' in r['status_transition'])
ceiling = sum(1 for r in rows if 'CEILING' in r['status_transition'])

print('\n=== Transition Summary ===')
for t, c in trans_counts.most_common():
    print('  %s: %d' % (t, c))
print('Repaired: %d, Persistent: %d, Agree: %d, Ceiling: %d' % (repaired, persistent, agree, ceiling))

# ── Decision logic ───────────────────────────────────────────────
lines = []
lines.append('# Active Probe V1 — PGD Budget Diagnostic Report')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Rows**: 11 disagreement windows')
lines.append('**Method**: PGD3 vs PGD10, same 3 frames/window, no-env decode')
lines.append('**Comparison**: PGD3_3frame vs PGD10_3frame (SAME frames, NOT cross-sampling)')
lines.append('**VIS reference**: PGD20 (batch1) or PGD40x3 (batch3), WITH env.step')
lines.append('**PGD20**: SKIPPED (prohibitively slow on 7B model)')
lines.append('')

# Per-row table
lines.append('## Per-Row Diagnostic Results')
lines.append('')
lines.append('| Window | Type | VIS | Clean3f | PGD3_3f | PGD10_3f | t-c PGD3 | t-c PGD10 | Transition | Diagnosis |')
lines.append('|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(rows, key=lambda r: -r['p10_mc']):
    lines.append('| %s | %s | %s | %s | %s | %s | %+d | %+d | %s | %s |' % (
        r['candidate_id'], r['disagreement_type'][:12], r['vis_open_count'],
        r['clean_open_count'], r['targeted_open_count_pgd3'], r['targeted_open_count_pgd10'],
        r['p3_mc'], r['p10_mc'],
        r['status_transition'][:30], r['diagnosis'][:50]))
lines.append('')

# Transition detail
lines.append('## Transition Breakdown')
lines.append('')
for t, c in trans_counts.most_common():
    lines.append('### %s (n=%d)' % (t, c))
    for r in rows:
        if r['status_transition'] == t:
            lines.append('- **%s**: PGD3 t-c=%+d streak=%s, PGD10 t-c=%+d streak=%s, VIS open=%s | %s' % (
                r['candidate_id'],
                r['p3_mc'], r['targeted_longest_streak_pgd3'],
                r['p10_mc'], r['targeted_longest_streak_pgd10'],
                r['vis_open_count'], r['diagnosis']))
    lines.append('')

# Budget effect vs sampling effect
lines.append('## Budget Effect vs Sampling Effect')
lines.append('')
lines.append('Same 3 frames used for PGD3_3f and PGD10_3f. No cross-sampling comparison needed.')
lines.append('')
for r in rows:
    p10_vs_p3 = r['p10_mc'] - r['p3_mc']
    direction = 'MORE open' if p10_vs_p3 > 0 else ('LESS open' if p10_vs_p3 < 0 else 'SAME')
    lines.append('- **%s**: PGD3 t-c=%+d → PGD10 t-c=%+d (delta=%+d, %s)' % (
        r['candidate_id'], r['p3_mc'], r['p10_mc'], p10_vs_p3, direction))
lines.append('')

# Decision logic
lines.append('## Decision Logic')
lines.append('')
lines.append('| Criteria | Count | Verdict |')
lines.append('|---|---|---|')
lines.append('| Repaired (>=6) | %d | %s |' % (repaired, 'PGD3_UNDERPOWERED_LIKELY' if repaired >= 6 else '—'))
lines.append('| Mixed (2-5) | %d | %s |' % (repaired, 'MIXED_BUDGET_AND_SURROGATE' if 2 <= repaired <= 5 else '—'))
lines.append('| No repair (<=1) | %d | %s |' % (repaired, 'NO_ENV_SURROGATE_UNRELIABLE_LIKELY' if repaired <= 1 else '—'))
lines.append('')

# Verdict
if repaired >= 6:
    verdict = 'PGD3_UNDERPOWERED_LIKELY'
    verdict_text = 'Higher PGD budget repairs most disagreements. PGD3 is too weak. PGD10 (or higher) no-env probe may serve as a practical surrogate, but cost must be estimated.'
    next_steps = '1. Estimate PGD10 cost per window for online deployment. 2. If acceptable, run PGD10 full31 to confirm. 3. Do NOT train detector yet.'
elif repaired >= 2:
    verdict = 'MIXED_BUDGET_AND_SURROGATE'
    verdict_text = 'PGD10 repairs some but not most disagreements. Both budget AND surrogate mismatch contribute. Run PGD20 spot check on 4 representative rows.'
    next_steps = '1. Select 2 persistent FP rows and 2 persistent FN rows. 2. Run PGD20 no-env on those 4 rows. 3. If PGD20 repairs them: budget is the main issue, PGD10 is insufficient. 4. If PGD20 does NOT repair: no-env surrogate is fundamentally unreliable.'
else:
    verdict = 'NO_ENV_SURROGATE_UNRELIABLE_LIKELY'
    verdict_text = 'PGD10 does not repair the disagreements. The no-env decode paradigm is likely not a reliable surrogate for rollout VIS attack outcomes, regardless of PGD budget.'
    next_steps = '1. Stop Active Probe as VIS-label predictor. 2. Redirect to rollout-aware short-horizon probe or direct VIS empirical search. 3. Do NOT train detector.'

lines.append('## Verdict: %s' % verdict)
lines.append('')
lines.append(verdict_text)
lines.append('')
lines.append('### Next Steps')
lines.append('')
lines.append(next_steps)
lines.append('')

REPORT_PATH = os.path.join(REPO, 'reports', 'ACTIVE_PROBE_V1_PGD_BUDGET_DIAGNOSTIC.md')
with open(REPORT_PATH, 'w') as f:
    f.write('\n'.join(lines))
print('\nWrote report: %s' % REPORT_PATH)
print('Verdict: %s' % verdict)
