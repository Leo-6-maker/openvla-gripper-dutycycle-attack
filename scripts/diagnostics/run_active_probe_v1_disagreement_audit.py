#!/usr/bin/env python3
"""Disagreement audit: compare probe v1 PGD3 results against VIS PGD40 labels."""
import csv, os, sys
import numpy as np

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
OUT_DIR = '/data/liuyu/outputs/active_probe_v1_temporal_20260607'

# ── Load data ────────────────────────────────────────────────────
with open(os.path.join(REPO, 'tables', 'active_probe_v1_disagreement_review_queue.csv')) as f:
    queue = list(csv.DictReader(f))

with open(os.path.join(SHARED, 'object_phase_response_labels_v2.csv')) as f:
    labels = { (r['task_key'].strip(), r['state_id'].strip(),
                int(r['window_start']), int(r['window_end'])): r
              for r in csv.DictReader(f) }

with open(os.path.join(OUT_DIR, 'window_features_full31_merged.csv')) as f:
    features = { (r['task_key'], r['state_id'],
                  int(r['window_start']), int(r['window_end'])): r
                for r in csv.DictReader(f) }

# ── Known VIS configs by batch ───────────────────────────────────
# From trace logs and calibration configs:
# batch1: PGD20, eps=6/255, restarts=1, objective=prefix_locked_gripper_open_margin, env.step=YES
# batch3: PGD40, eps=6/255, restarts=3, objective=prefix_locked_gripper_open_margin, env.step=YES
# batch3b: PGD40, eps=6/255, restarts=3, objective=prefix_locked_gripper_open_margin, env.step=YES (1R)

VIS_CONFIGS = {
    'batch1': {'pgd_steps': 20, 'restarts': 1, 'eps_raw': 6, 'objective': 'prefix_locked_gripper_open_margin', 'env_step': True},
    'batch3': {'pgd_steps': 40, 'restarts': 3, 'eps_raw': 6, 'objective': 'prefix_locked_gripper_open_margin', 'env_step': True},
    'batch3b': {'pgd_steps': 40, 'restarts': 3, 'eps_raw': 6, 'objective': 'prefix_locked_gripper_open_margin', 'env_step': True},
}

PROBE_CONFIG = {'pgd_steps': 3, 'restarts': 1, 'eps_raw': 6, 'objective': 'prefix_locked_gripper_open_margin', 'env_step': False}

# ── Build audit rows ─────────────────────────────────────────────
audit_rows = []
for q in queue:
    tk = q['task_key']; sid = q['state_id']; ws = q['window_start']; we = q['window_end']
    key = (tk, sid, int(ws), int(we))
    lbl = labels.get(key, {})
    feat = features.get(key, {})

    batch = lbl.get('source_batch', '?').strip()
    vis_cfg = VIS_CONFIGS.get(batch, {})

    # Probe metrics
    t_open = int(feat.get('targeted_open_count', 0))
    c_open = int(feat.get('clean_open_count', 0))
    r_open = int(feat.get('random_open_count', 0))
    n_frames = int(feat.get('n_probe_frames', 10))
    t_rate = round(t_open / max(n_frames, 1), 3)
    c_rate = round(c_open / max(n_frames, 1), 3)
    r_rate = round(r_open / max(n_frames, 1), 3)
    tmc = t_open - c_open
    tmr = int(feat.get('targeted_minus_random_open_count', 0))
    t_streak = int(feat.get('targeted_longest_open_streak', 0))
    c_streak = int(feat.get('clean_longest_open_streak', 0))

    # VIS metrics
    vis_open_count = int(lbl.get('vis_open_count', 0) or 0)
    vis_total_frames = int((lbl.get('VIS_OPEN', '') or '').split('/')[1]) if '/' in str(lbl.get('VIS_OPEN', '')) else 18
    qpos_delta = float(lbl.get('qpos_opening_delta', 0) or 0)
    qpos_label = lbl.get('qpos_label', '?').strip()
    phys_resp = float(lbl.get('label_physical_response', 0) or 0)
    taxonomy = lbl.get('taxonomy', '?').strip()
    label_status = lbl.get('label_status', '?').strip()
    provenance = lbl.get('provenance_status', '?').strip()

    # ── Diagnosis ────────────────────────────────────────────────
    dt = q.get('disagreement_type', '?').strip()
    diagnoses = []

    # Check PGD budget mismatch
    probe_budget = PROBE_CONFIG['pgd_steps'] * PROBE_CONFIG['restarts']  # 3
    vis_budget = vis_cfg.get('pgd_steps', 20) * vis_cfg.get('restarts', 1)  # 20 or 120
    budget_ratio = vis_budget / max(probe_budget, 1)
    if budget_ratio >= 5:
        diagnoses.append('probe_surrogate_mismatch: PGD%d probe vs PGD%d VIS (%.0fx budget gap)' % (
            probe_budget, vis_budget, budget_ratio))

    # Check env.step difference
    if not PROBE_CONFIG['env_step']:
        diagnoses.append('probe_surrogate_mismatch: no-env decode vs env.step rollout')

    # Check ceiling
    if q.get('disagreement_type', '') == 'CEILING_POSITIVE':
        diagnoses.append('ceiling_artifact: clean model already OPEN in %d/%d frames' % (c_open, n_frames))

    # Check if probe finds signal where VIS doesn't
    if dt == 'HIGH_PROBE_NEGATIVE_LABEL':
        if tmc >= 4 and vis_open_count == 0:
            if 'polluted' in taxonomy.lower():
                diagnoses.append('label_noise_suspected: VIS trace labeled polluted, probe suggests real susceptibility')
            elif 'no_action_bridge' in taxonomy.lower():
                diagnoses.append('label_noise_suspected: VIS label says no action bridge, but probe induces OPEN')
            else:
                diagnoses.append('probe_surrogate_mismatch: probe PGD3 induce OPEN but VIS PGD%d did not' % vis_budget)

    # Check if VIS finds signal where probe doesn't
    if dt == 'POSITIVE_LABEL_LOW_PROBE':
        if vis_open_count >= 6 and tmc <= 0:
            diagnoses.append('probe_surrogate_mismatch: PGD3 too weak; VIS PGD%d succeeded with %d/%d open' % (
                vis_budget, vis_open_count, vis_total_frames))
            if budget_ratio >= 10:
                diagnoses.append('probe_surrogate_mismatch: PGD budget gap %.0fx — PGD3 severely underpowered vs PGD%d' % (
                    budget_ratio, vis_budget))

    # Window alignment
    window_len = int(we) - int(ws) + 1
    if n_frames < window_len // 2:
        diagnoses.append('window_alignment_suspected: probe sampled %d frames in %d-step window' % (n_frames, window_len))

    # Clean baseline ceiling for positives
    if c_rate >= 0.7 and dt == 'POSITIVE_LABEL_LOW_PROBE':
        diagnoses.append('ceiling_artifact: clean baseline already at %.0f%% open rate' % (c_rate * 100))

    if not diagnoses:
        diagnoses.append('metric_artifact: check scoring')

    audit_rows.append({
        'candidate_id': '%s_s%s_w%s_%s' % (tk, sid, ws, we),
        'task_key': tk, 'state_id': sid,
        'window_start': ws, 'window_end': we,
        'disagreement_type': dt,
        'label_status': label_status, 'taxonomy': taxonomy,
        'provenance': provenance,
        'source_batch': batch,
        # Probe metrics
        'probe_pgd_steps': str(PROBE_CONFIG['pgd_steps']),
        'probe_eps_raw': str(PROBE_CONFIG['eps_raw']),
        'probe_objective': PROBE_CONFIG['objective'],
        'probe_env_step': str(PROBE_CONFIG['env_step']),
        'probe_n_frames': str(n_frames),
        'clean_open_count': str(c_open), 'clean_open_rate': str(c_rate),
        'targeted_open_count': str(t_open), 'targeted_open_rate': str(t_rate),
        'random_open_count': str(r_open), 'random_open_rate': str(r_rate),
        'tmc_count': str(tmc), 'tmr_count': str(tmr),
        'probe_longest_open_streak': str(t_streak),
        'clean_longest_open_streak': str(c_streak),
        # VIS metrics
        'vis_pgd_steps': str(vis_cfg.get('pgd_steps', '?')),
        'vis_restarts': str(vis_cfg.get('restarts', '?')),
        'vis_eps_raw': str(vis_cfg.get('eps_raw', '?')),
        'vis_objective': vis_cfg.get('objective', '?'),
        'vis_env_step': str(vis_cfg.get('env_step', '?')),
        'vis_open_count': str(vis_open_count),
        'vis_total_frames': str(vis_total_frames),
        'qpos_delta': str(round(qpos_delta, 6)),
        'qpos_label': qpos_label,
        'phys_resp': str(phys_resp),
        # Conventions
        'probe_open_convention': 'decoded_gripper_+1_OPEN_after_normalize_invert',
        'vis_open_convention': 'decoded_gripper_+1_OPEN_after_normalize_invert',
        'window_alignment': 'same [ws,we]',
        # Diagnosis
        'diagnosis': '; '.join(diagnoses),
    })

# ── Write audit CSV ──────────────────────────────────────────────
AUDIT_CSV = os.path.join(REPO, 'tables', 'active_probe_v1_disagreement_audit.csv')
keys = list(audit_rows[0].keys())
with open(AUDIT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader(); w.writerows(audit_rows)
print('Wrote %d rows to %s' % (len(audit_rows), AUDIT_CSV))

# ── Report ───────────────────────────────────────────────────────
lines = []
lines.append('# Active Probe V1 — Disagreement Audit')
lines.append('')
lines.append('**Date**: 2026-06-07')
lines.append('**Disagreements**: %d from full31' % len(audit_rows))
lines.append('')
lines.append('## Config Comparison')
lines.append('')
lines.append('| Aspect | Probe v1 | VIS Label |')
lines.append('|---|---|---|')
lines.append('| PGD steps | 3 | 20 (batch1) / 40 (batch3) |')
lines.append('| Restarts | 1 | 1 (batch1) / 3 (batch3) |')
lines.append('| Effective budget | 3 | 20-120 |')
lines.append('| eps_raw | 6 | 6 |')
lines.append('| Objective | prefix_locked_gripper_open_margin | same |')
lines.append('| env.step | NO | YES |')
lines.append('| Open convention | decoded +1=OPEN | same |')
lines.append('| Frames sampled | ~10/window | 18/window (full attack) |')
lines.append('')

# Summary by type
from collections import Counter
type_counts = Counter(r['disagreement_type'] for r in audit_rows)
lines.append('## By Disagreement Type')
lines.append('')
for t, c in type_counts.most_common():
    lines.append('- **%s**: %d' % (t, c))
lines.append('')

# Diagnosis summary
diag_all = []
for r in audit_rows:
    diag_all.extend(r['diagnosis'].split('; '))
diag_counts = Counter(diag_all)
lines.append('## Diagnosis Summary')
lines.append('')
for d, c in diag_counts.most_common():
    cat = d.split(':')[0]
    lines.append('- %s (x%d): %s' % (cat, c, d.split(': ', 1)[1] if ': ' in d else d))
lines.append('')

# Per-row detail
lines.append('## Per-Disagreement Detail')
lines.append('')
for r in sorted(audit_rows, key=lambda r: -int(r['tmc_count'])):
    lines.append('### %s' % r['candidate_id'])
    lines.append('')
    lines.append('- **Type**: %s | label=%s taxonomy=%s batch=%s' % (
        r['disagreement_type'], r['label_status'], r['taxonomy'], r['source_batch']))
    lines.append('- **Probe**: clean=%s/%s (rate=%s), targeted=%s/%s (rate=%s), random=%s/%s (rate=%s)' % (
        r['clean_open_count'], r['probe_n_frames'], r['clean_open_rate'],
        r['targeted_open_count'], r['probe_n_frames'], r['targeted_open_rate'],
        r['random_open_count'], r['probe_n_frames'], r['random_open_rate']))
    lines.append('- **Probe deltas**: t-c=%s, t-r=%s, streak=%s (clean streak=%s)' % (
        r['tmc_count'], r['tmr_count'], r['probe_longest_open_streak'], r['clean_longest_open_streak']))
    lines.append('- **VIS**: open=%s/%s, qpos=%s, phys=%s, PGD budget=%s' % (
        r['vis_open_count'], r['vis_total_frames'], r['qpos_delta'], r['phys_resp'],
        r['vis_pgd_steps']))
    lines.append('- **Diagnosis**: %s' % r['diagnosis'])
    lines.append('')

# Root cause conclusion
lines.append('## Root Cause Assessment')
lines.append('')
pgd_mismatch = sum(1 for r in audit_rows if 'probe_surrogate_mismatch' in r['diagnosis'])
label_noise = sum(1 for r in audit_rows if 'label_noise_suspected' in r['diagnosis'])
ceiling = sum(1 for r in audit_rows if 'ceiling_artifact' in r['diagnosis'])

lines.append('### Primary: PGD Budget Gap (PGD3 vs PGD20-120)')
lines.append('')
lines.append('The probe uses PGD3 (3 gradient steps, no restarts) while VIS labels come from')
lines.append('PGD20 (batch1) or PGD40x3 (batch3, effective budget 120). The 7-40x budget gap means:')
lines.append('')
lines.append('- PGD3 may fail to find perturbations that PGD20/40 finds (false negatives in probe)')
lines.append('- PGD3 may find perturbations that PGD20/40 avoids (different local minima behavior)')
lines.append('- No-env decode vs env.step further amplifies the discrepancy')
lines.append('')

lines.append('### Secondary: Label Noise / Contamination')
lines.append('')
lines.append('Several "negative" or "polluted" windows show strong probe signal. The probe is not')
lines.append('necessarily wrong — the VIS trace may have been contaminated or the PGD budget/restart')
lines.append('may have missed a valid attack direction.')
lines.append('')

lines.append('### Tertiary: Ceiling Effect')
lines.append('')
lines.append('%d windows have clean model already commanding OPEN. Delta-to-clean is invalid.' % ceiling)
lines.append('')
lines.append('## Recommendation')
lines.append('')
lines.append('1. **Do NOT use PGD3 no-env as a surrogate for PGD20+ env.step VIS attack.**')
lines.append('   The budget gap is too large for reliable proxy.')
lines.append('2. **If a cheap probe is needed, run PGD10 no-env on the 11 disagreement rows**')
lines.append('   to check if higher PGD budget closes the gap.')
lines.append('3. **If PGD20 no-env still disagrees, the no-env probe is fundamentally not a')
lines.append('   reliable surrogate for rollout VIS** — the env.step / temporal dynamics matter.')
lines.append('4. **Audit the 4 HIGH_PROBE_NEGATIVE_LABEL windows** with fresh VIS to rule out')
lines.append('   trace contamination vs genuine probe false positive.')
lines.append('')

REPORT_PATH = os.path.join(REPO, 'reports', 'ACTIVE_PROBE_V1_DISAGREEMENT_AUDIT.md')
with open(REPORT_PATH, 'w') as f:
    f.write('\n'.join(lines))
print('Wrote report to %s' % REPORT_PATH)
print('DONE')
