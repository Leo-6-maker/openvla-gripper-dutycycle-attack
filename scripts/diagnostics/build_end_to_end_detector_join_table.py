#!/usr/bin/env python3
"""Build unified end-to-end detector join table.
Joins: labels_v2 + mechanism taxonomy + detector v27 predictions +
        ProprioNoStep phase scores + clean controls + calibration +
        adaptive 1R screening.

Output:
  tables/end_to_end_detector_join_table.csv
  reports/END_TO_END_DETECTOR_JOIN_AUDIT.md
"""

import csv, os, sys
from collections import Counter, defaultdict
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────
REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
SHARED = '/data/liuyu/outputs/shared_detector_v25_inputs_20260606'
CODEX = '/data/liuyu/outputs/codex_phase_detector_twostage_20260606/tables'
PROPRIO = '/data/liuyu/outputs/proprionostep_cpu_20260602'
OUT_CSV = os.path.join(REPO, 'tables/end_to_end_detector_join_table.csv')
OUT_MD = os.path.join(REPO, 'reports/END_TO_END_DETECTOR_JOIN_AUDIT.md')

def read_csv(path):
    if not os.path.exists(path):
        print(f'  MISSING: {path}')
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def make_key(r, tk='task_key', sid='state_id', ws='window_start', we='window_end'):
    return (str(r.get(tk,'')).strip(),
            str(r.get(sid,'')).strip(),
            str(r.get(ws,'')).strip(),
            str(r.get(we,'')).strip())

def safe_float(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

# ── Load all sources ───────────────────────────────────────────────
print('=== Loading data sources ===')

# 1. Labels v2 (gold standard)
labels_v2 = read_csv(os.path.join(SHARED, 'object_phase_response_labels_v2.csv'))
labels_by_key = {make_key(r): r for r in labels_v2}
print(f'Labels v2: {len(labels_v2)} rows')

# 2. Mechanism taxonomy audit
mech_tax = read_csv(os.path.join(REPO, 'tables/vulnerability_mechanism_taxonomy_audit.csv'))
mech_by_key = {make_key(r): r for r in mech_tax}
print(f'Mechanism taxonomy: {len(mech_tax)} rows')

# 3. Detector v27 predictions (all variants)
det_preds = read_csv(os.path.join(CODEX, 'detector_v27_phase_aware_predictions.csv'))
print(f'Detector v27 predictions: {len(det_preds)} rows')

# 4. Detector v27 dataset (features + labels)
det_dataset = read_csv(os.path.join(CODEX, 'detector_v27_phase_aware_dataset.csv'))
det_dataset_by_key = {make_key(r): r for r in det_dataset}
print(f'Detector v27 dataset: {len(det_dataset)} rows')

# 5. ProprioNoStep phase scores (from covered subset)
phase_joined = read_csv(os.path.join(REPO, 'tables/real_phase_vuln_joined_covered_subset.csv'))
phase_by_key = {}
for r in phase_joined:
    key = make_key(r)
    phase_by_key[key] = r
print(f'Phase scores (covered subset): {len(phase_joined)} rows')

# 6. ProprioNoStep raw checkpoint scores (step-level)
phase_checkpoint = read_csv(os.path.join(PROPRIO, 'checkpoint_scores_clean_traces.csv'))
# Aggregate step-level to window-level
# checkpoint has: task, step, window_start, window_end, hazard_score, release_safe_score
# We'll compute: hazard_score_mean/max, release_safe_score_mean/min per (task, window_start, window_end)
print(f'Phase checkpoint scores (step-level): {len(phase_checkpoint)} rows')

# 7. Clean control bank v2
clean_ctrl = read_csv(os.path.join(CODEX, 'clean_control_negative_bank_v2.csv'))
clean_ctrl_by_key = {make_key(r): r for r in clean_ctrl}
print(f'Clean control bank: {len(clean_ctrl)} rows')

# 8. Calibration summaries
calib_1r = read_csv(os.path.join(REPO, 'tables/calib_1r_summary.csv'))
calib_3r = read_csv(os.path.join(REPO, 'tables/calib_3r_summary.csv'))
print(f'Calibration: 1R={len(calib_1r)} rows, 3R={len(calib_3r)} rows')

# 9. Adaptive 1R screening summary
adapt_summary = read_csv(os.path.join(SHARED, 'adaptive_vis_1r_screening_summary.csv'))
adapt_by_key = {}
for r in adapt_summary:
    key = make_key(r, tk='task_key')
    adapt_by_key[key] = r
print(f'Adaptive 1R screening summary: {len(adapt_summary)} rows')

# 10. Calibration v2 candidates
calib_v2 = read_csv(os.path.join(REPO, 'tables/vis_1r_vs_3r_calibration_v2_candidates.csv'))
calib_v2_by_key = {make_key(r): r for r in calib_v2}
print(f'Calibration v2 candidates: {len(calib_v2)} rows')

# ── Aggregate ProprioNoStep step scores to window-level ───────────
print('\n=== Aggregating ProprioNoStep step scores ===')
phase_agg = defaultdict(lambda: {'hazard_vals': [], 'release_safe_vals': []})
for r in phase_checkpoint:
    tk = str(r.get('task','')).strip()
    ws = str(r.get('window_start','')).strip()
    we = str(r.get('window_end','')).strip()
    # We don't have state_id in checkpoint scores; use task+window
    key = (tk, ws, we)
    hs = safe_float(r.get('hazard_score'))
    rs = safe_float(r.get('release_safe_score'))
    if hs is not None:
        phase_agg[key]['hazard_vals'].append(hs)
    if rs is not None:
        phase_agg[key]['release_safe_vals'].append(rs)

# Build task→(ws,we)→agg mapping
phase_agg_by_task_window = {}
for (tk, ws, we), vals in phase_agg.items():
    if tk not in phase_agg_by_task_window:
        phase_agg_by_task_window[tk] = {}
    h = vals['hazard_vals']
    r = vals['release_safe_vals']
    phase_agg_by_task_window[tk][(ws, we)] = {
        'hazard_score_mean': sum(h)/len(h) if h else None,
        'hazard_score_max': max(h) if h else None,
        'release_safe_score_mean': sum(r)/len(r) if r else None,
        'release_safe_score_min': min(r) if r else None,
    }
print(f'  Aggregated to {sum(len(v) for v in phase_agg_by_task_window.values())} task-window entries')

# ── Build best detector prediction per (task, state, ws, we) ──────
# Predictions have many variants; pick V3_weighted_LR as canonical
print('\n=== Selecting canonical detector predictions ===')
CANONICAL_VARIANT = 'V3_weighted'
CANONICAL_MODEL = 'LR'
CANONICAL_FEATURE = 'D_causal_safe_fastvis'

# First check what combinations exist
pred_variants = Counter()
for p in det_preds:
    pred_variants[(p.get('variant','?'), p.get('model','?'), p.get('feature_set','?'))] += 1
print('  Prediction variants (top 10):')
for (v, m, f), c in pred_variants.most_common(10):
    print(f'    {v} / {m} / {f}: {c}')

# Canonical: V0_gold_only / LR / D_causal_safe (gold-only, causal features, LR)
CANONICAL_VARIANT = 'V0_gold_only'
CANONICAL_MODEL = 'LR'
CANONICAL_FEATURE = 'D_causal_safe'

det_best_by_key = {}
# First pass: exact canonical match
for p in det_preds:
    if (p.get('variant','') == CANONICAL_VARIANT and
        p.get('model','') == CANONICAL_MODEL and
        p.get('feature_set','') == CANONICAL_FEATURE):
        key = make_key(p)
        det_best_by_key[key] = p

# Second pass (fallback): V0_gold_only LR H_descriptor_upper
if len(det_best_by_key) < 22:
    for p in det_preds:
        key = make_key(p)
        if key not in det_best_by_key:
            if (p.get('variant','') == CANONICAL_VARIANT and
                p.get('model','') == CANONICAL_MODEL and
                p.get('feature_set','') == 'H_descriptor_upper'):
                det_best_by_key[key] = p

print(f'  Canonical predictions ({CANONICAL_VARIANT}/{CANONICAL_MODEL}/{CANONICAL_FEATURE}): {len(det_best_by_key)} unique keys')

# ── Build unified join table ──────────────────────────────────────
print('\n=== Building unified join table ===')

# Collect all unique keys from all sources
all_keys = set()
all_keys.update(labels_by_key.keys())
all_keys.update(mech_by_key.keys())
all_keys.update(det_best_by_key.keys())
all_keys.update(phase_by_key.keys())
all_keys.update(clean_ctrl_by_key.keys())
all_keys.update(adapt_by_key.keys())
# Also add detector dataset keys
all_keys.update(det_dataset_by_key.keys())

print(f'  Total unique keys across all sources: {len(all_keys)}')

# Define output columns
OUT_COLS = [
    # Identity
    'task_key', 'state_id', 'window_start', 'window_end',
    # Label status (from labels_v2 or mechanism taxonomy)
    'label_vulnerability_ready', 'label_source', 'label_status',
    'train_use', 'taxonomy', 'mechanism_type',
    # Vulnerability detector
    'vulnerability_score', 'vulnerability_pred', 'detector_variant',
    'detector_model', 'detector_feature_set', 'vuln_score_available',
    # Phase detector (from covered subset)
    'predicted_phase', 'phase_confidence', 'phase_is_critical',
    'hazard_score_mean', 'hazard_score_max',
    'release_safe_score_mean', 'release_safe_score_min',
    'phase_available', 'phase_source',
    # Mechanism evidence
    'qpos_delta', 'vis_open_count', 'task_failure',
    'qpos_label', 'action_bridge_status', 'denominator_status',
    # Provenance
    'source_batch', 'provenance_status', 'provenance_note',
    # Join status
    'join_status', 'missing_reason',
    # Adaptive 1R
    'label_1r', 'adaptive_status',
    # Calibration
    'in_calibration_v2',
]

rows = []
for key in sorted(all_keys):
    tk, sid, ws, we = key
    row = {c: '' for c in OUT_COLS}
    row['task_key'] = tk
    row['state_id'] = sid
    row['window_start'] = ws
    row['window_end'] = we

    available = []
    missing = []

    # ── Labels v2 ──
    lv2 = labels_by_key.get(key)
    if lv2:
        row['label_vulnerability_ready'] = lv2.get('label_vulnerability_ready','')
        row['label_source'] = lv2.get('label_source','gold_v2')
        row['label_status'] = lv2.get('label_status','')
        row['train_use'] = lv2.get('label_use','')
        row['taxonomy'] = lv2.get('taxonomy','')
        row['qpos_delta'] = lv2.get('qpos_opening_delta','')
        row['vis_open_count'] = lv2.get('vis_open_count','')
        row['task_failure'] = lv2.get('done','')
        row['qpos_label'] = lv2.get('qpos_label','')
        row['action_bridge_status'] = lv2.get('label_action_bridge','')
        row['denominator_status'] = lv2.get('denominator_clean','')
        row['source_batch'] = lv2.get('source_batch','')
        row['provenance_status'] = lv2.get('provenance_status','')
        row['provenance_note'] = lv2.get('provenance_note','')
        available.append('labels_v2')
    else:
        missing.append('labels_v2')

    # ── Mechanism taxonomy ──
    mt = mech_by_key.get(key)
    if mt:
        row['mechanism_type'] = mt.get('mechanism_type','')
        if not row['taxonomy']:
            row['taxonomy'] = mt.get('taxonomy','')
        if not row['label_vulnerability_ready']:
            row['label_vulnerability_ready'] = mt.get('label_vulnerability_ready','')
        if not row['label_source']:
            row['label_source'] = mt.get('label_source','')
        if not row['train_use']:
            row['train_use'] = mt.get('train_use','')
        available.append('mechanism')
    else:
        missing.append('mechanism')

    # ── Vulnerability detector ──
    det = det_best_by_key.get(key)
    if det:
        row['vulnerability_score'] = det.get('pred','')
        row['vulnerability_pred'] = det.get('pred','')
        row['detector_variant'] = det.get('variant','')
        row['detector_model'] = det.get('model','')
        row['detector_feature_set'] = det.get('feature_set','')
        row['vuln_score_available'] = '1'
        available.append('vuln_detector')
    else:
        missing.append('vuln_detector')

    # ── Phase detector (covered subset) ──
    ph = phase_by_key.get(key)
    if ph:
        row['predicted_phase'] = ph.get('phase_bin_proxy','')
        row['hazard_score_mean'] = ph.get('hazard_score_mean','')
        row['hazard_score_max'] = ph.get('hazard_score_max','')
        row['release_safe_score_mean'] = ph.get('release_safe_score_mean','')
        row['release_safe_score_min'] = ph.get('release_safe_score_min','')
        row['phase_available'] = '1'
        row['phase_source'] = 'covered_subset'
        available.append('phase')
    else:
        # Try to find phase from checkpoint aggregation
        if tk in phase_agg_by_task_window:
            pw = phase_agg_by_task_window[tk].get((ws, we))
            if pw:
                row['hazard_score_mean'] = pw['hazard_score_mean'] or ''
                row['hazard_score_max'] = pw['hazard_score_max'] or ''
                row['release_safe_score_mean'] = pw['release_safe_score_mean'] or ''
                row['release_safe_score_min'] = pw['release_safe_score_min'] or ''
                row['phase_available'] = '1'
                row['phase_source'] = 'checkpoint_agg'
                available.append('phase_checkpoint')
            else:
                missing.append('phase')
        else:
            missing.append('phase')

    # ── Clean control ──
    cc = clean_ctrl_by_key.get(key)
    if cc:
        if not row['label_source']:
            row['label_source'] = cc.get('label_source','')
        if not row['label_status']:
            row['label_status'] = cc.get('label_status','')
        if not row['train_use']:
            row['train_use'] = 'ablation_only'
        if not row['phase_is_critical']:
            row['phase_is_critical'] = cc.get('phase_is_critical','')
        available.append('clean_control')
    else:
        missing.append('clean_control')

    # ── Adaptive 1R ──
    ad = adapt_by_key.get(key)
    if ad:
        row['label_1r'] = ad.get('label_1r','')
        row['adaptive_status'] = ad.get('status','')
        available.append('adaptive_1r')
    else:
        missing.append('adaptive_1r')

    # ── Detector dataset ──
    dd = det_dataset_by_key.get(key)
    if dd:
        if not row['predicted_phase']:
            row['predicted_phase'] = dd.get('predicted_phase','')
        if not row['phase_confidence']:
            row['phase_confidence'] = dd.get('phase_confidence','')
        if not row['phase_is_critical']:
            row['phase_is_critical'] = dd.get('phase_is_critical','')
        available.append('det_dataset')
    else:
        missing.append('det_dataset')

    # ── Calibration v2 ──
    if calib_v2_by_key.get(key):
        row['in_calibration_v2'] = '1'
        available.append('calib_v2')

    # ── Join status ──
    has_phase = 'phase' in available or 'phase_checkpoint' in available
    has_vuln = 'vuln_detector' in available
    has_mech = 'mechanism' in available
    has_label = 'labels_v2' in available

    if has_phase and has_vuln and has_mech:
        row['join_status'] = 'fully_joined'
    elif has_vuln and has_mech:
        row['join_status'] = 'vuln_mech_only_no_phase'
    elif has_vuln:
        row['join_status'] = 'vuln_only'
    elif has_label:
        row['join_status'] = 'label_only'
    else:
        row['join_status'] = 'orphan'

    row['missing_reason'] = ','.join(missing) if missing else 'none'

    rows.append(row)

# ── Write CSV ─────────────────────────────────────────────────────
fieldnames = OUT_COLS
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f'\nWrote {len(rows)} rows to {OUT_CSV}')

# ── Generate audit report ─────────────────────────────────────────
print('\n=== Generating audit report ===')

# Coverage stats
fully_joined = [r for r in rows if r['join_status'] == 'fully_joined']
vuln_mech = [r for r in rows if r['join_status'] == 'vuln_mech_only_no_phase']
vuln_only = [r for r in rows if r['join_status'] == 'vuln_only']
label_only = [r for r in rows if r['join_status'] == 'label_only']
orphan = [r for r in rows if r['join_status'] == 'orphan']

# By source
all_keys = set(make_key(r) for r in rows)
vuln_keys = set(make_key(r) for r in rows if r['vuln_score_available'] == '1')
phase_keys = set(make_key(r) for r in rows if r['phase_available'] == '1')
mech_keys = set(make_key(r) for r in rows if r['mechanism_type'] != '')
label_keys = set(make_key(r) for r in rows if r['label_source'] != '')

# Mechanism type distribution in joined rows
mech_types = Counter(r['mechanism_type'] for r in rows if r['mechanism_type'])

# Label status distribution
label_statuses = Counter(r['label_status'] for r in rows if r['label_status'])

# By task
task_coverage = defaultdict(lambda: {'total': 0, 'phase': 0, 'vuln': 0, 'mech': 0, 'label': 0})
for r in rows:
    tk = r['task_key']
    task_coverage[tk]['total'] += 1
    if r['phase_available'] == '1': task_coverage[tk]['phase'] += 1
    if r['vuln_score_available'] == '1': task_coverage[tk]['vuln'] += 1
    if r['mechanism_type']: task_coverage[tk]['mech'] += 1
    if r['label_source']: task_coverage[tk]['label'] += 1

# Determine full-phase coverage: tasks with ProprioNoStep model
PHASE_TASKS = {'cream_cheese', 'salad_dressing', 'ketchup'}

with open(OUT_MD, 'w') as f:
    f.write(f"""# End-to-End Detector Join Audit
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total unique rows | {len(rows)} |
| Fully joined (phase+vuln+mech) | {len(fully_joined)} |
| Vuln+mech (no phase) | {len(vuln_mech)} |
| Vuln only | {len(vuln_only)} |
| Label only | {len(label_only)} |
| Orphan | {len(orphan)} |

## Coverage by Source

| Source | Rows | Keys Matched |
|--------|------|-------------|
| Labels v2 | {len(labels_v2)} | {len(label_keys)} |
| Mechanism Taxonomy | {len(mech_tax)} | {len(mech_keys)} |
| Detector v27 Predictions | {len(det_preds)} → {len(det_best_by_key)} unique | {len(vuln_keys)} |
| Phase (covered subset) | {len(phase_joined)} | {len(phase_keys)} |
| Clean Control Bank | {len(clean_ctrl)} | {len(set(make_key(r) for r in clean_ctrl))} |
| Adaptive 1R | {len(adapt_summary)} | {len(set(make_key(r, 'task_key') for r in adapt_summary))} |

## Join Status Distribution

| Status | Count |
|--------|-------|
| fully_joined | {len(fully_joined)} |
| vuln_mech_only_no_phase | {len(vuln_mech)} |
| vuln_only | {len(vuln_only)} |
| label_only | {len(label_only)} |
| orphan | {len(orphan)} |

## Mechanism Type Distribution (in joined table)

| Mechanism Type | Count |
|----------------|-------|
""")
    for mt, c in mech_types.most_common():
        f.write(f'| {mt} | {c} |\n')

    f.write(f"""
## Label Status Distribution

| Status | Count |
|--------|-------|
""")
    for ls, c in label_statuses.most_common():
        f.write(f'| {ls} | {c} |\n')

    f.write(f"""
## Coverage by Task

| Task | Total | Phase | Vuln | Mech | Label |
|------|-------|-------|------|------|-------|
""")
    for tk in sorted(task_coverage.keys()):
        tc = task_coverage[tk]
        f.write(f'| {tk} | {tc["total"]} | {tc["phase"]} | {tc["vuln"]} | {tc["mech"]} | {tc["label"]} |\n')

    f.write(f"""
## Phase Coverage Detail

ProprioNoStep model covers: {sorted(PHASE_TASKS)}

| Task | Has Phase Coverage |
|------|-------------------|
""")
    for tk in sorted(task_coverage.keys()):
        has_proprio = tk in PHASE_TASKS
        phase_count = task_coverage[tk]['phase']
        yes_no = "YES" if has_proprio else "NO"
        f.write(f'| {tk} | {yes_no} ({phase_count} rows with phase) |\n')

    # Which rows are fully joined
    f.write(f"""
## Fully Joined Rows ({len(fully_joined)})

| task | state | window | mechanism | vuln_pred | phase_available | phase_source |
|------|-------|--------|-----------|-----------|-----------------|-------------|
""")
    for r in fully_joined[:50]:
        f.write(f'| {r["task_key"]} | {r["state_id"]} | [{r["window_start"]},{r["window_end"]}] | {r["mechanism_type"]} | {r["vulnerability_pred"]} | {r["phase_available"]} | {r["phase_source"]} |\n')

    f.write(f"""
## Missing Links

### Rows missing phase (ProprioNoStep model covers cream_cheese/ketchup/salad_dressing only)

**Tasks with ProprioNoStep model but missing phase data:**
""")
    for r in rows:
        if r['task_key'] in PHASE_TASKS and r['phase_available'] != '1' and r['label_source']:
            f.write(f"- {r['task_key']} s{r['state_id']} [{r['window_start']},{r['window_end']}] — source={r['label_source']}\n")

    f.write(f"""
### Rows missing vuln prediction:
""")
    vuln_missing = [r for r in rows if r['vuln_score_available'] != '1']
    if vuln_missing:
        for r in vuln_missing[:20]:
            f.write(f"- {r['task_key']} s{r['state_id']} [{r['window_start']},{r['window_end']}] — source={r['label_source']}\n")
    else:
        f.write("None\n")

    f.write(f"""
### Rows missing mechanism type:
""")
    mech_missing = [r for r in rows if not r['mechanism_type']]
    if mech_missing:
        for r in mech_missing[:20]:
            f.write(f"- {r['task_key']} s{r['state_id']} [{r['window_start']},{r['window_end']}] — source={r['label_source']}\n")
    else:
        f.write("None\n")

    f.write(f"""
## Recommendations

1. **Phase detector coverage is limited to 3 tasks** (cream_cheese, ketchup, salad_dressing).
   - Only {sum(1 for r in rows if r['task_key'] in PHASE_TASKS)} rows belong to these tasks.
   - {sum(1 for r in rows if r['task_key'] in PHASE_TASKS and r['phase_available']=='1')} of those have phase scores available.
   - Phase detector CANNOT serve as a universal pipeline stage.

2. **Vulnerability detector coverage**: {len(vuln_keys)} unique keys have predictions.
   - Canonical predictions (V3_weighted LR): covers {sum(1 for p in det_preds if 'V3_weighted' in p.get('variant','') and p.get('model')=='LR')} rows.

3. **Mechanism coverage**: {len(mech_keys)} rows have mechanism types.
   - {mech_types.get('physical_bridge_positive', 0)} physical_bridge_positive
   - {mech_types.get('clean_control_negative', 0)} clean_control_negative
   - {mech_types.get('mechanism_unclear', 0)} mechanism_unclear

4. **Full pipeline feasibility**:
   - Only {len(fully_joined)} rows can run through the complete pipeline (phase+vuln+mech).
   - Phase detector is the bottleneck — only 3 tasks covered.
""")

print(f'Wrote audit to {OUT_MD}')
print('\nDone.')
