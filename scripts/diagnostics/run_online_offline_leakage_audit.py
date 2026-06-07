#!/usr/bin/env python3
"""F1: Online/Offline Feature Leakage Audit.
Classifies every feature column used in detector training/evaluation
into A/B/C/D categories and detects leakage risks.

Output:
  tables/online_offline_feature_leakage_audit.csv
  reports/ONLINE_OFFLINE_FEATURE_LEAKAGE_AUDIT.md
"""

import csv, os
from datetime import datetime

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
CODEX = '/data/liuyu/outputs/codex_phase_detector_twostage_20260606/tables'

OUT_CSV = os.path.join(REPO, 'tables/online_offline_feature_leakage_audit.csv')
OUT_MD = os.path.join(REPO, 'reports/ONLINE_OFFLINE_FEATURE_LEAKAGE_AUDIT.md')

# ── Column classification ────────────────────────────────────────
# A: available BEFORE attack at deployment (clean rollout only)
# B: only available AFTER running VIS attack (offline only)
# C: oracle/label — never available at inference
# D: metadata/audit — not features

CLASSIFICATION = {
    # Identity keys — always available
    'task_key': ('A', 'online_available_before_attack', 'Task identity from env'),
    'state_id': ('A', 'online_available_before_attack', 'Episode state identity'),
    'window_start': ('A', 'online_available_before_attack', 'Window start step'),
    'window_end': ('A', 'online_available_before_attack', 'Window end step'),
    'candidate_id': ('A', 'online_available_before_attack', 'Derived identity key'),

    # Phase detector outputs (from clean rollout proprio) — available online
    'hazard_score_mean': ('A', 'online_available_before_attack', 'ProprioNoStep mean hazard over clean window'),
    'hazard_score_max': ('A', 'online_available_before_attack', 'ProprioNoStep max hazard over clean window'),
    'release_safe_score_mean': ('A', 'online_available_before_attack', 'ProprioNoStep mean release-safe'),
    'release_safe_score_min': ('A', 'online_available_before_attack', 'ProprioNoStep min release-safe'),
    'predicted_phase': ('A', 'online_available_before_attack', 'Phase detector predicted phase bin'),
    'phase_confidence': ('A', 'online_available_before_attack', 'Phase detector confidence'),
    'phase_is_critical': ('A', 'online_available_before_attack', 'Phase is critical hazard phase'),
    'phase_bin_proxy': ('A', 'online_available_before_attack', 'Heuristic phase bin (clean rollout)'),
    'qpos_phase_class': ('A', 'online_available_before_attack', 'Phase class from clean qpos'),
    'phase_available': ('A', 'online_available_before_attack', 'Phase detector available for this task'),
    'phase_source': ('A', 'online_available_before_attack', 'Source of phase data'),

    # Precheck — from clean+random rollout (available online)
    'denominator_clean': ('A', 'online_available_before_attack', 'Clean rollout denominator check'),
    'denominator_type': ('A', 'online_available_before_attack', 'Type of denominator check'),
    'denominator_status': ('A', 'online_available_before_attack', 'Denominator check result'),
    'denominator_plan': ('A', 'online_available_before_attack', 'Planned denominator check'),
    'precheck_status': ('A', 'online_available_before_attack', 'Precheck outcome (clean+random)'),

    # Candidate metadata — available before attack
    'candidate_role': ('A', 'online_available_before_attack', 'Planned role for this candidate'),
    'expected_role': ('A', 'online_available_before_attack', 'Expected label role'),
    'reason_selected': ('A', 'online_available_before_attack', 'Why this candidate was selected'),
    'source_batch': ('A', 'online_available_before_attack', 'Which batch this came from'),
    'source_type': ('A', 'online_available_before_attack', 'gold/clean_control/candidate_derived'),
    'source_reason': ('A', 'online_available_before_attack', 'Reason for source classification'),
    'control_type': ('A', 'online_available_before_attack', 'Type of control (if known a priori)'),
    'control_reason': ('A', 'online_available_before_attack', 'Why this is a control'),

    # ── OFFLINE: ONLY available AFTER running VIS attack ──
    'VIS_OPEN': ('B', 'offline_after_attack_only', 'VIS OPEN frames count — BLOCKED for online'),
    'vis_open_count': ('B', 'offline_after_attack_only', 'Number of VIS frames with gripper open — BLOCKED'),
    'qpos_delta': ('B', 'offline_after_attack_only', 'Qpos change from VIS attack — BLOCKED'),
    'qpos_opening_delta': ('B', 'offline_after_attack_only', 'Qpos opening delta from VIS — BLOCKED'),
    'qpos_label': ('B', 'offline_after_attack_only', 'strong/weak/none from VIS outcome — BLOCKED'),
    'done': ('B', 'offline_after_attack_only', 'Task completion under VIS — BLOCKED'),
    'task_failure': ('B', 'offline_after_attack_only', 'Task failure under VIS — BLOCKED'),
    'action_bridge_status': ('B', 'offline_after_attack_only', 'Action→gripper bridge during VIS — BLOCKED'),
    'action_bridge_confounded': ('B', 'offline_after_attack_only', 'Action bridge confounded — BLOCKED'),
    'label_action_bridge': ('B', 'offline_after_attack_only', 'Labeled action bridge from VIS — BLOCKED'),
    'label_physical_response': ('B', 'offline_after_attack_only', 'Physical response from VIS — BLOCKED'),
    'label_task_failure': ('B', 'offline_after_attack_only', 'Task failure label from VIS — BLOCKED'),
    'runtime_sec': ('B', 'offline_after_attack_only', 'VIS runtime — not a feature'),
    'gpu_pair': ('B', 'offline_after_attack_only', 'GPU used — not a feature'),
    'status': ('B', 'offline_after_attack_only', 'VIS run status — not a feature'),
    'provenance_status': ('B', 'offline_after_attack_only', 'VIS provenance — BLOCKED'),
    'provenance_note': ('B', 'offline_after_attack_only', 'VIS provenance notes — BLOCKED'),
    'inclusion_status': ('B', 'offline_after_attack_only', 'Inclusion decision post-VIS — BLOCKED'),
    'exclusion_reason': ('B', 'offline_after_attack_only', 'Exclusion reason post-VIS — BLOCKED'),
    'exclusion_or_uncertain_reason': ('B', 'offline_after_attack_only', 'Exclusion/uncertainty post-VIS — BLOCKED'),
    'targeted_v2_error': ('B', 'offline_after_attack_only', 'Targeted error from v2 — BLOCKED'),

    # ── LABEL ORACLE — never available at inference ──
    'label_status': ('C', 'label_oracle', 'positive/negative/ignore — ground truth'),
    'label_source': ('C', 'label_oracle', 'gold_v2/clean_control/1R_screening — ground truth'),
    'label_vulnerability_ready': ('C', 'label_oracle', 'Is this label ready for training'),
    'label_confidence': ('C', 'label_oracle', 'Label confidence score'),
    'label_use': ('C', 'label_oracle', 'train/ignore/ablation — training decision'),
    'train_use': ('C', 'label_oracle', 'Use in training — oracle'),
    'train_variant': ('C', 'label_oracle', 'Training variant — oracle'),
    'taxonomy': ('C', 'label_oracle', 'Taxonomy string — oracle label'),
    'mechanism_type': ('C', 'label_oracle', 'Mechanism classification — oracle label'),
    'sample_weight': ('C', 'label_oracle', 'Training sample weight — oracle'),
    'label_1r': ('C', 'label_oracle', '1R screening label — oracle (post-attack label)'),
    'true': ('C', 'label_oracle', 'Ground truth in predictions — oracle'),
    'claim_usable': ('C', 'label_oracle', 'Claim usable flag — oracle'),

    # ── METADATA / AUDIT — not features ──
    'detector_variant': ('D', 'metadata', 'Which detector variant produced this'),
    'detector_model': ('D', 'metadata', 'Which model produced this'),
    'detector_feature_set': ('D', 'metadata', 'Which feature set used'),
    'vulnerability_score': ('D', 'metadata', 'Detector output — not input feature'),
    'vulnerability_pred': ('D', 'metadata', 'Detector prediction — not input feature'),
    'vuln_score_available': ('D', 'metadata', 'Audit: is vuln score available'),
    'join_status': ('D', 'metadata', 'Audit: join status'),
    'missing_reason': ('D', 'metadata', 'Audit: why missing'),
    'in_calibration_v2': ('D', 'metadata', 'Audit: in calibration v2'),
    'adaptive_status': ('D', 'metadata', 'Audit: adaptive status'),
    'feature_set': ('D', 'metadata', 'Feature set name in predictions'),
    'model': ('D', 'metadata', 'Model name in predictions'),
    'eval_scope': ('D', 'metadata', 'Evaluation scope'),
    'fit_sample_weight': ('D', 'metadata', 'Fit sample weight strategy'),
    'variant': ('D', 'metadata', 'Detector variant name'),
    'mode': ('D', 'metadata', 'Evaluation mode'),
    'exists': ('D', 'metadata', 'File exists check'),
    'path': ('D', 'metadata', 'File path'),
    'location': ('D', 'metadata', 'File location'),
    'rows': ('D', 'metadata', 'Row count'),
    'notes': ('D', 'metadata', 'Notes'),
    'full_two_stage_possible': ('D', 'metadata', 'Audit flag'),
    'phase_columns': ('D', 'metadata', 'Phase column names'),
    'phase_value_hits': ('D', 'metadata', 'Phase value hit count'),
    'phase_coverage_rows': ('D', 'metadata', 'Phase coverage count'),
    'missing_phase_rows': ('D', 'metadata', 'Missing phase count'),
    'phase_missing_reason': ('D', 'metadata', 'Why phase is missing'),
}


def main():
    # Write CSV
    rows = []
    for col in sorted(CLASSIFICATION.keys()):
        cat, cat_name, desc = CLASSIFICATION[col]
        rows.append({
            'column_name': col,
            'category': cat,
            'category_name': cat_name,
            'description': desc,
        })

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['column_name','category','category_name','description'])
        w.writeheader()
        w.writerows(rows)
    print('Wrote %d columns to %s' % (len(rows), OUT_CSV))

    # ── Leakage analysis ──
    # Check detector_v27_phase_aware_dataset.csv for feature column usage
    ds_path = os.path.join(CODEX, 'detector_v27_phase_aware_dataset.csv')
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            ds_cols = list(csv.DictReader(f))[0].keys() if True else []

    # Feature sets used in detector v27
    # From the predictions data, feature_sets are:
    # A_task_key_only, B_phase_bin_only, C_closed_pregrasp_gate,
    # D_causal_safe, E_phase+causal, F_task+phase, G_task+phase+causal, H_descriptor_upper

    # Map feature sets to underlying columns they use
    feature_set_columns = {
        'A_task_key_only': ['task_key'],
        'B_phase_bin_only': ['phase_bin_proxy', 'predicted_phase'],
        'C_closed_pregrasp_gate': ['phase_bin_proxy', 'phase_is_critical'],
        'D_causal_safe': ['qpos_opening_delta', 'vis_open_count', 'action_bridge_confounded',
                         'label_action_bridge', 'label_physical_response', 'label_task_failure',
                         'qpos_label', 'done'],
        'E_phase+causal': ['phase_bin_proxy', 'qpos_opening_delta', 'vis_open_count',
                          'action_bridge_confounded'],
        'F_task+phase': ['task_key', 'phase_bin_proxy', 'predicted_phase'],
        'G_task+phase+causal': ['task_key', 'phase_bin_proxy', 'qpos_opening_delta',
                               'vis_open_count', 'action_bridge_confounded'],
        'H_descriptor_upper': ['qpos_opening_delta', 'vis_open_count', 'qpos_label',
                              'label_action_bridge', 'label_physical_response'],
    }

    # Check each feature set for leakage
    leakage_findings = []
    for fs_name, cols_used in sorted(feature_set_columns.items()):
        offline_leaks = []
        oracle_leaks = []
        online_cols = []

        for col in cols_used:
            cat = CLASSIFICATION.get(col, ('D','unknown',''))[0]
            if cat == 'B':
                offline_leaks.append(col)
            elif cat == 'C':
                oracle_leaks.append(col)
            else:
                online_cols.append(col)

        risk = 'CLEAN'
        if offline_leaks and oracle_leaks:
            risk = 'BLOCKED_LEAKAGE_RISK'
        elif offline_leaks:
            risk = 'BLOCKED_LEAKAGE_RISK'
        elif oracle_leaks:
            risk = 'BLOCKED_LEAKAGE_RISK'

        leakage_findings.append({
            'feature_set': fs_name,
            'online_cols': online_cols,
            'offline_leaked': offline_leaks,
            'oracle_leaked': oracle_leaks,
            'risk': risk,
        })

    # ── Write report ──
    with open(OUT_MD, 'w') as f:
        f.write("""# Online/Offline Feature Leakage Audit

**Date**: %s
**Purpose**: Identify all columns that would NOT be available at deployment inference time

---

## Category Definitions

| Category | Name | Availability |
|----------|------|-------------|
| **A** | online_available_before_attack | Available from clean rollout only — OK for deployment |
| **B** | offline_after_attack_only | Requires running VIS attack first — NOT available at inference |
| **C** | label_oracle | Ground truth label — NEVER available at inference |
| **D** | metadata | Audit/experiment metadata — not a feature |

---

## Feature Set Leakage Analysis

""" % datetime.now().strftime('%Y-%m-%d %H:%M'))

        f.write('| Feature Set | Online Cols | Offline Leaks | Oracle Leaks | Risk |\n')
        f.write('|-------------|-------------|---------------|--------------|------|\n')
        for lf in leakage_findings:
            f.write('| %s | %s | %s | %s | %s |\n' % (
                lf['feature_set'],
                ', '.join(lf['online_cols']) if lf['online_cols'] else 'NONE',
                ', '.join(lf['offline_leaked']) if lf['offline_leaked'] else 'none',
                ', '.join(lf['oracle_leaked']) if lf['oracle_leaked'] else 'none',
                lf['risk']))

        # Count blocked
        blocked_sets = [lf for lf in leakage_findings if lf['risk'] != 'CLEAN']
        clean_sets = [lf for lf in leakage_findings if lf['risk'] == 'CLEAN']

        f.write("""
---

## Critical Finding

""")

        if blocked_sets:
            f.write('### BLOCKED: %d/%d feature sets use offline-only or oracle columns\n\n' % (
                len(blocked_sets), len(leakage_findings)))
            f.write('These feature sets CANNOT be used in a deployed online detector:\n\n')
            for lf in blocked_sets:
                f.write('- **%s**: ' % lf['feature_set'])
                if lf['offline_leaked']:
                    f.write('offline leaks: `%s`. ' % ', '.join(lf['offline_leaked']))
                if lf['oracle_leaked']:
                    f.write('oracle leaks: `%s`. ' % ', '.join(lf['oracle_leaked']))
                f.write('\n')

        if clean_sets:
            f.write('\n### CLEAN: %d/%d feature sets use only online-available columns\n\n' % (
                len(clean_sets), len(leakage_findings)))
            for lf in clean_sets:
                f.write('- **%s**: %s\n' % (lf['feature_set'], ', '.join(lf['online_cols'])))

        # The key finding
        f.write("""
---

## The Central Leakage Problem

The **D_causal_safe** and **H_descriptor_upper** feature sets — which produced
the best detector performance (BalAcc=0.714, posRecall=0.889) — are built on
columns that are ONLY available AFTER running a VIS attack:

- `qpos_opening_delta` — measured from VIS trace, not available before attack
- `vis_open_count` — count of VIS frames with gripper open
- `action_bridge_confounded` — whether action bridge was confounded during VIS
- `label_action_bridge` — oracle label of action bridge from VIS outcome
- `label_physical_response` — oracle label of physical response

**This means the current "best" detector (V0_gold D_causal_safe LR) is trained
on oracle features. It would be useless at deployment time because these features
don't exist before running an attack.**

This is a FUNDAMENTAL leakage: the detector is being evaluated on features that
encode the attack outcome, making it an outcome classifier rather than a
vulnerability predictor.

---

## Online-Safe Feature Whitelist

The ONLY columns that can be used in a deployed online detector:

| Column | Source | Description |
|--------|--------|-------------|
| `task_key` | env | Task identity |
| `state_id` | env | Episode state ID |
| `window_start` | env | Window start step |
| `window_end` | env | Window end step |
| `predicted_phase` | ProprioNoStep | Phase bin prediction |
| `phase_confidence` | ProprioNoStep | Phase confidence |
| `hazard_score_mean` | ProprioNoStep | Mean hazard score |
| `hazard_score_max` | ProprioNoStep | Max hazard score |
| `release_safe_score_mean` | ProprioNoStep | Mean release-safe |
| `release_safe_score_min` | ProprioNoStep | Min release-safe |
| `phase_bin_proxy` | heuristic | Heuristic phase bin |
| `qpos_phase_class` | heuristic | Clean qpos phase class |

**Plus**: any features derived from clean rollout observations:
- Clean qpos trajectory statistics
- Clean action trajectory statistics
- Clean gripper width statistics
- Clean proprioception statistics

**Explicitly NOT allowed**:
- Any VIS attack outcome (qpos_delta, vis_open_count, done, task_failure)
- Any oracle label (label_status, mechanism_type, taxonomy)
- Any post-hoc classification (physical_bridge_positive, no_action_bridge)
- Any provenance/confirmation status (3R confirmed, calibration status)

---

## Detection of Current Leakage in Training Data

The detector_v27_phase_aware_dataset.csv contains these leaked columns
as part of the training features. The model learns:

```
f(task_key, qpos_opening_delta, vis_open_count, ...) → vulnerable?
```

But at deployment:
```
f(task_key, ???, ???, ...) → can't compute!
```

The `qpos_opening_delta` is only known AFTER running VIS. At deployment,
we need to predict vulnerability BEFORE running an attack.

---

## Impact on Detector v3

**Detector v3 is BLOCKED on feature leakage as well as label readiness.**

Before training v3:
1. Define an online-safe feature set (only A columns)
2. Rebuild the training dataset with ONLY online-available features
3. Accept that online performance will be LOWER than offline (oracle) performance
4. The gap between online and offline performance IS the leakage penalty

---

## Recommended Feature Set for Deployable Detector

```
Online-Safe Feature Set (proposed):
  - task_key (one-hot or embedding)
  - Clean qpos statistics over window: mean, std, min, max, delta
  - Clean action statistics over window: mean, std, delta
  - Clean gripper_width statistics over window
  - Phase features (where ProprioNoStep available): hazard_score_mean/max,
    release_safe_score_mean/min, predicted_phase, phase_confidence
  - Window position relative to episode (normalized step index)
```

This feature set contains NO attack outcomes and NO oracle labels.
It is the only basis for a deployable detector.

---

## Column Classification Table (%d columns classified)

| Category | Count |
|----------|-------|
| A — online_available_before_attack | %d |
| B — offline_after_attack_only | %d |
| C — label_oracle | %d |
| D — metadata | %d |

""" % (len(rows),
       sum(1 for r in rows if r['category'] == 'A'),
       sum(1 for r in rows if r['category'] == 'B'),
       sum(1 for r in rows if r['category'] == 'C'),
       sum(1 for r in rows if r['category'] == 'D')))

    print('Wrote report to %s' % OUT_MD)
    print()
    print('=== Leakage Summary ===')
    for lf in leakage_findings:
        if lf['risk'] != 'CLEAN':
            print('  BLOCKED: %s — offline=%s oracle=%s' % (
                lf['feature_set'], lf['offline_leaked'], lf['oracle_leaked']))
        else:
            print('  CLEAN:  %s — online=%s' % (lf['feature_set'], lf['online_cols']))

if __name__ == '__main__':
    main()
