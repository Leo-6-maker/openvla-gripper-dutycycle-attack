# Exact-W32 Calibration Diagnosis — 2026-07-22

**Status:** HEAD_LEVEL_POLICY_REPLAY — Awaiting Codex V5 scheduler contract

## 1. Erratum Summary

The initial offline FSM replay contained a critical known-mask violation:
- Manipulation head included in any-head union without `manipulation_known_mask` check
- `manipulation_known_mask` is False on ~all background steps (o2_i0: 1/24467)
- Inflated any-head P50 from 0.004 to 0.516 and streak estimates by ~3x

**Erratum:** [EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md](EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md)
**Status:** RETRACTED_DUE_TO_KNOWN_MASK_BUG → CORRECTED_HEAD_LEVEL_REPLAY

## 2. 12-Split Complete Results

All 12 exact-W32 inner-CV splits completed (train→predict→evaluate→audit PASS).

### Release Discrimination (stable)

| Metric | Min | Max | Mean | Range |
|--------|-----|-----|------|-------|
| Release AUPRC | 0.832 | 0.884 | 0.862 | 0.052 |
| Release Short AUPRC | 0.927 | 0.985 | 0.949 | 0.058 |
| Release First AUPRC | 0.860 | 0.930 | 0.887 | 0.070 |
| Release Later AUPRC | 0.813 | 0.897 | 0.859 | 0.084 |

### Safety Gates

| Gate | Pass Rate | Range | Worst |
|------|-----------|-------|-------|
| background_false_emit ≤ 0.10 | 2/12 | 0.080-0.174 | o0_i2 (0.174) |
| release_overlap ≤ 0.05 | 12/12 | 0.007-0.017 | — |
| unsupported_emit = 0 | 12/12 | 0.0 | — |

**Worst-split frozen gate: FAIL** (o0_i2 bg=0.174, o2_i0 bg=0.174)

### Background Emit Decomposition

| Split | BG Emit | bg_grasp | bg_release | Gate |
|-------|---------|----------|------------|------|
| o1_i1 | **0.080** | 0.076 | 0.006 | ✓ |
| o1_i2 | **0.093** | 0.086 | 0.009 | ✓ |
| o1_i0 | 0.109 | 0.093 | 0.020 | ✗ |
| o2_i1 | 0.114 | 0.102 | 0.015 | ✗ |
| o2_i2 | 0.127 | 0.115 | 0.014 | ✗ |
| o3_i1 | 0.128 | 0.113 | 0.017 | ✗ |
| o0_i0 | 0.142 | 0.132 | 0.022 | ✗ |
| o0_i1 | 0.166 | 0.141 | 0.029 | ✗ |
| o2_i0 | 0.174 | 0.153 | 0.034 | ✗ |
| o0_i2 | 0.174 | 0.157 | 0.023 | ✗ |

(2 splits pending CSV finalization)

### Corrected Consecutive Background Streaks (tau=0.5, known-only)

All 12 splits show elevated consecutive bg emit. Representative values:
- o1_i2: max=106, 136/180 eps with >1 consecutive, 112/180 with >2
- o2_i0: max=292, 156/180 eps with >1, 147/180 with >2
- o0_i1: max=384 (highest observed)
- o1_i1: max=206

These streaks use grasp|release with known_mask filtering at tau=0.5. Awaiting long-streak semantic audit to determine whether they represent genuine false trigger risk or reasonable pre-grasp anticipation.

## 3. Per-Split Individual Feasibility

**Awaiting Pareto scan completion.** See `analysis/student_trigger_calibration/per_split_pareto.csv`.

Preliminary (from corrected 2-split analysis):
- o1_i2: feasible points exist (bg=0.093 passes at tau=0.5)
- o2_i0: NO feasible point with recall≥0.5 AND false≤0.1 at any scanned parameter

## 4. Cross-Split Common Feasibility

**Awaiting full 12-split scan.** Preliminary (2-split): 0/420 viable.

See `analysis/student_trigger_calibration/common_operating_point_search.csv`.

## 5. Class Weight Shift Audit

Cross-split pos_weight variation is minimal:

| Head/Route | Min PW | Max PW | Range | Log Range |
|------------|--------|--------|-------|-----------|
| grasp/single | 0.878 | 0.883 | 0.005 | 0.006 |
| release/single | 1.206 | 1.241 | 0.036 | 0.029 |
| grasp/multi | 0.630 | 0.653 | 0.023 | 0.036 |
| release/multi | 0.604 | 0.628 | 0.024 | 0.039 |

**Verdict: POS_WEIGHT_SHIFT_REJECTED_AS_PRIMARY_CAUSE** (max log-range = 0.039)

Score distribution differences across splits are NOT primarily driven by class_weight calibration differences.

## 6. Identity Composition Assessment

**Awaiting leave-one-identity-out analysis completion.**

Preliminary: identity composition is CONFOUNDED with checkpoint and split assignment. With 4 outer folds × 3 inner folds, each split's held-out identities differ. We cannot cleanly separate identity effects from checkpoint effects without cross-prediction.

**Assessment: IDENTITY_EFFECT_CONFOUNDED_WITH_CHECKPOINT_AND_SPLIT**

## 7. Calibration Transfer Experiments

**Awaiting completion.** Methods tested:
- A. Analytic de-weighting (z - log(pos_weight))
- B. Train-only intercept calibration
- C. Leave-one-split-out threshold transfer

See `analysis/student_trigger_calibration/calibration_transfer_results.csv`.

## 8. Long-Streak Semantic Audit

**Awaiting completion.** Top-20 longest streaks exported to `analysis/student_trigger_calibration/long_streak_case_audit.csv`.

Key question: Are high grasp scores on background steps actually pre-grasp anticipation, post-release artifacts, or genuine false triggers?

## 9. V5 Scheduler Replay Status

**Current:** HEAD_LEVEL_POLICY_REPLAY — NOT actual V5 FSM.

**Pending:** Codex delivery of `v5_scheduler.py` with:
- State machine (GRASPED/MANIPULATING/RELEASED)
- Dwell timing
- 3-of-5 persistence
- Release/regrasp veto
- candidate_close definition
- Offline-reconstructable fields

**Contract:** `analysis/student_trigger_calibration/replay_contract.json`

## 10. Model Decision

**Current case:** Pending Pareto completion. Based on preliminary (2-split) analysis:

- If all splits individually feasible: CASE_2 (cross-split calibration shift)
- If some splits individually infeasible: CASE_3 (mixed)
- Root cause: identity composition confounded with checkpoint

**Recommendation:** HOLD on model architecture changes. Proceed to Full-FIT + independent CAL only after:
1. Pareto scan confirms per-split feasibility
2. Calibration transfer shows shared calibrator viability
3. Codex V5 scheduler contract received

## 11. GO / HOLD Summary

| Item | Status |
|------|--------|
| Finish 12 splits | **DONE** (12/12 PASS) |
| Engineering Full-FIT | **HOLD** (pending calibration diagnosis) |
| Modify loss | **HOLD** (pending root cause determination) |
| Passive shadow | **HOLD** (pending V5 scheduler + CAL) |
| Active smoke | **HOLD** |
| Student training launched | **NO** (this round) |
| OpenVLA rollout launched | **NO** |
| Attack launched | **NO** |
| CAL/CHECK accessed | **NO** |
| Old artifacts modified | **NO** |

## 12. Output Files

| File | Status |
|------|--------|
| `docs/reports/EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md` | Written |
| `analysis/student_trigger_calibration/metric_contract.json` | Written |
| `analysis/student_trigger_calibration/per_split_metrics.csv` | Written (12 splits) |
| `analysis/student_trigger_calibration/per_split_pareto.csv` | Computing |
| `analysis/student_trigger_calibration/common_operating_point_search.csv` | Computing |
| `analysis/student_trigger_calibration/identity_failure_breakdown.csv` | Computing |
| `analysis/student_trigger_calibration/long_streak_case_audit.csv` | Computing |
| `analysis/student_trigger_calibration/class_weight_shift_audit.json` | Computing |
| `analysis/student_trigger_calibration/calibration_transfer_results.csv` | Computing |
| `analysis/student_trigger_calibration/replay_contract.json` | Computing |
| `tests/detector_v5/test_known_mask_contract.py` | Written (10/10 PASS) |
| `docs/reports/EXACT_W32_CALIBRATION_DIAGNOSIS_20260722.md` | This file |
