# L12 E4C.2 — RC1a Remap + Teacher-P Eligibility Report

**Date**: 2026-06-15
**Stage**: E4C.2
**Branch**: `exp/l12-critical-close-window-selector-20260615`
**Remapper**: `rc1a_corrected_v2_e1_5`
**Host**: klfy-SYS-4028GR-TR2 (8× RTX 2080 Ti, CUDA available)
**Elapsed**: 9.4s (42.7 traces/s)
**TRAINING_STARTED: NO**

---

## Gate Results

| Gate | Pass | Fail |
|------|------|------|
| Provenance (SHA + row count + existence) | 402 | 0 |
| Field validity (all rows, 9 required fields) | 402 | 0 |
| Open convention (decoded_open_bool iff env < -0.5) | 402 | 0 |
| RC1a remap (valid rows + zero invariants) | 402 | 0 |

All 402 frozen schema-passing traces from E4C.1 passed all 4 hardware gates.
No provenance failures, no field corruption, no open-convention inversions,
no RC1a invariant violations.

## Eligibility Classification

| Category | Count | % of 402 |
|----------|-------|----------|
| ELIGIBLE_MULTI_CANDIDATE | 261 | 64.9% |
| ELIGIBLE_SINGLE_CANDIDATE | 54 | 13.4% |
| TEACHER_P_UNAVAILABLE | 87 | 21.6% |
| PROVENANCE_FAIL | 0 | 0.0% |
| FIELD_VALIDITY_FAIL | 0 | 0.0% |
| OPEN_CONVENTION_FAIL | 0 | 0.0% |
| RC1A_REMAP_FAIL | 0 | 0.0% |

**Teacher-P available**: 315/402 (78.4%)
**Multi-candidate traces eligible for ranking**: 261/402 (64.9%)
**Single-candidate traces (no ranking negative possible)**: 54/402 (13.4%)

## Candidate Statistics

- Total CLOSE candidates enumerated: 1,780
- Teacher-P candidates: 315
- Non-P candidates: 1,465
- Average non-P candidates per eligible trace: 4.7 (1465/315)
- Average non-P candidates per multi-candidate trace: 5.6 (1465/261)

## Per-Task Coverage

| Task | Total | TP Available | Multi | Single | TP Unavailable | TP Rate |
|------|-------|-------------|-------|--------|---------------|---------|
| alphabet_soup | 38 | 35 | 30 | 5 | 3 | 92.1% |
| bbq_sauce | 43 | 19 | 16 | 3 | 24 | 44.2% |
| butter | 42 | 29 | 29 | 0 | 13 | 69.0% |
| chocolate_pudding | 41 | 32 | 32 | 0 | 9 | 78.0% |
| cream_cheese | 42 | 36 | 36 | 0 | 6 | 85.7% |
| ketchup | 42 | 39 | 32 | 7 | 3 | 92.9% |
| milk | 41 | 31 | 16 | 15 | 10 | 75.6% |
| orange_juice | 41 | 35 | 19 | 16 | 6 | 85.4% |
| salad_dressing | 37 | 31 | 23 | 8 | 6 | 83.8% |
| tomato_sauce | 35 | 28 | 28 | 0 | 7 | 80.0% |

All 10 tasks have at least 16 multi-candidate traces (bbq_sauce minimum).
Teacher-P availability ranges from 44.2% (bbq_sauce) to 92.9% (ketchup).

## bbq_sauce Note

bbq_sauce has the lowest Teacher-P availability (19/43, 44.2%) with 24
Teacher-P-unavailable traces. These traces have valid grasp privilege but no
close candidate satisfies all 5 Teacher-P criteria (close_onset + eef near
object + sustained vertical lift). This is consistent with bbq_sauce being
a task where the gripper closes early or the object is not lifted vertically
after close. These 24 traces must abstain — they cannot be used as negatives.

## Data Quality Assessment

- **Provenance**: All 402 file SHAs match E4C.1 inventory. No server-side
  file modification since E4C.1 snapshot.
- **Field validity**: All 402 traces have every row of all 9 required
  fields non-empty, parseable, and finite. No 50-row sampling — full
  row-level audit.
- **Open convention**: Zero violations of `decoded_open_bool == 1` iff
  `clean_gripper_env < -0.5`. RC1a convention consistently applied.
- **Placement privilege**: 0/402 traces have target coordinate fields
  (confirmed in E4C.1). Teacher-P is grasp-only.

## CPU/GPU Parity

The same 10-trace parity subset was processed identically:
- Both runs used identical algorithms, thresholds, and code
- GPU available but current algorithms are pure Python/numpy (no CUDA ops)
- Results: 10/10 traces identical classification and candidate enumeration
- No GPU-specific code paths affect label generation

## Training Eligibility Pool

**Eligible for within-trace ranking (D1b training)**: 261 traces
- All have Teacher-P available + >= 2 CLOSE candidates
- Non-P candidates serve as negatives within same trace
- 10/10 tasks represented

**Eligible for binary classification only**: 54 traces
- Teacher-P available but only 1 candidate
- Cannot construct within-trace ranking negatives
- Can be used for prevalence estimation

**Abstain (Teacher-P unavailable)**: 87 traces
- Must NOT be used as negatives
- May be used for abstention-rate calibration

## Next Step

D1b preregistration: freeze train/val/test split on the 261 multi-candidate
traces, run leakage audit, freeze model config and checkpoint rule, then
await reviewer authorization before training.
