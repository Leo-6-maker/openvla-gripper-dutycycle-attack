# L12 E4C.2b — Final Label Pool

**Date**: 2026-06-15
**Stage**: E4C.2b (accepted)
**Commit**: `2450031`
**Branch**: `exp/l12-critical-close-window-selector-20260615`

## Final Eligibility Pool

| Category | Count |
|----------|-------|
| ELIGIBLE_MULTI_CANDIDATE (primary ranking pool) | 131 |
| ELIGIBLE_SINGLE_CANDIDATE (binary only, no within-trace neg) | 53 |
| TEACHER_P_AMBIGUOUS (>1 TP-qualifying, needs disambiguation) | 131 |
| TEACHER_P_UNAVAILABLE (0 TP-qualifying, must abstain) | 79 |
| NO_CLOSE_CANDIDATE (0 CLOSE candidates) | 8 |
| All gate failures | 0 |
| **Total** | **402** |

Per-task distribution of primary ranking pool:

| Task | Multi |
|------|-------|
| alphabet_soup | 18 |
| bbq_sauce | 13 |
| butter | 17 |
| chocolate_pudding | 7 |
| cream_cheese | 15 |
| ketchup | 21 |
| milk | 6 |
| orange_juice | 8 |
| salad_dressing | 10 |
| tomato_sauce | 16 |

## Label Rules

**Positive**: exactly one `is_teacher_p=1` candidate per trace (the unique
TP-qualifying candidate). 131 traces × 1 positive = 131 positives.

**Negative**: all other CLOSE candidates in the same trace (non-TP-qualifying).
131 traces × 760 total candidates - 131 positives = 629 negatives.

**Abstain/Excluded**:
- 53 single-candidate traces: no within-trace negative, excluded from ranking loss
- 131 ambiguous traces: >1 TP-qualifying, no unique positive label
- 79 unavailable traces: 0 TP-qualifying
- 8 no-candidate traces: 0 CLOSE candidates

## Provenance

Manifest and three label-producing source files (remapper, phase_detector,
critical_close_selector) passed frozen SHA comparison at runtime. The runner
SHA was recorded. Config SHA was expected-only (not deployed in runtime
pipeline directory).

Source SHAs (matched at runtime):
- remap_v4_trace_for_l12.py: `5d9cf327...`
- phase_detector.py: `f9cc7e90...`
- critical_close_selector.py: `81b510ec...`

## Data Quality

- 9 consistency assertions all PASS
- 3197 descriptive domain warnings (non-gating)
- 0 missing/parse/non-finite field violations
- 0 open-convention violations
- 0 RC1a invariant violations
- `time_since_last_open`: 2952/2957 candidates have valid values

## Candidate Features (all exported)

All 16 frozen E4B.3 features per candidate:
total_score, raw_crossing_bonus, close_streak_bonus, close_onset_qpos_bonus,
eef_deceleration_bonus, qpos_ready_bonus, eef_speed_now, eef_speed_prev,
eef_deceleration_delta, close_streak, raw_crossing, close_onset, qpos,
time_since_prev_close, time_since_last_open, candidate_index.

Plus: selector_abstain_reason, selector_emittable, and 6 Teacher-P evidence fields.
