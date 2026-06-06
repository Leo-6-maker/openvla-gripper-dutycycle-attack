# Active Probe V1 — PGD Budget Diagnostic Report

**Date**: 2026-06-07
**Rows**: 11 disagreement windows
**Method**: PGD3 vs PGD10, same 3 frames/window, no-env decode
**Comparison**: PGD3_3frame vs PGD10_3frame (SAME frames, NOT cross-sampling)
**VIS reference**: PGD20 (batch1) or PGD40x3 (batch3), WITH env.step
**PGD20**: SKIPPED (prohibitively slow on 7B model)

## Per-Row Diagnostic Results

| Window | Type | VIS | Clean3f | PGD3_3f | PGD10_3f | t-c PGD3 | t-c PGD10 | Transition | Diagnosis |
|---|---|---|---|---|---|---|---|---|---|
| butter_s3_w29_46 | HIGH_PROBE_N | 0 | 1 | 1 | 2 | +0 | +1 | NEW_FP: PGD10 introduces false | BUDGET_OVERSHOOT: PGD10 creates false positive tha |
| cream_cheese_s4_w28_45 | POSITIVE_LAB | 18 | 0 | 1 | 1 | +1 | +1 | AGREE_POSITIVE | ALIGNED: both PGD3 and PGD10 agree with VIS label; |
| salad_dressing_s0_w31_48 | HIGH_PROBE_N | 0 | 1 | 3 | 2 | +2 | +1 | PERSISTENT_FP: both PGD3 and P | PARTIAL_BUDGET: PGD10 changes signal but does not  |
| alphabet_soup_s4_w4_21 | POSITIVE_LAB | 18 | 2 | 2 | 2 | +0 | +0 | PERSISTENT_FN: both PGD3 and P | BUDGET_INSENSITIVE: PGD10 same as PGD3 — not a bud |
| milk_s4_w19_36 | POSITIVE_LAB | 18 | 1 | 0 | 1 | -1 | +0 | PERSISTENT_FN: both PGD3 and P | PARTIAL_BUDGET: PGD10 changes signal but does not  |
| salad_dressing_s5_w28_45 | HIGH_PROBE_N | 0 | 0 | 2 | 0 | +2 | +0 | REPAIRED: PGD10 fixed FP | PGD3_UNDERPOWERED: higher budget corrects the erro |
| ketchup_s1_w21_38 | POSITIVE_LAB | 18 | 1 | 0 | 0 | -1 | -1 | PERSISTENT_FN: both PGD3 and P | BUDGET_INSENSITIVE: PGD10 same as PGD3 — not a bud |
| alphabet_soup_s6_w40_57 | HIGH_PROBE_N | 0 | 3 | 3 | 2 | +0 | -1 | CEILING: PGD10 < PGD3 (unexpec | CEILING_ARTIFACT: delta-to-clean invalid; raw targ |
| ketchup_s0_w16_33 | CEILING_POSI | 18 | 3 | 2 | 1 | -1 | -2 | CEILING: PGD10 < PGD3 (unexpec | CEILING_ARTIFACT: delta-to-clean invalid; raw targ |
| milk_s5_w25_42 | POSITIVE_LAB | 18 | 2 | 1 | 0 | -1 | -2 | PERSISTENT_FN: both PGD3 and P | PARTIAL_BUDGET: PGD10 changes signal but does not  |
| ketchup_s5_w9_26 | CEILING_POSI | 18 | 3 | 1 | 0 | -2 | -3 | CEILING: PGD10 < PGD3 (unexpec | CEILING_ARTIFACT: delta-to-clean invalid; raw targ |

## Transition Breakdown

### PERSISTENT_FN: both PGD3 and PGD10 miss VIS signal (n=4)
- **alphabet_soup_s4_w4_21**: PGD3 t-c=+0 streak=1, PGD10 t-c=+0 streak=1, VIS open=18 | BUDGET_INSENSITIVE: PGD10 same as PGD3 — not a budget issue; NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget
- **ketchup_s1_w21_38**: PGD3 t-c=-1 streak=0, PGD10 t-c=-1 streak=0, VIS open=18 | BUDGET_INSENSITIVE: PGD10 same as PGD3 — not a budget issue; NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget
- **milk_s4_w19_36**: PGD3 t-c=-1 streak=0, PGD10 t-c=+0 streak=1, VIS open=18 | PARTIAL_BUDGET: PGD10 changes signal but does not align with VIS; NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget
- **milk_s5_w25_42**: PGD3 t-c=-1 streak=1, PGD10 t-c=-2 streak=0, VIS open=18 | PARTIAL_BUDGET: PGD10 changes signal but does not align with VIS; NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

### CEILING: PGD10 < PGD3 (unexpected) (n=3)
- **ketchup_s0_w16_33**: PGD3 t-c=-1 streak=2, PGD10 t-c=-2 streak=1, VIS open=18 | CEILING_ARTIFACT: delta-to-clean invalid; raw targeted_open_rate is primary metric
- **ketchup_s5_w9_26**: PGD3 t-c=-2 streak=1, PGD10 t-c=-3 streak=0, VIS open=18 | CEILING_ARTIFACT: delta-to-clean invalid; raw targeted_open_rate is primary metric; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget
- **alphabet_soup_s6_w40_57**: PGD3 t-c=+0 streak=3, PGD10 t-c=-1 streak=1, VIS open=0 | CEILING_ARTIFACT: delta-to-clean invalid; raw targeted_open_rate is primary metric; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

### NEW_FP: PGD10 introduces false positive (n=1)
- **butter_s3_w29_46**: PGD3 t-c=+0 streak=1, PGD10 t-c=+1 streak=2, VIS open=0 | BUDGET_OVERSHOOT: PGD10 creates false positive that PGD3 avoids; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

### AGREE_POSITIVE (n=1)
- **cream_cheese_s4_w28_45**: PGD3 t-c=+1 streak=1, PGD10 t-c=+1 streak=1, VIS open=18 | ALIGNED: both PGD3 and PGD10 agree with VIS label; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

### PERSISTENT_FP: both PGD3 and PGD10 disagree with VIS (n=1)
- **salad_dressing_s0_w31_48**: PGD3 t-c=+2 streak=3, PGD10 t-c=+1 streak=1, VIS open=0 | PARTIAL_BUDGET: PGD10 changes signal but does not align with VIS; NO_ENV_SURROGATE_SUSPECTED: even PGD10 no-env disagrees with rollout VIS; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

### REPAIRED: PGD10 fixed FP (n=1)
- **salad_dressing_s5_w28_45**: PGD3 t-c=+2 streak=1, PGD10 t-c=+0 streak=0, VIS open=0 | PGD3_UNDERPOWERED: higher budget corrects the error; NOTE: VIS uses PGD40 rollout; PGD10 no-env still 4x under budget

## Budget Effect vs Sampling Effect

Same 3 frames used for PGD3_3f and PGD10_3f. No cross-sampling comparison needed.

- **ketchup_s0_w16_33**: PGD3 t-c=-1 → PGD10 t-c=-2 (delta=-1, LESS open)
- **alphabet_soup_s4_w4_21**: PGD3 t-c=+0 → PGD10 t-c=+0 (delta=+0, SAME)
- **butter_s3_w29_46**: PGD3 t-c=+0 → PGD10 t-c=+1 (delta=+1, MORE open)
- **cream_cheese_s4_w28_45**: PGD3 t-c=+1 → PGD10 t-c=+1 (delta=+0, SAME)
- **ketchup_s1_w21_38**: PGD3 t-c=-1 → PGD10 t-c=-1 (delta=+0, SAME)
- **ketchup_s5_w9_26**: PGD3 t-c=-2 → PGD10 t-c=-3 (delta=-1, LESS open)
- **milk_s4_w19_36**: PGD3 t-c=-1 → PGD10 t-c=+0 (delta=+1, MORE open)
- **milk_s5_w25_42**: PGD3 t-c=-1 → PGD10 t-c=-2 (delta=-1, LESS open)
- **salad_dressing_s0_w31_48**: PGD3 t-c=+2 → PGD10 t-c=+1 (delta=-1, LESS open)
- **alphabet_soup_s6_w40_57**: PGD3 t-c=+0 → PGD10 t-c=-1 (delta=-1, LESS open)
- **salad_dressing_s5_w28_45**: PGD3 t-c=+2 → PGD10 t-c=+0 (delta=-2, LESS open)

## Decision Logic

| Criteria | Count | Verdict |
|---|---|---|
| Repaired (>=6) | 1 | — |
| Mixed (2-5) | 1 | — |
| No repair (<=1) | 1 | NO_ENV_SURROGATE_UNRELIABLE_LIKELY |

## Verdict: NO_ENV_SURROGATE_UNRELIABLE_LIKELY

PGD10 does not repair the disagreements. The no-env decode paradigm is likely not a reliable surrogate for rollout VIS attack outcomes, regardless of PGD budget.

### Next Steps

1. Stop Active Probe as VIS-label predictor. 2. Redirect to rollout-aware short-horizon probe or direct VIS empirical search. 3. Do NOT train detector.
