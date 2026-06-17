# H3: Attack-Window Mapping — Corrected Report

**Classification:** H3_POINT_ONLY

## Summary

- 21 preregistered steps across 3 parents
- 9 clean-eligible (CLEAN_ELIGIBLE, gripper=31872 CLOSE)
- 12 clean-ineligible (CLEAN_ALREADY_TARGET, gripper=31744 OPEN)
- 18 frame-seed results (12 new + 6 reused anchors)
- 7 frame-seed PASS, all at anchor steps

## Per-Parent Windows

| Parent | Anchor | Eligible Steps | Two-Seed PASS | Max Width |
|--------|--------|---------------|---------------|----------|
| butter_s11 | 60 | [60, 61, 62, 63] | [60] | 1 |
| tomato_sauce_s23 | 141 | [141] | [141] | 1 |
| salad_dressing_s11 | 59 | [59, 60, 61, 62] | [59] | 1 |

## Scientific Interpretation

The V4 hard-feasible VIS attack exhibits **narrow temporal sensitivity** centered on Teacher-P/clean-CLOSE anchor steps. Only 7 of 18 clean-eligible frame-seeds pass the hard-feasible gate, all at the three anchor points. Non-anchor frames within +-3 steps fail the hard-feasible gate (NO_FEASIBLE_PGD_CANDIDATE).

For tomato_sauce_s23, all 6 non-anchor steps in the +-3 window are CLEAN_ALREADY_TARGET (clean policy already outputs OPEN=31744), so they provide no counterfactual condition for a CLOSE-to-OPEN attack.

**Key limitation:** Only butter_s11 has D5 first emit coincident with the attack-effective point (D5=anchor=60). Tomato (D5=69, anchor=141) and salad (D5=128, anchor=59) do not have D5-aligned effective points.
