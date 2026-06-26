# Phase A+B Canonical Ledger Reconciliation

Generated: 2026-06-26T17:33:40.938271

## Executive Summary

- **Total run directories with episode_summary.json**: 398
- **Total with .done marker**: 388
- **Total with summary but no .done**: 10
- **Duplicate scientific keys**: 0
- **Checkpoint SHA violations**: 0

## Why Not 411?

The expected 411 was: 270 (original) + 87 (Phase A) + 54 (Phase B) = 411.
SHUFFLED_T10 (33 runs) were part of the 270 but are not in the current directory tree.
4 canary runs + 4 seed expansion runs have summaries but no .done marker.

### SHUFFLED Discrepancy

The SHUFFLED_T10 33 runs are NOT in the current directory tree.
No directory names contain "shuffled" and no episode_summary.json references shuffled.

### Missing .done Markers (Data Present)

8 runs have complete episode_summary.json but no .done marker:

- canary/canary_a_tv_vis: success=False, atk_frames=10, emit=76
- canary/canary_c_nc: success=False, atk_frames=0, emit=-1
- canary/canary_b_tv_rand: success=True, atk_frames=10, emit=76
- early_shift/canary_e2_tomato_s42: success=True, atk_frames=10, emit=117
- early_shift/canary_e1_salad_s42: success=False, atk_frames=10, emit=-1
- random_time/canary_r1_salad_s42: success=True, atk_frames=10, emit=74
- random_time/canary_r2_tomato_s42: success=True, atk_frames=10, emit=117
- seed2026_vis/bbq_sauce_s0_s2026: success=False, atk_frames=10, emit=77
- seed789_ours_armlock/bbq_sauce_s0_s4789: success=False, atk_frames=10, emit=77
- seed789_rand/tomato_sauce_s0_s5789: success=True, atk_frames=10, emit=117
- seed789_tma_vanilla/milk_s4_s2789: success=False, atk_frames=10, emit=72

## TMA Early Final

- butter_s2 seed123: COMPLETED (success=False, atk_frames=10, emit=82)
- **TMA Early FR: 12/27 = 0.444**
- Matches Prefix Early FR: 12/27 = 0.444

## 5-Seed Corrected 2x2 Matrix

| Condition | FR |
|-----------|----|
| TMA no-lock | 36/45 = 80.0% |
| TMA ArmLock | 37/45 = 82.2% |
| Prefix no-lock | 36/45 = 80.0% |
| Prefix ArmLock | 45/45 = 100.0% |
