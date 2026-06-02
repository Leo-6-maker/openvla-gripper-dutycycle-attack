# Full10 Object Oracle Sensitivity — Frozen Result

**Status**: FROZEN | **Date**: 2026-05-30 | **Rollouts**: 100/100

## Experimental Design

- 10 Object tasks × 5 states (s0–s4) × 2 conditions (clean + oracle_open)
- Detector: ProprioNoStep TCN (SHA: 4b3f3d479...)
- Success: official = done_any (LIBERO)
- No fresh Xid. Server HEAD: c62214f (blob-equiv to 0870443).

## Clean Stability

| Task | Clean SR | Failed States |
|------|----------|---------------|
| cream_cheese | 5/5 | — |
| tomato_sauce | 5/5 | — |
| chocolate_pudding | 5/5 | — |
| ketchup | 5/5 | — |
| salad_dressing | 5/5 | — |
| orange_juice | 5/5 | — |
| milk | 4/5 | s3 |
| alphabet_soup | 3/5 | s0, s4 |
| butter | 3/5 | s1, s4 |
| bbq_sauce | 2/5 | s1, s2, s4 |

## Oracle Sensitivity Ranking

Oracle SR computed over **clean-eligible states only** (clean-failed states excluded from denominator).

| Rank | Task | Eligible | Oracle SR | Score | Class |
|------|------|----------|-----------|-------|-------|
| 1 | cream_cheese | 5/5 | **1/5** | 0.20 | high |
| 2 | tomato_sauce | 5/5 | **1/5** | 0.20 | high |
| 3 | butter | 3/5 | 1/3 | 0.33 | medium |
| 4 | chocolate_pudding | 5/5 | 2/5 | 0.40 | medium |
| 5 | bbq_sauce | 2/5 | 1/2 | 0.50 | medium |
| 6 | alphabet_soup | 3/5 | 2/3 | 0.67 | medium |
| 7 | milk | 4/5 | 3/4 | 0.75 | low |
| 8 | orange_juice | 5/5 | 4/5 | 0.80 | low |
| 9 | ketchup | 5/5 | **5/5** | 1.00 | robust |
| 10 | salad_dressing | 5/5 | **5/5** | 1.00 | robust |

## Key Findings

1. **Oracle physically opens gripper on ALL tasks** (qpos goes +0.030 → +0.017). Robustness is NOT from failed attack delivery.
2. **Tomato is NOT unique**: cream_cheese has identical sensitivity (1/5).
3. **Two tasks are truly robust**: ketchup and salad_dressing (5/5 oracle).
4. **Clean instability** on bbq_sauce (2/5), butter (3/5), alphabet_soup (3/5) limits sensitivity measurement.
5. **Sensitivity is task/object dependent**, not universal.

## Artifacts

- `tables/object_oracle_sensitivity_full10x5_ranking.csv`
- `tables/object_oracle_clean_stability_full10x5.csv`
- `tables/object_oracle_matched_denominator_full10x5.csv`
- `tables/object_oracle_qpos_vs_failure_full10x5.csv`
- `figures/object_oracle_sensitivity_full10x5_ranking.txt`
- `reports/OBJECT_WIDE_ORACLE_SENSITIVITY_TABLE.md`
- `reports/TASK_DEPENDENT_GRIPPER_SENSITIVITY_FULL_OBJECT.md`

## Interpretation

**Task-dependent oracle sensitivity confirmed across all 10 Object tasks.** Detector-selected windows are gripper-relevant, but causal impact varies strongly by task. Sustained proxy / future VIS should target high-sensitivity tasks (cream_cheese, tomato_sauce, butter, chocolate_pudding).
