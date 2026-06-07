# Detector Incremental Readiness Snapshot

**Time**: 2026-06-06 12:48
**VIS1R milestones reached**: 20

## Current State

| Metric | Value |
|--------|-------|
| Labels v2 train | 22 (9 pos, 13 neg, 9 tasks) |
| VIS-1R completed | 20 |
| VIS-1R infra/precheck fail | 18 |
| Pending | 106 |
| Calibration paired | 10/6 |
| Calibration PASS | yes |

## Projected Labels

| Scenario | Train rows | Notes |
|----------|-----------|-------|
| Conservative (v2 only) | 22 | only confirmed labels |
| Optimistic (+ all 1R) | 42 | includes provisional silver |

## Gates

| Gate | Requirement | Status |
|------|------------|--------|
| v2.5 ablation | >=30 train + >=8 silver_1R | **PASS** |
| v3 diagnostic | >=35 train + >=6 hard_neg + calibration PASS | BLOCKED |

## Notes

- provisional_silver_positive_1r is NOT gold. Sample weight = 0.5 only in ablation.
- pending_negative_1r NEVER enters train.
- infra/precheck-fail excluded.
- Detector NOT auto-trained.
