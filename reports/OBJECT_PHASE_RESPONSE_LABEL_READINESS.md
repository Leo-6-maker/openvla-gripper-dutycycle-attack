# Object Phase-Response Label Readiness v0

**Date**: 2026-06-04

## Label Distribution

| Label | Count |
|-------|-------|
| vulnerable (1) | 2 |
| not vulnerable (0) | 4 |
| ignored | 4 |
| Total | 10 |

## Positives by Phase Bin

| approach_far_closed_proxy | 2 |

## Positives by Task

| alphabet_soup | 1 |
| ketchup | 1 |

## Negatives

| alphabet_soup | approach_far_closed_proxy | weak_physical |
| butter | approach_near_closed_proxy | action_only_physical_none |
| butter | pre_lock_closed_proxy | physical_strong_task_negative |
| ketchup | pre_lock_closed_proxy | physical_strong_task_negative |

## Ignored

|  | denominator_not_clean |
|  | denominator_not_clean |
|  | denominator_not_clean |
| butter | infrastructure_or_polluted |

## Gates

| Gate | Requirement | Status |
|------|------------|--------|
| Smoke training | positives>=3, negatives>=2, tasks>=2 | FAIL |
| Paper-level | positives>=5, negatives>=5, tasks>=4 | FAIL |

## Verdict

Smoke gate NOT YET passed (pos=2, neg=4, tasks=2). More VIS results needed.