# Split And Leakage Spec V1

Status: PLANNING_ONLY

## Unit Of Separation

`parent_key` and initial state hash are split units. No child, seed,
condition, state hash, normalization stat, checkpoint selection signal, or
threshold-selection signal may cross split boundaries.

## Required Split Manifests

- `parent_random_split_v1` for pooled detector train/val/test.
- `object_leave_task_out_v1` for Object task-timing memorization checks.
- `suite_loso_split_v1` for held-out suite evaluation.

## Detector Regimes

| Regime | Train | Test | Forbidden |
|---|---|---|---|
| Object-only | Object train | Object and held suites | target-suite normalization |
| Pooled | train partitions from all suites | held-out test partitions | test labels or outcomes |
| LOSO | three suites | held-out suite | held-out normalization, weights, checkpoint selection |

## Test Freeze

Validation may set thresholds before test access. Test results cannot change
thresholds, detector architecture, feature normalization, attack protocol, or
metric definitions.
