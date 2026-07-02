# CLEAN2000 Model Selection Protocol V1

## Frozen Before Test

All checkpoint selection rules are frozen BEFORE any test-suite metrics are computed. Post-hoc re-selection based on test performance is prohibited.

## Primary Selection Metric

**Validation suite-macro event F1**

Computed as: macro-average of per-suite event F1 scores on validation split. Only training suites contribute (LOSO: 3 training suites; Pooled: validation split from all 4 suites).

## Tie-Break Order

1. Lower false emits per episode (on validation)
2. Lower post-release trigger rate (on validation)
3. Simpler / earlier checkpoint (fewer training steps)

## Forbidden Selection Criteria

The following MUST NOT be used for checkpoint selection:
- Test-suite F1 (any aggregation)
- Worst test-suite F1
- Downstream attack ASR (TRUE_T10 or any condition)
- Any metric computed on test split
- Human visual inspection of test predictions

## Early Stopping

```
Patience: TBD (suggested: 20 epochs on validation F1)
Monitor: Validation suite-macro event F1
Direction: maximize
Min delta: 0.001
```

Training stops when validation F1 does not improve for `patience` epochs. Best checkpoint restored.

## Seed Policy

- Fixed 3 seeds per detector variant
- Seeds: TBD (deterministic, documented in config)
- All 3 seeds reported; no seed selection based on results
- Aggregate: median with IQR, not mean of best seed

## Threshold Selection

- tau_corridor and tau_release selected on validation only
- For LOSO: thresholds optimized on 3 training suites' validation split
- Threshold sweep grid: TBD (suggested: 0.1 to 0.9 in 0.05 steps)
- Selection criterion: maximize validation event F1
- Thresholds frozen BEFORE test evaluation
