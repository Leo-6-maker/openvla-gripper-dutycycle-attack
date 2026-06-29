# CLEAN2000 LOSO Protocol V1

## Design

```
4 folds × 3 training suites × 1 held-out test suite
```

## Isolation Requirements

For each fold, the test suite is:
- Excluded from ALL training data
- Excluded from normalization statistics (mean/std computed from 3 train suites only)
- Excluded from class weight estimation
- Excluded from threshold selection
- Excluded from checkpoint selection
- Excluded from early stopping decisions
- Excluded from validation split

Validation split is drawn from the 3 training suites (episode-grouped, 80/20 train/val within training suites).

## Per-Fold Output

```
fold_{heldout_suite}/
  train_log.csv
  validation_metrics.json
  test_metrics.json (on held-out suite)
  checkpoint.pt
  normalization.json
  threshold.json
  config.yaml
```

## LOSO Aggregate Metrics

Report all 4 folds individually, plus:
- Mean over 4 folds
- Standard deviation over 4 folds
- Worst-suite performance
- Best-suite performance

Do NOT pool all 4 test predictions and report a single micro-average. Each fold's test predictions are on different suites with potentially different positive class prevalence.

## Comparison to Object-only

LOSO Fold 4 (test on Object) is particularly informative:
- Object-only frozen detector was trained on Object data
- LOSO Fold 4 was trained on Spatial+Goal+LIBERO-10 (NO Object data)
- Comparing them tests whether Object-specific patterns are learnable from other suites

## Comparison to Balanced Pooled

- Pooled trains on all 4 suites → supervised interpolation
- LOSO trains on 3 → held-out suite generalization
- Gap between Pooled and LOSO on a suite = generalization gap for that suite
