# D1b — Checkpoint Selection Rule (Frozen)

**Date**: 2026-06-15
**Stage**: D1b prereg only
**TRAINING_STARTED: NO**

## Rule (preregistered, immutable)

1. **Primary metric**: highest validation per-trace Teacher-P top-1 accuracy.
   A trace is "correct" iff the highest-scoring candidate is the unique
   Teacher-P candidate (is_teacher_p=1).

2. **Tiebreak 1**: lower validation mean absolute timing error.
   Error = |highest_scoring_step - teacher_p_step|.

3. **Tiebreak 2**: earlier epoch.

## Threshold Selection

The decision threshold for emitting a prediction is determined on the
validation set only, using the frozen checkpoint. The threshold is set
to maximize validation per-trace Teacher-P top-1 accuracy.

No candidate that scored below the threshold is emitted as a prediction
(treated as abstain). The abstain rate is reported separately.

## Forbidden

- Selecting checkpoint or threshold after seeing test results
- Iterating checkpoint rule after seeing validation results
- Using test set for any purpose before final evaluation
- Tuning any hyperparameter based on validation results and then
  re-running the same validation split

## Single Evaluation Protocol

After checkpoint and threshold are frozen:
1. Load frozen checkpoint
2. Apply frozen normalization (train statistics only)
3. Apply frozen threshold
4. Run ONE inference pass on test set
5. Report all metrics
6. No retraining, no threshold adjustment, no feature subset selection
