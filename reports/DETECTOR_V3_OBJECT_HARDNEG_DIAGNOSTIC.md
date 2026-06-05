# Detector V3 Object Hard-Negative Diagnostic

**Status**: BLOCKED_NOT_READY
**Reason**: readiness report is not READY_FOR_DETECTOR_V3
**Labels CSV**: `tables/object_phase_response_labels_v3_candidate.csv`
**Metrics output**: `tables/detector_v3_object_hardneg_metrics.csv`
**Predictions output**: `tables/detector_v3_predictions.csv`

This scaffold does not train detector v3. It is CPU-only dry-run support.

## Required Comparisons

- prevalence / always_positive
- task_key_only
- phase_only
- D_causal_safe LR
- D_causal_safe RF if available
- D_causal_safe + qpos_verified features if available
- task + phase + causal_safe diagnostic

## Metrics

- balanced_accuracy
- macro_F1
- F1_pos
- F1_neg
- negative_recall
- positive_recall
- MCC
- FP count
- FN count
- LOTO if feasible
- source-batch breakdown
- hard-negative FP reduction vs v2