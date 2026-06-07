# Detector V2 Error Analysis

**Status**: BLOCKED_MISSING_INPUTS
**Labels CSV**: `tables/object_phase_response_labels_v2.csv`
**Predictions CSV**: `tables/detector_v2_predictions.csv`
**Metrics CSV**: `tables/detector_v2_metrics.csv`
**Metrics rows**: 0
**Error rows written**: 0
**False positives**: 0
**False negatives**: 0

This is a CPU-only analysis. It does not run rollout, VIS, GPU work, or detector training.

## Blocking / Review Notes

- BLOCKED_MISSING_INPUT: labels CSV not found: tables/object_phase_response_labels_v2.csv
- BLOCKED_MISSING_INPUT: predictions CSV not found: tables/detector_v2_predictions.csv

## Error Rows

- No error rows available.

## Expected Gate

- The current Batch4 planning assumption expects 6 detector-v2 false positives and 1 false negative.
- If the observed count differs, Batch4 candidates should be reviewed before any server-side execution.
