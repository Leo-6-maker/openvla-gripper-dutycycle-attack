# Labels V2 MuJoCo Qpos Phase Audit

**Status**: BLOCKED_MISSING_LABELS_V2
**Input**: `tables/object_phase_response_labels_v2.csv`
**Rows audited**: 0
**true_closed rows**: 0
**natural_open rows**: 0
**phase_proxy_mismatch rows**: 0

This is a CPU-only CSV audit. It does not run rollout, VIS, GPU work, or detector training.

## Status Counts

- None.

## Notes

- labels CSV not found: tables/object_phase_response_labels_v2.csv

## Interpretation Boundary

- `true_closed` is an input-quality marker for Batch4 candidate selection, not evidence of attack success.
- `natural_open` windows are poor hard-negative candidates unless explicitly intended as controls.
- `phase_proxy_mismatch` rows require manual review before entering detector training or Batch4 execution.
