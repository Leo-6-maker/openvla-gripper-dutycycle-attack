# Batch4 FP-Driven Qpos-Verified Candidates

**Status**: BLOCKED_MISSING_INPUTS
**Error CSV**: `tables/detector_v2_error_analysis.csv`
**Qpos/phase audit CSV**: `tables/labels_v2_mujoco_qpos_phase_audit.csv`
**Labels CSV**: `tables/object_phase_response_labels_v2.csv`
**Candidates**: 0
**Hard-negative/control rows**: 0
**Controls**: 0

This generator is CPU-only. It does not run rollout, VIS, GPU work, watcher jobs, or detector training.

## Notes

- missing input: tables/object_phase_response_labels_v2.csv

## Candidate Summary

- No candidates generated.

## Claim Boundary

- These rows are scheduling candidates only.
- Phase D/E proxy labels are excluded and must not be treated as gold labels.
- Candidate readiness still depends on schema audit and DeepSeek server-side generation of full labels_v2 artifacts.
