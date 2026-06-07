# Phase E Aligned Windows V0

**Candidates source**: `tables/fast_vis_calibration_candidates_v0.csv`
**Labels source**: `tables/object_phase_response_labels_v2.csv`
**closed_threshold**: 0.015
**open_threshold**: 0.005
**Rows generated**: 120
**Recommended for Phase E**: 0
**MuJoCo trace rows**: 0
**Obs trace only rows**: 0
**Missing qpos rows**: 120
**Obs all-zero rows**: 0
**Dry run**: True
**allow_obs_qpos_recommendation**: False

This is a CPU-only candidate audit. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.

## Notes

- labels CSV not found: tables/object_phase_response_labels_v2.csv; qpos/denominator fields may be missing

## MuJoCo Requirement

- MuJoCo qpos is required for automatic Phase E recommendation.
- Obs qpos is audit-only by default because server traces showed all-zero obs qpos false positives.
- Obs-only windows require manual review and cannot launch GPU canary unless an explicit override is used.
- `obs_qpos_all_zero_untrusted` rows are never auto-recommended.

## Qpos Phase Rule

- `qpos >= 0.015`: `true_closed`.
- `qpos <= 0.005`: `natural_open`.
- Otherwise: `transitional-pre-open`.
- `true_closed` may be recommended when denominator/provenance/mismatch gates pass.
- `transitional-pre-open` may be recommended when `true_closed_score >= 0.35` and gates pass.
- `natural_open` and missing-qpos rows are rejected.

## Qpos Phase Counts

- `true_closed`: 0
- `transitional-pre-open`: 0
- `natural_open`: 0
- `missing`: 120

## Selection Rule

- Do not assume centered L10 is valid.
- Recommend true_closed windows directly after denominator/provenance/mismatch gates.
- Recommend transitional-pre-open windows only when true_closed_score is at least 0.35.
- MuJoCo qpos is required for automatic recommendation; obs qpos is audit-only by default.
- Polluted denominators, severe phase proxy mismatch, and infra-failed provenance block recommendation.

## Trace Root Guidance

- Broad `/data/liuyu/outputs` scans may miss traces because the script caps CSV scanning for safety.
- Prefer specific trace roots when available:
  - `/data/liuyu/outputs/nightly_object_batch3_20260604`
  - `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604`
  - `/data/liuyu/outputs/object_phase_response_batch4_...`
