# Phase E Aligned Windows V0

**Candidates source**: `tables/fast_vis_calibration_candidates_v0.csv`
**Labels source**: `tables/object_phase_response_labels_v2.csv`
**Rows generated**: 120
**Recommended for Phase E**: 0
**Missing qpos rows**: 120
**Dry run**: True

This is a CPU-only candidate audit. It does not run rollout, VIS, watcher jobs, GPU work, or detector training.

## Notes

- labels CSV not found: tables/object_phase_response_labels_v2.csv; qpos/denominator fields may be missing

## Selection Rule

- Do not assume centered L10 is valid.
- Recommend only true_closed or transitional-pre-open windows with low natural-open score.
- MuJoCo qpos is preferred; obs qpos is fallback; missing qpos is never auto-recommended.
- Polluted denominators, severe phase proxy mismatch, and infra-failed provenance block recommendation.
