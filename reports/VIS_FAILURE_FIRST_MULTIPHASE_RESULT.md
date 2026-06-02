# VIS Failure-First Multi-Phase Result

**Date**: 2026-06-02

Status: pending rollout completion.

The failure-first queue targets state0 seed0 for `cream_cheese`, `salad_dressing`, `ketchup`, and `tomato_sauce` using the official 20260525 environment. Results will be populated from:

- `tables/vis_failure_first_rollout_summary.csv`
- `tables/vis_failure_first_phasewise_metrics.csv`
- `tables/vis_failure_first_random_controls.csv`
- `tables/vis_failure_first_claims.csv`

No claim is made until the rollout summary, matched random controls, and qpos/CQ/manual audit are reconciled.
