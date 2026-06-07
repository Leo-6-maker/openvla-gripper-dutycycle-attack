# Stage-B Qpos Raw Recovery Audit

**Date**: 2026-06-07

## Classification

| Status | Count | % |
|---|---|---|
| DIRECT_RECOVERABLE | 3 | 1.5% |
| ACTION_REPLAY_REQUIRED | 0 | 0% |
| NOT_RECOVERABLE | 203 | 98.5% |

## Root Cause

Old labeling script saved only:
- `gripper_qpos` = constant 0.5 (wrong qpos[-2:]/qpos[-1:] indices)
- `env_grip` = decoded gripper scalar (+1/-1)
- No full `env_action` (7-dim) vector saved

Without full action vectors, action replay reconstruction is impossible.
The trace CSVs lack the 7-dim action needed to feed back into the LIBERO env.

## Samples

DIRECT_RECOVERABLE traces (n=3):
- trace_butter_random_linf_job40.csv
- trace_orange_juice_random_linf_job70.csv
- trace_salad_dressing_random_linf_job22.csv

These likely came from a code path where qpos was computed differently
(e.g., random_linf without PGD, or from the post-fix script).

## Decision

- **physical_response_label = BLOCKED_QPOS_MEASUREMENT** for all 203 non-recoverable traces
- **Future traces**: fixed runner uses `obs["robot0_gripper_qpos"]` correctly
- **No action replay possible** without full env_action vectors
- **Smoke-B**: run on command_susceptible_label only

## Recommendation

For future labeling runs, always save per-step:
- `obs_robot0_gripper_qpos_0`, `obs_robot0_gripper_qpos_1`
- `env_action_0..6` (full 7-dim action vector)
- `qpos_source` = "obs_robot0_gripper_qpos"
