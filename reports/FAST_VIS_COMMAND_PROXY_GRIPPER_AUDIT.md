# Fast VIS Command Proxy Gripper Audit

Date: 2026-06-05

Scope: static code audit and interface repair only. No GPU, rollout, VIS, watcher, or detector v2 training was started.

## Verdict

Status: PATCHED_FOR_RETRY

The previous Phase D v0 result pattern, `8/8 done=False, qpos=0, 300 steps`, is invalid for scientific comparison because the command proxy measured gripper response from `env._joint_positions`. That field is not a gripper qpos source for this audit.

## Fixes Applied

- `env._joint_positions` is no longer used for gripper qpos.
- Gripper qpos is read from `obs["robot0_gripper_qpos"]` first.
- If the observation key is unavailable, the script falls back to MuJoCo gripper joint qpos lookup.
- If neither source is available, the row is marked `MEASUREMENT_FAILED`.
- `MEASUREMENT_FAILED` rows use `label_confidence=not_label_measurement_failed` and must not enter proxy-label comparison.
- Forced OPEN is injected after the production normalize/invert transform, directly in final `env.step()` action space.

## Versioned Fields Added

The command-proxy output schema now records:

- `measurement_version`
- `action_injection_version`
- `gripper_qpos_source`
- `clean_gripper_action`
- `forced_gripper_action`
- `forced_open_value_used`
- `post_transform_gripper_action`

## Semantics

- `clean_gripper_action`: decoded raw model gripper action before transform, recorded from the first forced-window step.
- `forced_gripper_action`: raw canonical OPEN value, currently `0.0`.
- `forced_open_value_used`: final env-step OPEN value after normalize/invert transform, currently `+1.0`.
- `post_transform_gripper_action`: actual gripper value passed to `env.step()` during the forced window.
- `qpos_opening_delta`: first pre-window forced-step qpos minus minimum post-step qpos inside the forced window.

## Remaining Requirement

DeepSeek should discard the previous Phase D v0 command-proxy rows and rerun after syncing this patch. The rerun must pass `scripts/diagnostics/audit_fast_vis_outputs.py` before any comparison or advisor-facing claim.
