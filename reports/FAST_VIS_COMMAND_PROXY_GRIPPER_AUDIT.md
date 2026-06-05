# Fast VIS Command Proxy Gripper Audit

Date: 2026-06-05

Scope: static code audit and interface repair only. No GPU, rollout, VIS, watcher, or detector v2 training was started.

## Verdict

Status: PATCHED_FOR_RETRY_V2

DeepSeek found that commit `c17e2c9` still had P0 command-proxy issues. The previous Phase D v0 result pattern, `8/8 done=False, qpos=0, 300 steps`, remains invalid for scientific comparison because the measurement chain could silently use a zero-valued observation qpos path.

## Fixes Applied

- MuJoCo gripper joint qpos is now the primary measurement source.
- `obs["robot0_gripper_qpos"]` is fallback/audit comparison only, not the default primary source.
- The output records `gripper_qpos_mujoco`, `gripper_qpos_obs`, `gripper_qpos_used`, and `gripper_qpos_source_priority`.
- If MuJoCo and obs qpos differ by more than `1e-3`, the output records `gripper_qpos_warning=mujoco_obs_qpos_mismatch` instead of silently using zero.
- If neither source is available, the row is marked `MEASUREMENT_FAILED`.
- `MEASUREMENT_FAILED` rows use `label_confidence=not_label_measurement_failed` and must not enter proxy-label comparison.
- Forced OPEN is injected after the production normalize/invert transform, directly in final `env.step()` action space.
- The script now uses local `MODEL_PATH=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` with `local_files_only=True` where supported.
- Task selection now uses the fixed LIBERO Object task-id mapping instead of string matching.
- Environment setup now aligns with `vis_rollout_adaptive_v3.py`: `benchmark_dict["libero_object"]()`, `OffScreenRenderEnv`, `initial_states[state_id]`, and `env.set_init_state(init_state)`.

## Versioned Fields Added

The command-proxy output schema now records:

- `measurement_version`
- `action_injection_version`
- `gripper_qpos_source`
- `gripper_qpos_mujoco`
- `gripper_qpos_obs`
- `gripper_qpos_used`
- `gripper_qpos_source_priority`
- `gripper_qpos_warning`
- `clean_gripper_action`
- `forced_gripper_action`
- `forced_open_value_used`
- `post_transform_gripper_action`

## Semantics

- `clean_gripper_action`: decoded raw model gripper action before transform, recorded from the first forced-window step.
- `forced_gripper_action`: raw canonical OPEN value, currently `0.0`.
- `forced_open_value_used`: final env-step OPEN value after normalize/invert transform, currently `+1.0`.
- `post_transform_gripper_action`: actual gripper value passed to `env.step()` during the forced window.
- `qpos_opening_delta`: first pre-window forced-step used qpos minus minimum post-step used qpos inside the forced window.

## Microcheck Status

DeepSeek's local microcheck v2 validates MuJoCo qpos measurement and final env-step OPEN injection. Codex did not rerun GPU/rollout; this patch only aligns the formal script with that verified path and dry-run checks.

## Remaining Requirement

DeepSeek should discard the previous Phase D v0 command-proxy rows and rerun after syncing this patch. The rerun must pass `scripts/diagnostics/audit_fast_vis_outputs.py` before any comparison or advisor-facing claim. Command proxy remains an upper-bound proxy, not VIS proof.
