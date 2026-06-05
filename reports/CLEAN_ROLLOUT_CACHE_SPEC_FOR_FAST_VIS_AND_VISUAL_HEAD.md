# Clean Rollout Cache Spec For Fast VIS And VisualTransferHead

This is a future cache spec only. It does not implement replay or training.

## Required Fields

- `image_path`
- `frame_step`
- `agentview_image`
- `gripper_qpos_mujoco`
- `robot0_gripper_qpos_obs`
- `clean_raw_action`
- `clean_env_action`
- `eef_pose`
- `task_key`
- `state_id`
- `window_start`
- `window_end`
- `provenance_status`

## Purpose

- Unblock policy-only Phase C checks.
- Unblock VisualTransferHead dataset construction.
- Unblock Phase E aligned windows with MuJoCo qpos rather than obs-only qpos.
- Reduce repeated clean replay cost.

## Boundaries

- Cache fields must come from clean rollout only.
- Attack outcome, task outcome, VIS result, or manual audit labels must not be detector inputs.
- GPU3 and GPU7 remain blacklisted for any future cache generation.
