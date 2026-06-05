# Fast VIS Output Schema Audit

**Overall status**: BLOCKED_MISSING_FAST_VIS_OUTPUTS

This is a CPU-only schema audit. It does not run rollout, VIS, watcher, or detector training.

## Dataset Status

| Dataset | Rows | Status | Issues |
|---|---:|---|---|
| policy_only | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |
| command_proxy | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |
| low_budget | 0 | BLOCKED_MISSING_FAST_VIS_OUTPUTS | missing_csv |

## Checks

- Required columns: task_key, state_id, window_start, window_end, label, label_source, label_confidence, gpu_pair, runtime_sec, provenance_status.
- Command-proxy additionally requires measurement_version, action_injection_version, gripper_qpos_source, gripper_qpos_mujoco, gripper_qpos_obs, gripper_qpos_used, gripper_qpos_source_priority, forced_open_value_used, post_transform_gripper_action, clean_gripper_action, and forced_gripper_action.
- denominator_status is required, including explicit not_applicable values for policy-only and command-proxy outputs.
- Proxy labels must not be marked gold.
- Rows with INFRA_FAILED/Xid/OOM/CUDA failures must not be treated as trainable labels.
- Rows with MEASUREMENT_FAILED must not be treated as proxy labels.

## Claim Boundary

- Missing output CSVs are reported as BLOCKED_MISSING_FAST_VIS_OUTPUTS, not as failed experiments.
- Policy-only and command-open proxy results are screening/proxy evidence only, not gold VIS labels.
