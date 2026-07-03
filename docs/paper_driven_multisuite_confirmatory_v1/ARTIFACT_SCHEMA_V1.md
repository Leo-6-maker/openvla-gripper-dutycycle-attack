# Artifact Schema V1

Status: PLANNING_ONLY

## Unique Keys

`experiment_id`, `suite`, `parent_key`, `condition`, `branch_id`, `attempt`.

## Version Rule

Every artifact includes `schema_version`, `created_at_utc`, `producer_git_sha`,
`producer_file_sha256`, and `input_manifest_sha256`. Required fields cannot be
removed without a new schema version.

## Episode Manifest Row

| field | dtype | required |
|---|---|---|
| experiment_id | string | yes |
| population_id | string | yes |
| suite | string | yes |
| task_id | string | yes |
| parent_key | string | yes |
| condition | string | yes |
| branch_id | string | yes |
| clean_prefix_snapshot_sha256 | sha256 | yes |
| output_root | path | yes |
| retry_group_id | string | yes |

## Runtime Telemetry Row

| field | dtype | required |
|---|---|---|
| step_index | int | yes |
| attack_applied_actual | bool | yes |
| delta_linf | float | attack rows |
| delta_l2 | float | attack rows |
| target_loss | float | attack rows |
| gripper_command_raw | float | yes |
| gripper_command_executed | float | yes |
| gripper_qpos | float | yes |
| gripper_width | float | yes |
| arm_action_l2_vs_clean | float | matched rows |

## Condition Envelope

| field | dtype | required |
|---|---|---|
| condition_id | string | yes |
| manifest_sha256 | sha256 | yes |
| denominator_ledger_sha256 | sha256 | yes |
| retry_ledger_sha256 | sha256 | retryable phases |
| aggregate_metrics_sha256 | sha256 | yes |
| source_code_sha256s | object | yes |
| config_sha256s | object | yes |

No table cell may cite a result without a denominator ledger and a source
artifact SHA.
