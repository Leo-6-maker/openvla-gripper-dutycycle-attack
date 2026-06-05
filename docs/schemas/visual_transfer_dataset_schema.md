# Visual Transfer Dataset Schema

One row is one trigger-centered candidate-window sample for Stage 2 `vulnerability_ready` or VisualTransferHead diagnostics.

## Identity

| Field | Meaning |
|---|---|
| `sample_id` | Stable unique sample id. |
| `source_batch` | Batch/source name, for example `batch1`, `batch2b`, `batch3`, `batch3b`, `batch3c`. |
| `task_key` | Task identifier. |
| `state_id` | Initial state id. |
| `seed` | Seed if known. |
| `run_id` | Trace/run id if known. |
| `condition` | Source condition, for example clean/probed/VIS/random if known. |
| `candidate_role` | Role such as `pre_lock_control`, `stable_post_lock_control`, or blank if unavailable. |
| `phase_bin_proxy` | Phase/candidate bin. |
| `window_start` | Candidate window start step. |
| `window_end` | Candidate window end step. |
| `trigger_step` | Trigger-centered frame step, usually `window_start` for v0. |

## Online Proprio / Context Features

These are model-input eligible only if available before or at trigger time:

| Field | Meaning |
|---|---|
| `gripper_qpos_at_trigger` | Gripper qpos at trigger. |
| `gripper_width_at_trigger` | Gripper width at trigger. |
| `gripper_command_at_trigger` | Gripper command at trigger. |
| `eef_speed_mean_pre` | EEF speed summary before trigger. |
| `open_streak_pre` | Pre-trigger open streak. |
| `close_streak_pre` | Pre-trigger close streak. |
| `phase_gate_score` | Optional phase gate score. |
| `hazard_score` | Optional risk/hazard score. |

## Visual Paths

| Field | Meaning |
|---|---|
| `image_trigger_path` | RGB image path at trigger. |
| `image_trigger_minus4_path` | RGB image path at trigger - 4. |
| `image_trigger_minus8_path` | RGB image path at trigger - 8. |
| `image_camera_name` | Camera name. |
| `visual_available` | True only if trigger and requested past images are available. |
| `missing_visual_reason` | Missing reason if visual paths are incomplete. |

## Optional Embedding Fields

| Field | Meaning |
|---|---|
| `global_embedding_path` | Path to frozen global embedding. |
| `crop_embedding_path` | Path to frozen crop/object embedding. |
| `visual_encoder_name` | Encoder name. |
| `visual_feature_dim` | Feature dimension. |

## Labels

| Field | Meaning |
|---|---|
| `label_vulnerability_ready` | Stage 2 vulnerability-ready label. |
| `label_physical_response` | Physical qpos/gripper-response label. |
| `label_task_failure` | Task-failure label. |
| `label_control_negative` | Control-negative label. |
| `label_status` | `positive`, `negative`, `ignore`, or `manual_review`. |
| `label_source` | Label provenance. |

## Audit

| Field | Meaning |
|---|---|
| `denominator_status` | Denominator/provenance status. |
| `provenance_status` | Trace/provenance status. |
| `infra_status` | Infra status, for example Xid/OOM/missing trace. |
| `leakage_audit_pass` | Whether leakage audit passed. |

## Leakage Boundary

The following fields may be used only as labels or audit metadata, never as model inputs:

- `qpos_delta_after_attack`
- `done`
- `VIS_OPEN`
- `claim_usable`
- `vis_open_count`
- `denominator_clean`
- Random / oracle / manual audit outcomes.
- Any attack outcome field.

Deployable student inputs must not include `object_pose` or `target_pose`.
