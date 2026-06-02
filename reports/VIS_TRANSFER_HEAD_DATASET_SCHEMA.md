# VIS Transfer Head Dataset Schema

**Date**: 2026-06-02

## Purpose

VisualTransferHead is the proposed next mainline for the VIS physical-transfer gap. It should predict whether a VIS action trace is likely to produce physical gripper response and task failure susceptibility. This document defines the dataset boundary only; no model training is authorized here.

## Scientific Boundary

- Do not train VisualTransferHead unless Leon explicitly approves.
- Do not use simulator success alone as a failure label.
- Do not claim detector-triggered VIS is validated from this dataset.
- Treat current adaptive-controller rows as exploratory negative calibration.
- Use post-step qpos and repaired denominators for future rows whenever available.

## Allowed Model Inputs

Allowed inputs must be available at or before the decision point and must not directly encode future physical response or final outcome.

| Field Group | Example Fields | Input Status | Notes |
|-------------|----------------|--------------|-------|
| Task metadata | `task`, `task_family`, `object_name`, `receptacle_name`, `instruction_template_id` | allowed_input | Use stable categorical encodings; avoid free-text leakage from result summaries. |
| Window metadata | `window_start`, `window_end`, `window_steps`, `policy_step_norm` | allowed_input | Phase/timing is already handled by ProprioNoStep, but timing features can be used for ablation. |
| VIS configuration | `epsilon`, `step_size`, `attack_steps`, `objective`, `strategy`, `duration_budget` | allowed_input | Configuration known before rollout. |
| Clean decode trace | `clean_grip`, `clean_action_prefix`, `clean_action_l2_norm` | allowed_input | Per-step clean policy output before perturbation. |
| VIS decode trace | `adv_grip`, `token_flip`, `arm_l2`, `linf`, `open_streak_so_far`, `open_count_so_far` | allowed_input | Must be computed online from attacked steps only. |
| Pre-step proprio | `qpos_pre_step`, `qpos_delta_pre_so_far`, `eef_pos_pre_step` | allowed_input | Causally available before the next env action. |
| Controller state | `attacks_applied_so_far`, `controller_mode`, `budget_remaining` | allowed_input | Only if used to predict the next action/stop decision. |

## Label-Only / Audit-Only Fields

These fields must not be used as model inputs because they encode future response or final outcome.

| Field | Status | Reason |
|-------|--------|--------|
| `qpos_post_step` | audit_only | Observed only after `env.step`; useful for posthoc validation. |
| `qpos_delta_after_window` | label_only | Encodes the physical response the model is meant to predict. |
| `qpos_delta_post` | label_only | Future physical-transfer outcome. |
| `official_success` | label_only | Outcome label, not a predictor. |
| `cq_success` | label_only | Contact-quality outcome label. |
| `cq_failure` | label_only | Contact-quality outcome label. |
| `sr_cq_mismatch` | label_only | Manual/CQ audit outcome, unavailable online. |
| `manual_audit_needed` | audit_only | Triage flag, not a physical predictor. |
| `failure_phase` | label_only | Posthoc explanation. |
| `done`, `reward` | label_only | Simulator outcome after execution. |

## Required Row Granularity

Preferred dataset shape is per-step with episode-level labels joined by `run_id`:

| Key | Requirement |
|-----|-------------|
| `run_id` | Stable unique trace/run identifier. |
| `task` | Required. |
| `seed` | Required. |
| `policy_step` | Required. |
| `effective_attack_step_idx` | Required for attacked-step denominators. |
| `attack_attempted` | Required. |
| `pgd_applied` | Required. |
| `controller_active` | Required if adaptive rows are included. |
| `qpos_pre_step` | Required model-input proprio field. |
| `qpos_post_step` | Required audit/label-only field for future physical-response validation. |

## Future Baselines

The first approved VisualTransferHead experiment should include these baselines:

| Baseline | Inputs |
|----------|--------|
| `task_only` | Task metadata and window metadata only. |
| `visual_only` | VIS decode trace and perturbation metrics only. |
| `proprio_only` | Pre-step proprio and qpos history only. |
| `visual_plus_proprio` | VIS decode trace plus pre-step proprio. |

## Acceptance Checks Before Training

1. Every row has repaired denominator flags: `attack_attempted`, `pgd_applied`, `controller_active`, `controller_stopped`, and `effective_attack_step_idx`.
2. Every future physical-response field is marked `label_only` or `audit_only`.
3. `qpos_delta_after_window` is never included in model input columns.
4. CQ placeholders are present even if unknown: `official_success`, `cq_success`, `cq_failure`, `sr_cq_mismatch`, `manual_audit_needed`.
5. At least one random or clean denominator table exists for interpreting VIS-specific failure attribution.
