# Milestone 2C: Proprio Causal Student Baseline

## Purpose

Milestone 2C trains and evaluates a proprio-only causal student baseline for detector development. The model is intended to predict clean teacher phase, hazard, and release-safe labels from deployment-facing low-signal features. It is not a final online detector and it is not attack evidence.

## Inputs

Primary dataset:

```bash
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv
```

The dataset contains 87,474 timestep rows from 400 unique clean episodes.

## Output Artifact

Output root:

```bash
/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526
```

Export package:

```bash
/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/export/milestone_2c_proprio_causal_student_20260526.tar.gz
```

SHA256:

```text
2ebc109ebc384e4f59fc5ace2a6fc0e63414d93ecdcc2c8700e5d6c6381d5a68
```

## Model

Model: `ProprioCausalMLP`

Output heads:

- `phase_logits`: multiclass phase prediction
- `hazard_logit`: binary hazard prediction
- `release_safe_logit`: binary release-safe prediction
- optional confidence head if enabled by configuration

Loss combines phase cross entropy, hazard BCE, release-safe BCE, and optional confidence BCE.

## Allowed Input Features

Categorical inputs:

- `mechanism_type`
- `parse_confidence`

Numeric inputs:

- `gripper_command`
- `gripper_qpos`
- `gripper_width`
- `eef_x`
- `eef_y`
- `eef_z`
- `eef_vx`
- `eef_vy`
- `eef_vz`
- `action_dx`
- `action_dy`
- `action_dz`
- `action_gripper`
- `recent_close_streak`
- `recent_open_streak`
- `recent_gripper_flip_count`
- `normalized_step`

## Forbidden Features

The following are forbidden as model inputs:

- object pose
- target pose
- object-to-target distance
- future done or future success signals
- attack, oracle, random, or manual outcomes
- `teacher_window_start`, `teacher_window_end`, or `teacher_anchor_step`
- `task_id`, `state_id`, `run_id`, or `episode_key`
- hard-coded windows or state-specific lookup tables

Teacher window fields may be used only for evaluation.

## Split Protocol

Default split mode: stratified `task_id` split.

- Train episodes: 280
- Val episodes: 80
- Test episodes: 40

Timestep-random splitting is not used.

## Main Results

Training:

- Device: CPU
- Best epoch: 30
- Best validation hazard F1: 0.6823
- Best validation phase macro F1: 0.6429

Overall test metrics:

- Phase accuracy: 0.8791
- Phase macro F1: 0.7286
- Hazard accuracy: 0.9723
- Hazard precision: 0.7601
- Hazard recall: 0.6059
- Hazard F1: 0.6743
- Hazard AUROC: 0.9874
- Hazard AUPRC: 0.7310

Threshold sweep:

- Coverage-first: hazard >= 0.1 and release_safe < 0.3 gives window coverage 0.8706, false early trigger rate 0.0116, and miss rate 0.0061.
- Conservative: hazard >= 0.5 and release_safe < 0.5 gives window coverage 0.6059, false early trigger rate 0.0014, and miss rate 0.0187.

## Limitations

- Proprio-only; no visual features are used.
- No detector-triggered attack rollout was run.
- No VIS, oracle, random, or manual outcomes were used.
- Object remains a clean reproducibility gap and must not support strong attack claims yet.
- The release-safe head is weak and needs label or threshold review.
- Image path and visual feature linkage remain missing.
- The trained model is not final attack evidence.

## Next Steps

- Freeze this result as the Milestone 2C baseline.
- Compare the model against offline streaming replay.
- Add a time-only baseline.
- Run an ablation without `normalized_step`.
- Capture or extract visual features for Milestone 2D.
- Consider a small attack pilot only after offline replay validation.

## Boundary Statement

Milestone 2C trains and evaluates a proprio-only causal student baseline for detector development.
It does not use visual features.
It does not run detector-triggered attack rollouts.
It does not use attack/oracle/random/manual outcomes.
The trained model is not final attack evidence; it is a candidate online phase detector for offline validation.
