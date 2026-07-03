# Attack Protocol V1

Status: PLANNING_ONLY

No attack rollout is authorized by this document.

## Frozen Before Execution

```text
victim_model_sha
detector_checkpoint_sha
dataset_split_sha
preprocessing_sha
epsilon
epsilon_space
pgd_steps
step_size
K
target_token_or_objective
trigger_policy
fallback_policy
arm_gate
random_seed_semantics
exact_prefix_state_identity
```

## Primary Conditions

- Clean exact-prefix replay
- Ours: detector-triggered gripper target
- RAND_DIRECTION: same timing, random direction, same epsilon and K
- RANDOM_TIME: same payload, random time, same epsilon and K

## Mechanism Controls

- EARLY_SHIFT
- ARM_TARGETED
- COMMAND_OPEN_ORACLE

## Required Runtime Telemetry

```text
attack_applied_actual
attacked_frame_indices
delta_linf_per_frame
delta_l2_per_frame
nonzero_delta_frame_count
target_loss
gripper_command_raw
gripper_command_executed
gripper_qpos
gripper_width
arm_action_l2_vs_clean
prefix_hash
simulator_state_hash
```

