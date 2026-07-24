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

`exact_prefix_state_identity` is defined in `EXACT_PREFIX_BRANCHING_SPEC_V1.md`.

## Primary Conditions

- Clean exact-prefix replay
- Ours: detector-triggered gripper target
- RAND_DIRECTION: same timing, random direction, same epsilon and K
- RANDOM_TIME: same payload, random time, same epsilon and K
- Adapted TMA-OPEN: same victim, epsilon, K, prefix, preprocessing, denominator

## Mechanism Controls

- EARLY_SHIFT
- ARM_TARGETED
- COMMAND_OPEN_ORACLE
- SHUFFLED_GRADIENT
- UNTARGETED_PGD

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

## Parameter Freeze

Default formal values are:

```text
K = 10
PGD steps = 20
one global preprocessing backend
one global target objective
one global threshold
one global attack parameter set across suites
```

Epsilon may be selected once on an independent calibration split from
`{2/255, 4/255, 6/255}`. Selection must use the smallest epsilon satisfying the
pre-registered gripper-duty effect, weak matched RAND, arm NAD ceiling, and
complete actual-Linf telemetry. Test suites cannot retune epsilon.

The main experiment uses gripper-only loss and reports actual arm NAD. Arm lock
is ablation-only.

## Random-Time Rule

Before results are inspected, select a same-episode, same-K, same-horizon,
mechanism-legal random window using `EXACT_PREFIX_BRANCHING_SPEC_V1.md`. If no
legal window exists, label the branch `RANDOM_TIME_INELIGIBLE`.
