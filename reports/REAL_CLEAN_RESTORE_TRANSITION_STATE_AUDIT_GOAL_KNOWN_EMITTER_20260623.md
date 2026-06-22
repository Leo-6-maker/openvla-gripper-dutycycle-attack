# Real Clean Restore Transition-State Audit: Goal Known Emitter

## Status

```text
STAGE: C1_TRANSITION_STATE_AUDIT_ONLY
COMMIT: 342bbeb3d21b67b179b97537561d7c4fd2809874
RESULT: TRANSITION_STATE_AUDIT_COMPLETE
PARENT: libero_goal|4|1|0|CLEAN
EMIT_STEP: 51
GPU_MAPPING: CUDA_VISIBLE_DEVICES=1,3
OUTPUT_ROOT: /data/liuyu/layer3_outputs/transition_state_audit_goal_t4_s1_342bbeb_gpu13_20260623_010656
```

This run was infrastructure/root-cause only. It did not run formal restore 3x,
R2, VIS, RAND, shuffled, oracle, attack execution, or A800 formal evaluation.

## Gate Result

The captured prefix still reproduces the first policy output:

```text
first_action_exact: true
first_action_tokens_exact: true
```

The transition state is already divergent before the first replayed
`env.step(action_51)`.

```text
first_divergence_phase: PRE_STEP
transition_state_classification: CONTROLLER_GOAL_STATE_MISSING
first_divergence_field: robots[0].controller.attrs.J_full.head[0]
pre_diff_count: 158
post_diff_count: 271
```

Pre-step classification counts:

```text
CONTROLLER_GOAL_STATE_MISSING: 134
MUJOCO_SOLVER_STATE_MISSING: 13
UNKNOWN_TRANSITION_STATE: 11
```

Post-step classification counts:

```text
CONTROLLER_GOAL_STATE_MISSING: 109
MUJOCO_SOLVER_STATE_MISSING: 26
UNKNOWN_TRANSITION_STATE: 136
```

The replay next observation hash differs from the reference:

```text
reference_next_observation_sha256:
813015bdc77fea6a4178846de36d38c0d886745102f74b7507b68311ddd3fc4b

replay_next_observation_sha256:
3617aa6fdf95e74e05909b12e66e3be4fc79827bc8fab11464c0e68e999fd340
```

## Interpretation

The prior blocker is not policy-input restore. The current blocker is missing
pre-step transition state inside the robosuite control stack. The state mismatch
includes controller/Jacobian/goal-state fields and MuJoCo solver acceleration
state. This supports the next engineering route:

```text
C2_ROUTE_RECOMMENDATION:
  controller/robot-control-state snapshot restoration first
  include MuJoCo qacc in the restore payload
```

It does not authorize R2, formal restore 3x, VIS/RAND/shuffled/oracle/attack, or
A800 formal cross-suite execution.

## Evidence

Key files:

```text
run/transition_state_audit/transition_state_audit_summary.json
run/transition_state_audit/transition_state_pre_diff.csv
run/transition_state_audit/transition_state_post_diff.csv
run/transition_state_audit/recursive_sha256_manifest.csv
run/single_parent_restore_qualification_summary.json
```

Artifact seal:

```text
transition_state_audit recursive_sha256_manifest_sha256:
f7b6eb59d2970893e54cddd5c624c7010fd20316e364d64e8da83c63ce908122

root recursive_sha256_manifest_sha256:
e9a81abc0c6891c1ffe20ee583c294ba595eec22f6156df297bec3737495a311
```

## Allowed Claims

```text
Captured-prefix first action/tokens remain exact for the known Goal parent.
Reference and replay transition state diverge before env.step(action_51).
The dominant pre-step divergence category is controller goal/control state.
The current root-cause route should target controller/robot-control restore,
with MuJoCo qacc included in the state payload.
```

## Forbidden Claims

```text
Clean restore solved.
R2 authorized.
VIS/RAND/shuffled/oracle/attack authorized.
Layer3 attack effectiveness established.
Formal cross-suite or A800 evaluation authorized.
```
