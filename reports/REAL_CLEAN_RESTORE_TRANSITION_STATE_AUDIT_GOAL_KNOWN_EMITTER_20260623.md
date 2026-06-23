# Real Clean Restore Transition-State Audit: Goal Known Emitter

## Status

```text
STAGE: C1_TRANSITION_AUDIT
COMMIT: 3a008fb38fa4e257e8945a967768f8c66b7d91cd
RESULT: TRANSITION_STATE_AUDIT_COMPLETE
CLASSIFICATION: C1_TRANSITION_AUDIT = PASS
PARENT: libero_goal|4|1|0|CLEAN
EMIT_STEP: 51
FIRST_DIVERGENCE_PHASE: PRE_STEP
CONTROL_STACK_STATE_PARITY: FAIL
FORMAL_RESTORE_3X: NO_GO
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
pre_diff_count: 158 leaf rows
post_diff_count: 271 leaf rows
```

The original C1 run reported the first sorted field as:

```text
robots[0].controller.attrs.J_full.head[0]
```

That field is a controller-derived cache, not proof that `J_full` itself is an
authoritative mutable state to restore. A posthoc root-attribute aggregation of
the same pre-step diff rows gives:

```text
unique_root_attribute_count: 18
CONTROLLER_MUTABLE_GOAL_STATE: 4 roots / 28 leaf rows
CONTROLLER_DERIVED_CACHE: 4 roots / 60 leaf rows
MUJOCO_DERIVED_ACCELERATION: 1 root / 13 leaf rows
UNKNOWN_TRANSITION_STATE: 9 roots / 57 leaf rows
```

Key root attributes include:

```text
robots[0].controller.attrs.goal_pos
robots[0].controller.attrs.goal_ori
robots[0].controller.attrs.J_full
robots[0].controller.attrs.J_pos
robots[0].controller.attrs.J_ori
robots[0].controller.attrs.mass_matrix
mujoco.qacc
```

The replay next observation hash differs from the reference:

```text
reference_next_observation_sha256:
813015bdc77fea6a4178846de36d38c0d886745102f74b7507b68311ddd3fc4b

replay_next_observation_sha256:
3617aa6fdf95e74e05909b12e66e3be4fc79827bc8fab11464c0e68e999fd340
```

## Interpretation

The prior blocker is not policy-input restore. The current blocker is pre-step
control-stack state mismatch inside robosuite. The state mismatch includes both
mutable controller goal fields and derived controller cache fields, plus MuJoCo
derived acceleration (`qacc`). This supports a causal ablation route, not a blind
full controller dictionary restore:

```text
C2_ROUTE_RECOMMENDATION:
  first recompute derived controller caches from restored sim state
  then ablate strict mutable controller/interpolator/action-history groups
  evaluate qacc restore as a separate ablation, not as a permanent payload change
```

It does not authorize R2, formal restore 3x, VIS/RAND/shuffled/oracle/attack, or
A800 formal cross-suite execution.

## Evidence

Key files:

```text
run/transition_state_audit/transition_state_audit_summary.json
run/transition_state_audit/transition_state_pre_diff.csv
run/transition_state_audit/transition_state_post_diff.csv
tables/real_restore_transition_state_root_attribute_summary_20260623.csv
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
PRE_STEP_CONTROL_STACK_STATE_MISMATCH is established.
CONTROLLER_GOAL_STATE_MISSING is a leading hypothesis, not final proof.
J_full and related Jacobian/mass-matrix fields are derived-cache differences.
qacc is a derived acceleration difference and must be tested as a separate ablation.
```

## Forbidden Claims

```text
Clean restore solved.
J_full proven to be an authoritative hidden state.
Controller goal proven to be the unique root cause.
Permanent qacc restore authorized.
R2 authorized.
VIS/RAND/shuffled/oracle/attack authorized.
Layer3 attack effectiveness established.
Formal cross-suite or A800 evaluation authorized.
```
