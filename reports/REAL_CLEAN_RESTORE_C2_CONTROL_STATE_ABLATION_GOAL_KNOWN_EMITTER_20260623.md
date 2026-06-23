# C2 Control-State Causal Ablation: Goal Known Emitter

## Status

```text
STAGE: C2_CONTROL_STATE_CAUSAL_ABLATION
COMMIT: fc369ee4d04970c1a0f159108de77a1af089637e
RESULT: C2_ONE_STEP_POST_ACTION_STILL_DIVERGES
PARENT: libero_goal|4|1|0|CLEAN
EMIT_STEP: 51
GPU_MAPPING: CUDA_VISIBLE_DEVICES=1,3
OUTPUT_ROOT: /data/liuyu/layer3_outputs/control_state_ablation_goal_t4_s1_fc369ee_gpu13_20260623_095011
ORIGINAL_C2_EVIDENCE_REPORT_COMMIT: 8fcbbd91f74e162e521582eadd1377069bb48732
```

This was a root-cause ablation only. It did not run formal restore 3x, R2,
VIS, RAND, shuffled, oracle, attack execution, or A800 formal evaluation.

## C2 Result

```text
passing_ablation_count: 0
passing_ablations: []
reference_next_observation_sha256:
813015bdc77fea6a4178846de36d38c0d886745102f74b7507b68311ddd3fc4b
recursive_sha256_manifest_sha256:
6ddd1e5bac89db3aac953b4e59d0d3fff46f1f07b42b66872c16418cf450d4e8
```

All ablations produced the same replay next-observation SHA:

```text
3617aa6fdf95e74e05909b12e66e3be4fc79827bc8fab11464c0e68e999fd340
```

and all failed the post-action qpos/qvel exact gate.

## Ablation Summary

| Ablation | Intent | Pre diff | Post diff | First class | Result |
| --- | --- | ---: | ---: | --- | --- |
| A0_BASELINE | current restore, no new state | 158 | 271 | CONTROLLER_MUTABLE_GOAL_STATE | FAIL |
| A1_DERIVED_RECOMPUTE | controller.update(force=True) only | 161 | 271 | CONTROLLER_MUTABLE_GOAL_STATE | FAIL |
| A2_GOAL_STATE | A1 + goal_pos/goal_ori | 133 | 271 | MUJOCO_DERIVED_ACCELERATION | FAIL |
| A3_GOAL_INTERPOLATOR_STATE | A2 + requested interpolator state; no applicable interpolator state was present | 133 | 271 | MUJOCO_DERIVED_ACCELERATION | FAIL |
| A4_GOAL_INTERPOLATOR_ACTION_HISTORY | A2 + action history/counters; no applicable interpolator state was present | 133 | 271 | MUJOCO_DERIVED_ACCELERATION | FAIL |
| A5_QACC_ABLATION | A4 + qacc written after refresh | 120 | 271 | CONTROLLER_DERIVED_CACHE | FAIL |

## Application Audit

Post-hoc audit of the existing C2 artifacts is recorded in:

```text
tables/real_restore_c2_control_state_ablation_application_audit_20260623.csv
```

It confirms:

```text
A2 goal fields: APPLIED_AND_MATCHED
A3 interpolator: NO_APPLICABLE_INTERPOLATOR_STATE
A4 action history/counters: APPLIED_AND_MATCHED
A4 interpolator: NO_APPLICABLE_INTERPOLATOR_STATE
A5 qacc: APPLIED_AND_MATCHED
A5 interpolator: NO_APPLICABLE_INTERPOLATOR_STATE
```

Therefore A3/A4 must not be interpreted as evidence that interpolator mutable
state was restored and failed. The reference snapshot did not contain applicable
interpolator mutable state. The exercised additions after A2 were action
history/counters in A4 and qacc in A5.

## Interpretation

Derived-cache recompute alone does not help. Restoring controller goal state
removes the mutable-goal pre-step mismatch, but the one-step physical transition
still diverges. Adding action-history/counter and explicit qacc ablation also
does not recover post-action exactness. Interpolator restoration was requested
in A3/A4/A5, but this artifact did not contain applicable interpolator mutable
state, so those rows did not exercise an additional interpolator intervention.

This narrows the root cause but does not solve clean restore. The remaining
divergence likely involves additional robosuite control stack state, lower-level
simulation/actuation state not covered by the current whitelist, or state that is
only reconstructed by replaying the action prefix through the environment.

## Gate

```text
ONE_STEP_POST_ACTION_EXACT: FAIL
FIVE_STEP_CANARY: NO_GO
FORMAL_RESTORE_3X: NO_GO
R2: NO_GO
VIS_RAND_SHUFFLED_ATTACK: NO_GO
```

Given the stop-loss policy, the next route should be reviewed before more
controller snapshot work. A likely next engineering direction is
`EXACT_ACTION_PREFIX_REPLAY`, which naturally rebuilds controller goals,
interpolator phase, previous action buffers, derived caches, and solver history
through steps 0-50.

## Allowed Claims

```text
C2 ablations ran on the known Goal parent only.
The approved whitelist route produced no passing transition.
Controller goal restore reduced pre-step mismatch but did not fix transition.
Action-history/counter restore did not recover post-action exactness.
qacc ablation did not recover post-action exactness.
Interpolator restoration was requested but did not exercise applicable state in this artifact.
```

## Forbidden Claims

```text
Clean restore solved.
All controller snapshot strategies are impossible.
Controller snapshot route proven sufficient.
qacc should be permanently restored.
Five-step canary, formal restore 3x, R2, VIS/RAND/shuffled/oracle/attack, or A800 formal execution authorized.
```
