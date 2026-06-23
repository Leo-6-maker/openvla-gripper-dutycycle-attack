# Real Clean Restore Observation Audit: Known Goal Emitter

## Status

```text
FROZEN_THRESHOLD_RUNTIME: PASS
RUNTIME_EVALUATOR_PARITY: PASS
ONLINE_OFFLINE_FEATURE_TRAJECTORY_PARITY: PASS
KNOWN_GOAL_ONLINE_STUDENT_EMIT: PASS
OBSERVATION_RECONSTRUCTION_AUDIT: FAIL_AT_O1
RESTORE_ACTION_IDENTITY: NOT_REACHED
VIS_RAND_SHUFFLED_ATTACK: NOT_RUN
```

## Run

```text
commit: 6143d7ad4c66b6122721e85cf6ced1c11d4787c3
branch: feature/layer3-exact-branching-runner-20260622
server_checkout: /data/liuyu/repos/l3_restore_6143d7a
output_root: /data/liuyu/layer3_outputs/real_clean_restore_obs_audit_goal_t4_s1_6143d7a_gpu13_20260622_202056
mode: --real-libero-single-parent --observation-audit-only
CUDA_VISIBLE_DEVICES: 1,3
render_gpu: 1
parent: libero_goal|4|1|0|CLEAN
emit_step: 51
recursive_sha256_manifest_sha256: b9a05c5f17a176584bb852457b003ccc0257e8a9843ed173c4b1b408ee296b2c
```

An earlier attempt under
`/data/liuyu/layer3_outputs/real_clean_restore_obs_audit_goal_t4_s1_6143d7a_gpu13_20260622_201939`
failed before model execution because the hand-written known-parent
`selection_hash` did not match the runner's manifest contract. It is an
infrastructure-invalid attempt and is not part of the observation audit
denominator.

## O1/O2/O3 Results

| Check | Meaning | Observation hash | Policy input | Key evidence |
| --- | --- | --- | --- | --- |
| O1 | Same env, no restore, direct observation recapture | FAIL | FAIL | 24 observation-field mismatches; 4 policy-input mismatches |
| O2 | Same env after one clean step, then restore snapshot | FAIL | FAIL | Same mismatch signature as O1; restored flat sim hash equals prefix |
| O3 | Fresh env restore, formal-path analogue | FAIL | FAIL | Same mismatch signature as O1/O2 |

The common flat simulator hash is:

```text
6cca224f2cb047dca9c66262973724ea0fb5e28ebcf3451085a1dc2e5990ad34
```

O1 records:

```text
sim_state_changed_by_recapture = false
agentview pixel_diff_count = 11492
agentview pixel_max_abs_diff = 104
agentview pixel_mean_abs_diff = 0.3967539469401042
policy input mismatches = raw_agentview_sha256, raw_agentview_dtype,
prepared_image_sha256, pixel_values_sha256
```

O2 records:

```text
prefix_flat_sim_state_sha256 = 6cca224f2cb047dca9c66262973724ea0fb5e28ebcf3451085a1dc2e5990ad34
restored_flat_sim_state_sha256 = 6cca224f2cb047dca9c66262973724ea0fb5e28ebcf3451085a1dc2e5990ad34
```

## Interpretation

The blocker occurs before fresh-env restore and before any branch action
identity check. Even in the same env, without restore, the observation
recaptured through the current `get_observation_after_restore()` path does not
match the prefix observation that was actually fed to the policy.

This narrows the root cause to the observation/observable/render recapture
contract:

```text
same MuJoCo flat state
same env instance
no restore
direct observation recapture
!=
prefix obs_t returned by the online rollout path
```

The current evidence does not support saying that detector generalization
failed, that restore action identity failed, or that attack execution failed.
Those stages were not reached.

## Allowed Claims

```text
The known Goal parent emits online at step 51 under the frozen runtime.
Online/offline SC5 features match through the emit step for this parent.
Observation reconstruction fails at O1, before same-env restore, fresh-env
restore, action identity, or any attack path.
The failure is consistent with an observation API/cache/observable recapture
contract mismatch rather than a Layer2 detector issue.
```

## Forbidden Claims

```text
Exact restore passed.
Restore action identity was reached.
Detector generalization failed on this parent.
VIS/RAND/shuffled/oracle/attack was run.
Layer3 attack effectiveness was evaluated.
The O1/O2/O3 diagnostic result proves or disproves VIS > random.
```

## Next Engineering Direction

Do not launch R2, VIS/RAND/shuffled, or full restore attempts from this state.
The next repair should target the observation contract itself, for example by
separating:

1. the exact policy input captured from the online rollout path;
2. the recaptured env observation generated from simulator state;
3. the subset of observation fields required by OpenVLA;
4. any robosuite observable cache or `_get_observations()` update semantics that
   are not represented by MuJoCo flat state.

Any revised contract must be audited again on the same known parent before
formal clean restore or attack execution is re-authorized.
