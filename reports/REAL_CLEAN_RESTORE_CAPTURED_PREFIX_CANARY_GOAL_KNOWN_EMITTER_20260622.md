# Real Clean Restore Captured-Prefix Canary: Known Goal Emitter

## Status

```text
CAPTURED_PREFIX_OBSERVATION_CONTRACT: IMPLEMENTED_AND_SERVER_TESTED
R1_C_SINGLE_CANARY: FAIL
FAILURE_CLASS: POST_ACTION_ENV_RESTORE_FAIL
FIRST_ACTION_AND_TOKENS: PASS
POST_ACTION_TRAJECTORY_IDENTITY: FAIL
FORMAL_RESTORE_3X: NO_GO
R2: NO_GO
VIS_RAND_SHUFFLED_ATTACK: NOT_RUN
```

## Run

```text
commit: bdf2d3b1a6e3d86df2a6082438da18dbe7dd830d
branch: feature/layer3-exact-branching-runner-20260622
server_checkout: /data/liuyu/repos/l3_restore_bdf2d3b
output_root: /data/liuyu/layer3_outputs/real_clean_restore_captured_prefix_goal_t4_s1_bdf2d3b_gpu13_20260622_204519
mode: --real-libero-single-parent --captured-prefix-canary-only
CUDA_VISIBLE_DEVICES: 1,3
render_gpu: 1
parent: libero_goal|4|1|0|CLEAN
emit_step: 51
posthoc_recursive_manifest_sha256: 1e59f03c67230f84f0e97807a0218cb52a292754ee2aee0ac53bdb9d8abad82b
```

## Contract Change Tested

The canary used the revised prefix contract:

```text
branch_input_source = CAPTURED_PREFIX_OBSERVATION
```

The runner saved typed prefix artifacts:

```text
prefix_agentview.npy
prefix_agentview.png
prefix_policy_input_manifest.json
prefix_observation_diagnostic.npz
prefix_typed_observation_manifest.json
```

Typed prefix image evidence:

```text
prefix_agentview_dtype = uint8
prefix_agentview_shape = 256x256x3
prefix_agentview_sha256 = dd1a9a929277d72e6906ef44803aeb0df62e80adaa9c601dcaa06154cf9cf669
prefix_agentview_roundtrip_exact = true
captured_prefix_observation_sha256 = a3cba7189947b08c566ad6ea2f098bf928f17490cf61c7b5d48b81dbc1030116
captured_policy_input_sha256 = 4d6677dfbb7945ed37032f0d270274bbe66216839a3a6324915b3fc40640b697
```

## Result

The first policy call from the restored branch using captured `obs_51` passed:

```text
first action exact = PASS
exact 7 tokens = PASS
```

No `first action mismatch` or `first token mismatch` was raised. Therefore the
captured typed prefix observation can reproduce the clean action/token at the
branch boundary.

The canary failed immediately after executing step 51:

```text
reference_vs_replay_mismatch_count = 41
step0 observation_sha256 mismatch
step0 qpos_sha256 mismatch
step0 qvel_sha256 mismatch
step0 qpos_max_abs_diff = 0.0126966501
step0 qvel_max_abs_diff = 0.483355472
```

Step0 here is the first post-action observation row, corresponding to natural
`obs_52` after executing the saved clean action at step 51.

From step1 onward the divergence propagates into policy outputs:

```text
step1 action_sha256 mismatch
step1 token_sha256 mismatch
step1 detector_state_sha256 mismatch
step1 feature_history_sha256 mismatch
```

## Interpretation

This separates the previous blocker into two layers:

```text
captured prefix obs_t -> policy action/tokens: PASS
restored env after executing action_t -> obs_{t+1}: FAIL
```

The current failure is therefore not a policy-input or OpenVLA model restore
failure. It is a post-action environment restore failure: simulator/controller,
wrapper, or environment-internal state is still insufficient for exact
post-action trajectory identity.

This is a cleaner blocker than the earlier O1 failure:

```text
O1 old contract:
  cannot recapture obs_t from sim/env state

R1-C new contract:
  captured obs_t works for first action
  post-action env trajectory still diverges
```

## Allowed Claims

```text
Captured-prefix observation is a viable branch input for reproducing the first
clean policy action and exact 7 tokens on the known Goal parent.
Post-action exact restore is not yet achieved.
The remaining blocker is after env.step(action_t), not before policy inference.
```

## Forbidden Claims

```text
Single restore canary passed.
Formal clean restore 3x is authorized.
R2 parent expansion is authorized.
VIS/RAND/shuffled/oracle/attack was run.
Layer3 attack effectiveness was evaluated.
Detector generalization failed.
```

## Next Engineering Direction

Do not launch R2, formal 3x restore, or attack conditions. The next repair should
target environment/controller state after the branch action. Candidate areas:

1. robosuite controller/interpolator internal state;
2. mocap or action-buffer state not covered by MuJoCo flat state;
3. wrapper timestep/action-repeat state;
4. simulator state fields not included in the current capture;
5. whether `env.step()` uses hidden previous-action/controller targets.

The next diagnostic should preserve the same parent and captured-prefix input,
then instrument state immediately before and after `env.step(action_51)` in both
reference and replay.
