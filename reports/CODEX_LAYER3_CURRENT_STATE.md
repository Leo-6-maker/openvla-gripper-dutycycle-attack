# Codex Layer3 Current State

## Current Gate

```text
CODEX_SERVER: 2080Ti
CURRENT_PARENT: libero_goal|4|1|0|CLEAN
CURRENT_GATE: C2_CONTROL_STATE_CAUSAL_ABLATION_PREP
CURRENT_FAILURE: PRE_STEP_CONTROL_STACK_STATE_MISMATCH

Layer1: FROZEN_PASS
Layer2: FROZEN_ENGINEERING_PASS
Goal runtime/evaluator parity: PASS
Goal online/offline feature parity: PASS
Known Goal online emit: PASS
Captured-prefix first action/tokens: PASS
Post-action environment transition: FAIL
C1 transition-state audit: PASS

R2: NO_GO
FORMAL_RESTORE_3X: NO_GO
VIS_RAND_ATTACK: NO_GO
A800_FORMAL_CROSS_SUITE: NO_GO
```

## Frozen Evidence

```text
PR: #38
current head: 03a7a63963068c27db6ae315f0f5b71f52e6cfe7
C0_EVIDENCE_FREEZE: PASS

known parent:
libero_goal|4|1|0|CLEAN

known emit step:
51
```

Key server output:

```text
/data/liuyu/layer3_outputs/real_clean_restore_captured_prefix_goal_t4_s1_bdf2d3b_gpu13_20260622_204519
```

Key artifact hashes:

```text
posthoc_recursive_manifest_sha256:
1e59f03c67230f84f0e97807a0218cb52a292754ee2aee0ac53bdb9d8abad82b

prefix_agentview_sha256:
dd1a9a929277d72e6906ef44803aeb0df62e80adaa9c601dcaa06154cf9cf669

captured_prefix_observation_sha256:
a3cba7189947b08c566ad6ea2f098bf928f17490cf61c7b5d48b81dbc1030116

captured_policy_input_sha256:
4d6677dfbb7945ed37032f0d270274bbe66216839a3a6324915b3fc40640b697
```

## Current Failure

The captured-prefix canary reproduced the first clean policy action and exact
7-token output from captured `obs_51`, then diverged immediately after
`env.step(action_51)`.

```text
first action exact: PASS
exact 7 tokens: PASS
reference_vs_replay_mismatch_count: 41
step0 observation_sha256 mismatch
step0 qpos_sha256 mismatch
step0 qvel_sha256 mismatch
step0 qpos_max_abs_diff: 0.0126966501
step0 qvel_max_abs_diff: 0.483355472
```

This means the current blocker is after policy inference and during the
environment transition.

## C1 Transition Audit

```text
commit: 3a008fb38fa4e257e8945a967768f8c66b7d91cd
output:
/data/liuyu/layer3_outputs/transition_state_audit_goal_t4_s1_342bbeb_gpu13_20260623_010656

first_action_exact: true
first_action_tokens_exact: true
first_divergence_phase: PRE_STEP
refined_primary_blocker: PRE_STEP_CONTROL_STACK_STATE_MISMATCH
unique_root_attribute_count: 18
mutable_goal_roots: 4
derived_cache_roots: 4
qacc_roots: 1
```

The C1 result proves control-stack state mismatch before `env.step(action_51)`.
It does not prove that `J_full` is authoritative state, that controller goal is
the unique root cause, or that `qacc` must be permanently added to the restore
payload.

## Next Authorized Work

```text
C2_CONTROL_STATE_CAUSAL_ABLATION
Condition: remote CI green after moving codex_known_parent_manifest out of artifacts/
Scope: same known Goal parent only
Method: derived-cache recompute first, then strict mutable-state ablations
```

Forbidden until a later gate:

```text
candidate expansion
formal clean restore 3x
R2
VIS
RAND
shuffled
oracle
attack imports/execution
A800 formal cross-suite
```
