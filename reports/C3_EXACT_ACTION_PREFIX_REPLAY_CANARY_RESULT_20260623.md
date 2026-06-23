# C3 Exact Action-Prefix Replay Canary Result

```text
STAGE: C3_EXACT_ACTION_PREFIX_REPLAY_ONE_STEP
RESULT: PASS
DATE: 2026-06-23
HEAD: ec410e947f3516f266a645ad11e9064f0be7bc6b
PR: #39
GPU_RUN: CLEAN-ONLY INFRASTRUCTURE CANARY
VIS_RAND_SHUFFLED_ATTACK: NOT_RUN
```

## Scope

This result validates the C3 exact action-prefix replay infrastructure on one authorized development parent:

```text
parent: libero_goal|4|1|0|CLEAN
model: /data/aviary/models/openvla/openvla-7b-finetuned-libero-goal
env: /data/aviary/envs/openvla_official_libero_20260525
CUDA_VISIBLE_DEVICES: 1,3
render physical GPU: 1
```

It does not run VIS, RAND, shuffled, oracle, attack, or closed-loop attack evaluation.

## Output Roots

```text
successful output:
/data/liuyu/layer3_outputs/exact_action_prefix_replay_goal_t4_s1_ec410e9_gpu13_20260623_163424

successful preflight:
/data/liuyu/layer3_outputs/c3_preflight_ec410e9_gpu13_20260623_163424

preserved first infra attempt:
/data/liuyu/layer3_outputs/exact_action_prefix_replay_goal_t4_s1_50dcc9b_gpu13_20260623_162629

first-attempt preflight:
/data/liuyu/layer3_outputs/c3_preflight_50dcc9b_gpu13_20260623_162629
```

The first attempt at `50dcc9b` stopped before scientific comparison with CUDA OOM while loading the replay model after the reference model. The repair in `ec410e9` releases the reference policy/env before loading replay policy.

## Primary Result

```text
top_result: PASS
prefix_steps_completed: 51
branch_step: 51
first_divergence: null

branch_action_tokens_exact: true
branch_raw_action_exact: true
branch_env_action_exact: true
branch_student_emit_exact: true

post_branch_qpos_exact: true
post_branch_qvel_exact: true
post_branch_sim_state_exact: true
post_branch_observation_exact: true

root_recursive_sha256_manifest_sha256:
3492e20a44378b8650184e39e52354955f50a2490ce396edaecbe41f78ecadea

c3_recursive_sha256_manifest_sha256:
7651ed78f330a7d783de36b4d718421a392c92d965f18012132eb5153d7e4243
```

The artifact audit found no missing required C3 files in the root recursive seal.

## Required Artifacts

The root recursive manifest includes:

```text
run_manifest.json
GPU_before.txt
GPU_after.txt
dmesg_before.txt
dmesg_after.txt
exact_action_prefix_replay_canary/run_manifest.json
exact_action_prefix_replay_canary/dummy_wait_trace.jsonl
exact_action_prefix_replay_canary/original_prefix_trace.jsonl
exact_action_prefix_replay_canary/replay_prefix_trace.jsonl
exact_action_prefix_replay_canary/prefix_replay_step_diff.csv
exact_action_prefix_replay_canary/prefix_replay_first_divergence.json
exact_action_prefix_replay_canary/branch_boundary_manifest.json
exact_action_prefix_replay_canary/branch_action_exactness.json
exact_action_prefix_replay_canary/post_branch_diff.csv
exact_action_prefix_replay_canary/c3_prefix_replay_summary.json
```

## GPU State

Post-run `nvidia-smi` showed no active GPU processes and 0 MiB used on all visible physical GPUs. The post-run Xid tail contained only historical entries from 2026-06-20 and 2026-06-21; no new Xid was observed during this C3 retry.

## Validation

Local validation before server retry:

```text
python -m compileall -q scripts/stageb tests/stageb
PYTHONPATH=. pytest tests/stageb/test_layer3_exact_restore_runner.py -q
83 passed
PYTHONPATH=. pytest tests/stageb -q
271 passed, 8 warnings
```

Remote CI:

```text
stageb-cpu: success
https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/actions/runs/28013174763/job/82911021396
```

## Allowed Claims

```text
C3 exact action-prefix replay passed on the authorized single development parent.
The replay used recorded env_action prefix steps without policy calls before the branch.
At the branch step, raw action, env action, generated tokens, Student emit status, and post-branch state hashes matched exactly.
The required C3 evidence artifacts were written and sealed.
```

## Forbidden Claims

```text
VIS effectiveness
VIS > RAND
Layer3 attack success
closed-loop attack success
detector-triggered attack success
general LIBERO transfer
formal restore R2 success
A800 migration success
```

## Next Gate

Stop for external audit of PR #39 and the C3 server artifacts before any broader restore qualification, A800 migration, or attack-stage execution.
