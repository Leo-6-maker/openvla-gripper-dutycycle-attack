# Cross-Suite SC5 Clean Canary Results - 2026-06-19

## Status

```text
STAGE: CROSS_SUITE_CLEAN_CANARY_GATE1
COMMIT: 877c9c0f259635ccfd10d3ae18482db410a5da9d
BRANCH: feature/sc5-cross-suite-generalization-20260619
ENV: /data/aviary/envs/openvla_official_libero_20260525
VIS_RAND_EXECUTED: NO
ATTACK_EXECUTED: NO
LIBERO_ENV_STEP: CLEAN_ONLY
RESULT: CLEAN_CANARY_PASS_WITH_GPU4_INFRA_FAIL_QUARANTINED
```

This stage validates that the suite-agnostic clean collector can run
suite-matched OpenVLA checkpoints, record SC5 25D streaming features, run the
frozen Object-trained detector, and write hash-sealed artifacts. It does not
claim cross-suite generalization, attack effectiveness, VIS superiority, or
paired task degradation.

## Output Roots

Primary GPU26 root:

```text
/data/liuyu/outputs/cross_suite_clean_20260619_canary_877c9c0
```

GPU4 failed attempt root, retained as infra evidence. This is the same root as
the initial parallel canary launch; the failed worker is identified by
`logs/worker_34.log`, and only its completed `spatial_t4_s0` partial output is
present there:

```text
/data/liuyu/outputs/cross_suite_clean_20260619_canary_877c9c0
```

GPU26 retry root after GPU4 failure:

```text
/data/liuyu/outputs/cross_suite_clean_20260619_canary_retry26_after_gpu4_fail_877c9c0
```

## Clean Canary Denominator

The accepted clean denominator uses six CLEAN-only episodes. Three completed on
the initial GPU `(2,6)` worker, and three were rerun on `(2,6)` after the
parallel `(3,4)` worker failed with a GPU4 illegal-memory infra error.

| Job | Suite | Task | Steps | Success | Invalid Feature Steps | MLP Emit |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| spatial_t0_s0 | libero_spatial | 0 | 80 | True | 0 | -1 |
| goal_t1_s0 | libero_goal | 1 | 74 | True | 0 | -1 |
| libero10_t5_s0 | libero_10 | 5 | 160 | True | 0 | -1 |
| spatial_t4_s0 | libero_spatial | 4 | 130 | True | 0 | -1 |
| goal_t0_s0 | libero_goal | 0 | 107 | True | 0 | -1 |
| libero10_t0_s0 | libero_10 | 0 | 283 | True | 0 | 113 |

All six accepted episodes recorded:

```text
condition=CLEAN
vis_enabled=false
rand_enabled=false
attack_enabled=false
invalid_feature_steps=0
task_success=true
artifact_sha256.json present
privileged_valid=false
teacher_abstain=true
```

## GPU4 Infra Failure

The original `(3,4)` worker completed `spatial_t4_s0`, then failed during
`goal_t0_s0` with:

```text
RuntimeError: CUDA error: an illegal memory access was encountered
```

An Xid tail was also observed after the failure:

```text
NVRM: Xid ... 31 ... MMU Fault ... ACCESS_TYPE_VIRT_READ
```

That worker was not continued. Its partial output is retained as infra evidence
only and is not used as the accepted six-episode denominator.

## Evidence Table

See:

```text
tables/cross_suite_clean_canary_results_20260619.csv
```

The table records per-episode output directories, summary SHA256, manifest
SHA256, artifact recursive SHA256, detector dataset SHA, model path, and source
commit.

## Allowed Claims

```text
suite_agnostic_clean_collector_runs_on_selected_cross_suite_canaries
frozen_object_sc5_detector_can_be_evaluated_on_clean_cross_suite_rollouts
accepted_6_episode_clean_canary_has_zero_invalid_feature_steps
gpu4_path_failed_infra_and_was_quarantined
```

## Forbidden Claims

```text
CROSS_SUITE_GENERALIZATION_PROVEN
VIS_GT_RANDOM
ATTACK_EFFECT_ESTABLISHED
SC5_TIMING_TRANSFER_PROVEN
PHYSICAL_FAILURE_PROVEN
GPU4_QUALIFIED
```

## Next Gate

Before scaling to the full 18 clean rollouts, review the accepted artifacts and
decide whether the GPU4 failure requires changing the approved production GPU
set. No VIS/RAND or attack jobs were launched in this stage.
