# Train300 Human Gate Approval

```text
APPROVE_TRAIN300_FINAL_ACCEPTANCE
TRAIN300_FINAL_ACCEPTANCE = APPROVED
TRAIN300_CLASSIFICATION = PASS_300_VALID
TRAIN300_CORPUS = CROSS_SUITE_CLEAN_TRAIN300_S10_19
TRAIN300_SPLIT = FROZEN
```

Frozen denominator:

```text
planned = 300
valid primary = 300
clean success = 206
clean failure = 94
states 10-17 = training
states 18-19 = validation
states 0-9 = frozen CLEAN300 test
```

Mandatory hardened audit counters:

```text
identity_mismatch_count = 0
clean_contract_failure_count = 0
invalid_feature_episode_count = 0
sim_manifest_step_mismatch_count = 0
sim_array_missing_count = 0
sim_array_length_mismatch_count = 0
media_decode_failure_count = 0
artifact_manifest_coverage_failure_count = 0
artifact_sha_mismatch_count = 0
```

Retry lineage:

```text
failed key = libero_goal|1|11|0|CLEAN
failed pair = GPU (2,6), quarantined
retry pair = GPU (1,3)
retry count = exactly once
retry primary output =
/data/liuyu/outputs/cross_suite_clean_train300_s10_19_20260620/goal_t01_s11_infra_retry_v1_gpu13_20260620/libero_goal_t01_s11
```

Permanent constraints:

```text
do_not_recollect_train300 = true
do_not_retry_train300 = true
do_not_modify_master_manifest = true
do_not_modify_frozen_split = true
preserve_gpu2_failed_attempt = true
preserve_retry_lineage = true
keep_clean_failures_in_denominator = true
layer2_training_from_this_gate = NO_GO
VIS_RAND_SHUFFLED_ORACLE_ATTACK = NO_GO
```

This approval records corpus acceptance only. It does not approve Layer2
training/evaluation, detector telemetry inspection for Teacher decisions, full
CLEAN300 resolver execution, GPU/LIBERO collection, VIS/RAND/shuffled/oracle, or
attack experiments.
