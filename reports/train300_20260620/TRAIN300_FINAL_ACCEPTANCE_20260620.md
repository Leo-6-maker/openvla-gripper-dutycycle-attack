# Train300 Final Acceptance Report

```text
classification = PASS_300_VALID
planned = 300
primary_complete = 300
missing = 0
extra = 0
duplicate_primary_keys = 0
integrity_failure_count = 0
identity_mismatch_count = 0
clean_contract_failure_count = 0
invalid_feature_episode_count = 0
sim_manifest_step_mismatch_count = 0
sim_array_missing_count = 0
sim_array_length_mismatch_count = 0
media_decode_failure_count = 0
artifact_manifest_coverage_failure_count = 0
artifact_sha_mismatch_count = 0
clean_success = 206
clean_failure = 94
state_overlap_with_clean300_states_0_9 = 0
```

Retry key `libero_goal|1|11|0|CLEAN` uses primary output:

```text
/data/liuyu/outputs/cross_suite_clean_train300_s10_19_20260620/goal_t01_s11_infra_retry_v1_gpu13_20260620/libero_goal_t01_s11
```

Split freeze:

```text
states 10-17 = training
states 18-19 = validation
states 0-9 = frozen CLEAN300 test
```

No Layer2, VIS, RAND, shuffled, oracle, or attack execution was run by this audit.
