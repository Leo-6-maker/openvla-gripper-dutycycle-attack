# Provisional Layer3 Two-Suite Interface Smoke Retry Results

## Status

```text
STAGE: PROVISIONAL_LAYER3_TWO_SUITE_INTERFACE_SMOKE
HEAD: 7d89aea6e217d5c1562dff876163f6ae6fa32761
RESULT_CLASS: ENGINEERING_PASS_WITH_INFRA_RETRY_LINEAGE
TWO_SUITE_LAYER3_INTERFACE_SMOKE: ENGINEERING_PASS
THREE_SUITE_MAINLINE: NOT_RUN
VIS_GT_RANDOM: NOT_ESTABLISHED
ATTACK_EFFECTIVENESS: NOT_ESTABLISHED
PAPER_CLAIMS: BLOCKED
```

This result combines:

- Goal shard from the original output root.
- Spatial shard from the GPU1/5 retry root.
- The original PAIR_A GPU3/Xid failure preserved as retry lineage, not as a successful denominator.

## Output Roots

Original root:

```text
/data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621
```

Spatial retry root:

```text
/data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621_retry_spatial_gpu15_r1
```

Final denominator sources:

```text
Goal:
  /data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621/PAIR_B

Spatial:
  /data/liuyu/layer3_outputs/provisional_two_suite_interface_smoke_20260621_retry_spatial_gpu15_r1/PAIR_RETRY_SPATIAL_GPU15
```

## Retry Reason

The original Spatial shard failed on physical GPU3:

```text
failed_job = libero_spatial_3_4_0_CLEAN_shuffled
failure_class = CUDA_ILLEGAL_MEMORY_ACCESS_XID31
xid_time = 2026-06-21 12:44:29 CST
pci = 0000:08:00
gpu_uuid = GPU-c1ee1619-6791-a734-bf4c-d0d2c47df580
xid = 31
```

The retry used:

```text
CUDA_VISIBLE_DEVICES = 1,5
render_gpu = 5
retry_manifest_sha256 = efee514c2ca780a26d113baa64cef9e3f8da31e576e7be168962b1883a42f08c
```

GPU3 was not used as the C+G primary in the retry.

## Final Audit

The combined audit checked the final 16 planned parent-condition keys using Spatial retry outputs and original Goal outputs.

```text
planned_jobs = 16
audited_jobs = 16
complete_jobs = 16
duplicate_parent_condition_keys = 0
accepted_engineering_interface_smoke = true
```

All fail counts were zero:

```text
missing_summary = 0
missing_step_telemetry = 0
telemetry_length_mismatch = 0
raw_video_decode_failure = 0
overlay_video_decode_failure = 0
student_trigger_contract_failure = 0
arm_preservation_contract_failure = 0
invalid_feature_episode_count = 0
dataset_sha_mismatch = 0
checkpoint_sha_mismatch = 0
```

Audit artifacts:

```text
reports/provisional_layer3/two_suite_retry_combined_audit_summary_20260621.json
tables/provisional_layer3/two_suite_retry_combined_audit_rows_20260621.csv
```

## Interpretation

This establishes only an engineering interface smoke:

```text
Spatial and Goal Layer2-to-Layer3 rollout wiring executed for the planned matched conditions.
No-emit cases were retained.
Student-only trigger contract passed.
Arm action preservation mode passed.
Video decode and telemetry-length checks passed.
```

It does not establish attack effectiveness.

## Allowed Claims

```text
PROVISIONAL_LAYER3_TWO_SUITE_INTERFACE_SMOKE completed after a documented Spatial retry.
The final 16 planned parent-condition keys were reconciled.
Goal completed on original PAIR_B.
Spatial completed on retry GPU1/5.
The original GPU3/Xid failure is preserved as retry lineage.
```

## Forbidden Claims

```text
Do not claim three-suite mainline pass.
Do not claim PROVISIONAL_LAYER123_MAINLINE pass.
Do not claim VIS > RAND or VIS > SHUFFLED.
Do not claim attack effectiveness.
Do not claim H2 freeze.
Do not use this as paper evidence beyond engineering interface wiring.
Do not hide the original GPU3/Xid retry lineage.
```

## Next Action

Track B has produced provisional Layer2 v4 infrastructure, including a LIBERO-10 supplementary event bridge. Track C can only be considered as a new engineering smoke with the v4 detectors and a fresh preregistered output root; it must still avoid scientific claims.
