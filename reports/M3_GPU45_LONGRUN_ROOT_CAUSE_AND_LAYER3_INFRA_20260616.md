# M3 GPU45 Long-Run Root Cause and Layer3 Infrastructure - 2026-06-16

## Result

```text
FINAL_CLASSIFICATION: GPU45_DEVELOPMENT_ONLY_TIER_B
REPO_START_HEAD: d4b1f7c96269d347ea892f5cc496a992238ddb39
TOOLING_COMMIT: 4f32c355bc7eeb307a42a8c7b960af621116f8ae
BRANCH: exp/m3-arm-v5-clean-close-event-panel-20260616
PRIMARY_INPUT: /data/liuyu/outputs/m3_arm_v4_panel_capture_f41ab1a_r2/step78
FINAL_OUTPUT_ROOT: /data/liuyu/outputs/m3_gpu45_longrun_root_cause_4f32c35_20260616_0300
```

This was an infrastructure/root-cause run only.  It did not run a final frozen
8-frame panel, seed428198, seed85/86, TRUE_PGD21, RAND21, SHUFFLED_GRAD21,
LIBERO env.step, or a closed-loop rollout.

## Provenance

Pre-run checks passed:

```text
local HEAD: d4b1f7c96269d347ea892f5cc496a992238ddb39
remote branch head: d4b1f7c96269d347ea892f5cc496a992238ddb39
worktree: clean
GPU4: GPU-d0a54f5d-938c-a148-fff9-c135201e3f61
GPU5: GPU-9794d733-042f-46a2-fc86-5a3fe32a158a
GPU4/5 pre-run compute processes: none
```

Existing qualification outputs and model/input manifests were frozen under:

```text
/data/liuyu/outputs/m3_gpu45_longrun_provenance_freeze_20260616_023749_r2
```

Existing Xid records in `dmesg` mapped to PCI `0000:04:00.0`, physical GPU0,
not the authorized GPU4/5 pair.

## CPU Checks

Before GPU execution:

```text
python -m py_compile scripts/stageb/run_m3_gpu45_longrun_diagnostics.py scripts/stageb/run_m3_gpu45_fixed_frame_qualification.py
pytest tests/stageb/test_m3_gpu45_longrun_diagnostics.py tests/stageb/test_m3_step78_fixed_frame_runner.py -q
```

Result:

```text
38 passed
```

## Stage B - Step78 Root-Cause Matrix

### C0 / CUDA_VISIBLE_DEVICES=4,5

Output:

```text
/data/liuyu/outputs/m3_gpu45_longrun_b123_4f32c35_20260616_024549
```

Summary:

```text
direct_score_stable: false
direct_top_stable: false
generation_tokens_stable: false
generation_gripper_stable: false
generation_score_stable: false
gradient_hash_stable: false
gradient_all_finite: true
```

Direct forward was already unstable under the original physical order.  The
direct gripper row alternated among near-boundary variants including:

```text
31872 top, target-minus-close = -0.25
31744 top/tie, target-minus-close = 0.00
31872 top, target-minus-close = -0.50
```

Generation divergence was visible from autoregressive token index 0.  The
gripper token also flipped between `31872` and `31744`.

### Fresh Process / CUDA_VISIBLE_DEVICES=4,5

Output:

```text
/data/liuyu/outputs/m3_gpu45_longrun_fresh_b4_4f32c35_20260616_024741
```

Summary:

```text
fresh_process_count: 5
direct_score_stable: true
generation_tokens_stable: false
generation_gripper_stable: false
generation_score_stable: false
gradient_hash_stable: false
gradient_all_finite: true
device_map_stable: true
```

Fresh direct forward was stable, but fresh generation and gradient hashes were
not stable.  One fresh process emitted `31744`; the others emitted `31872`.

## Stage C - GPU Order Isolation

### C2 / CUDA_VISIBLE_DEVICES=5,4

Output:

```text
/data/liuyu/outputs/m3_gpu45_longrun_c2_reversed_4f32c35_20260616_025054
```

Same-process summary:

```text
direct_score_stable: true
direct_top_stable: true
generation_tokens_stable: true
generation_gripper_stable: true
generation_score_stable: true
gradient_hash_stable: true
gradient_all_finite: true
```

Fresh-process output:

```text
/data/liuyu/outputs/m3_gpu45_longrun_c2_fresh_4f32c35_20260616_025241
```

Fresh-process summary:

```text
fresh_process_count: 3
direct_score_stable: true
generation_tokens_stable: true
generation_gripper_stable: true
generation_score_stable: true
gradient_hash_stable: true
gradient_all_finite: true
device_map_stable: true
```

Interpretation:

```text
GPU_ORDER_OR_SHARD_PLACEMENT_SENSITIVE
```

The same development step78 input failed repeatability under physical order
`4,5` but passed direct, generation, and gradient repeatability under physical
order `5,4`.  This does not prove chip defects; it only identifies the current
auto placement / physical order as a primary infrastructure confounder.

## Stage D

Forward-hook divergence localization was not run.  Reason:

```text
C2 reversed physical order stabilized direct forward and generation on the
development step78 input, so the current actionable root cause is placement
sensitivity rather than an unresolved earliest-module divergence.
```

Artifact:

```text
/data/liuyu/outputs/m3_gpu45_longrun_root_cause_4f32c35_20260616_0300/first_divergent_module.json
```

## Stage E/F

Multiframe development qualification was skipped:

```text
DEVELOPMENT_INPUT_POOL_NOT_FROZEN
```

Harness dry-run was skipped because Stage F requires at least multiframe
Tier B.  No zero-perturbation candidate, one-step gradient update, PGD21,
RAND21, or SHUFFLED_GRAD21 was run.

## Independent Audit

Auditor artifact:

```text
/data/liuyu/outputs/m3_gpu45_longrun_root_cause_4f32c35_20260616_0300/independent_audit.json
```

Summary:

```text
required_files_present: true
forbidden_scientific_runs_absent: true
forbidden_seeds_absent: true
final_panel_input_absent: true
libero_rollout_absent: true
overall: PASS_WITH_STAGE_LIMITATION
```

## Required Tables

Local committed tables:

```text
tables/m3_gpu45_same_process_forward.csv
tables/m3_gpu45_generation_repeatability.csv
tables/m3_gpu45_gradient_repeatability.csv
tables/m3_gpu45_fresh_process_repeatability.csv
tables/m3_gpu45_device_map_matrix.csv
tables/m3_gpu45_multiframe_qualification.csv
```

Server artifact root:

```text
/data/liuyu/outputs/m3_gpu45_longrun_root_cause_4f32c35_20260616_0300
```

Includes:

```text
first_divergent_module.json
harness_dryrun_audit.json
qualification_summary.json
environment_manifest.json
model_bundle_manifest.csv
device_map_manifest.json
recursive_sha256_manifest.csv
command_ledger.csv
GPU before/after snapshots
Xid before/after snapshots
```

## Allowed Claims

```text
GPU45 physical UUID binding was enforced.
The original physical order 4,5 is repeatability-unstable on development step78.
The reversed physical order 5,4 is repeatability-stable on development step78.
The observed instability is consistent with GPU order or shard placement sensitivity.
No final panel or closed-loop Layer3 run was performed.
```

## Forbidden Claims

```text
Do not claim TRUE_PGD > RAND.
Do not claim official-token attack effect.
Do not claim V5.2 scientific qualification.
Do not claim final 8-frame panel readiness.
Do not claim closed-loop Layer3 success.
Do not generalize beyond the development step78 input.
```

## Next Action

Stop for external audit.  If approved, the next infrastructure step should
freeze a production-safe explicit device-map or the reversed physical GPU order
as a new preregistered infrastructure profile, then rerun qualification on a
pre-frozen development-only input pool before any final panel or attack budget.

