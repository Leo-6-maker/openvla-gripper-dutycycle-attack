# Provisional Layer3 Three-Suite Mainline Smoke Results

## Status

```text
STAGE: PROVISIONAL_LAYER3_THREE_SUITE_MAINLINE_SMOKE
RESULT_CLASS: ENGINEERING_PASS
PROVISIONAL_LAYER123_MAINLINE: ENGINEERING_PASS
H2_SCIENTIFIC_FREEZE: NOT_GRANTED
VIS_GT_RANDOM: NOT_ESTABLISHED
ATTACK_EFFECTIVENESS: NOT_ESTABLISHED
PAPER_CLAIMS: BLOCKED
```

This is an engineering smoke for end-to-end Layer1-to-Layer2-to-Layer3 wiring
across Spatial, Goal, and LIBERO-10 supplementary-event parents. It is not a
scientific effectiveness result.

## Inputs

Branch:

```text
feature/layer3-provisional-three-suite-mainline-smoke-20260621
```

Output root:

```text
/data/liuyu/layer3_outputs/provisional_three_suite_mainline_smoke_20260621
```

Worker:

```text
assigned_worker = PAIR_A
CUDA_VISIBLE_DEVICES = 1,5
render_gpu = 5
environment = /data/aviary/envs/openvla_official_libero_20260525
```

Frozen job manifest:

```text
reports/provisional_layer3/three_suite_mainline_smoke_manifest_20260621/provisional_layer3_three_suite_job_manifest.csv
sha256 = 769b573fda33c947d7be0dbaeba1d73f3cf941ccbf88a232cfd080215f30e71a
```

Dataset:

```text
dataset_v2_sha256 = b7a6d4bc4dd9106dba4f36e39c6e3058c7a43524f8f6fa84454594780eedecaf
```

Held-out detector checkpoints:

```text
Spatial = fd2f45f11f7ceca9c8744a9c9f74cddc7e68f01da2eb4838f05d6a802d7656ef
Goal = 2dd37f6eb68d697e6b0aecccf9f09135742917284c8ef176b65a84a0cb70da7b
LIBERO-10 = e2778bf499a83dfbe0d16826a72a5fad4141fe391d9db3b80b8507795e109f8e
```

## Planned Denominator

```text
parents = 6
conditions = CLEAN, VIS, RAND, SHUFFLED
planned_jobs = 24
```

Parent coverage:

```text
libero_spatial: 2 parents x 4 conditions = 8 jobs
libero_goal: 2 parents x 4 conditions = 8 jobs
libero_10: 2 supplementary-event parents x 4 conditions = 8 jobs
```

## Worker Result

The worker completed all planned jobs:

```text
job_count = 24
complete_count = 24
failed_count = 0
```

No new Xid was observed during this run. The only Xid entries in the monitored
tail were historical GPU2/GPU3 events from earlier runs.

## Audit Result

Audit artifacts:

```text
reports/provisional_layer3/three_suite_mainline_smoke_audit_summary_20260621.json
sha256 = 4ebd9c69269e756af4e35aff234993de8373be9281ac110de77905a345983c3d

tables/provisional_layer3/three_suite_mainline_smoke_audit_rows_20260621.csv
sha256 = 601fe4aadc6c61c0b81c47e36ed39ab8af9517eaeca0e717d1bb6e1b48242154
```

The audit reconciled all planned jobs:

```text
planned_jobs = 24
audited_jobs = 24
complete_jobs = 24
duplicate_parent_condition_keys = 0
accepted_engineering_smoke = true
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
checkpoint_sha_mismatch = 0
dataset_sha_mismatch = 0
```

Per-suite reconciliation:

```text
libero_spatial: planned 8, complete 8, audit_fail 0
libero_goal: planned 8, complete 8, audit_fail 0
libero_10: planned 8, complete 8, audit_fail 0
```

## Provisional Telemetry

This telemetry is included only to characterize the smoke run. It is not an
attack-effectiveness claim.

```text
Overall CLEAN:    success 5/6, Student trigger 3/6, attack_frames 0
Overall VIS:      success 4/6, Student trigger 3/6, attack_frames 30
Overall RAND:     success 5/6, Student trigger 3/6, attack_frames 30
Overall SHUFFLED: success 5/6, Student trigger 3/6, attack_frames 30
```

By suite:

```text
libero_spatial:
  CLEAN    success 2/2, trigger 0/2, no_emit 2/2, attack_frames 0
  VIS      success 2/2, trigger 0/2, no_emit 2/2, attack_frames 0
  RAND     success 2/2, trigger 0/2, no_emit 2/2, attack_frames 0
  SHUFFLED success 2/2, trigger 0/2, no_emit 2/2, attack_frames 0

libero_goal:
  CLEAN    success 1/2, trigger 2/2, no_emit 0/2, attack_frames 0
  VIS      success 1/2, trigger 2/2, no_emit 0/2, attack_frames 20
  RAND     success 1/2, trigger 2/2, no_emit 0/2, attack_frames 20
  SHUFFLED success 1/2, trigger 2/2, no_emit 0/2, attack_frames 20

libero_10:
  CLEAN    success 2/2, trigger 1/2, no_emit 1/2, attack_frames 0
  VIS      success 1/2, trigger 1/2, no_emit 1/2, attack_frames 10
  RAND     success 2/2, trigger 1/2, no_emit 1/2, attack_frames 10
  SHUFFLED success 2/2, trigger 1/2, no_emit 1/2, attack_frames 10
```

No-emit cases were retained and counted in the denominator.

## Allowed Claims

```text
PROVISIONAL_LAYER123_MAINLINE reached ENGINEERING_PASS for this smoke.
The three-suite Layer1-to-Layer2-to-Layer3 provisional interface ran end to end.
All 24 planned matched parent-condition jobs completed.
Spatial, Goal, and LIBERO-10 supplementary-event parents each contributed 8 jobs.
Student-only trigger contract passed.
Arm action preservation mode passed.
Telemetry length, video decode, dataset SHA, and checkpoint SHA audits passed.
No-emit cases were retained.
```

## Forbidden Claims

```text
Do not claim H2 scientific freeze.
Do not claim finalized Teacher ground truth.
Do not claim cross-suite detector generalization as a scientific result.
Do not claim VIS > RAND or VIS > SHUFFLED.
Do not claim attack effectiveness.
Do not use this as paper evidence beyond engineering pipeline wiring.
Do not treat LIBERO-10 supplementary bridge as primary single-object evidence.
```

## Next Action

```text
Stop for external review.
H2 remains NOT_GRANTED.
Layer2 scientific training/evaluation remains blocked outside this provisional engineering path.
VIS/RAND/SHUFFLED scientific attack claims remain blocked.
```
