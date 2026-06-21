# Provisional Layer3 Three-Suite Mainline Smoke Preregistration

## Status

```text
STAGE: PROVISIONAL_LAYER3_THREE_SUITE_MAINLINE_SMOKE
STATUS: PREREGISTERED_ENGINEERING_ONLY
H2: NOT_GRANTED
PAPER_CLAIMS: BLOCKED
VIS_GT_RANDOM: NOT_ESTABLISHED
ATTACK_EFFECTIVENESS: NOT_ESTABLISHED
```

This smoke is authorized only because the provisional LIBERO-10 supplementary event bridge reached a Layer2 engineering pass. It tests whether the provisional three-suite Layer1-to-Layer2-to-Layer3 wiring can execute end to end. It is not a scientific H2 freeze and not an attack-effectiveness result.

## Inputs

Parent manifest:

```text
reports/provisional_layer3/three_suite_mainline_smoke_manifest_20260621/provisional_layer3_three_suite_parent_manifest.csv
```

Job manifest:

```text
reports/provisional_layer3/three_suite_mainline_smoke_manifest_20260621/provisional_layer3_three_suite_job_manifest.csv
```

Output root:

```text
/data/liuyu/layer3_outputs/provisional_three_suite_mainline_smoke_20260621
```

Sentinel:

```text
PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS
```

## Parent Denominator

The planned denominator is six parents:

```text
libero_spatial: 2 primary single-object parents
libero_goal: 2 primary single-object parents
libero_10: 2 supplementary multi-event grasp-carry bridge parents
```

The LIBERO-10 parents are consumed DEV_CANARY / DIAGNOSTIC_HOLDOUT rows joined to the supplementary bridge labels:

```text
libero_10|1|5|0|CLEAN
libero_10|1|8|0|CLEAN
```

No untouched confirmatory or final-blind rows are used.

## Conditions

Each parent runs:

```text
CLEAN
VIS
RAND
SHUFFLED
```

Total planned jobs:

```text
6 parents x 4 conditions = 24 jobs
```

## Detectors

The smoke uses the v4 held-out-suite detectors:

```text
Spatial held-out detector:
  /data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v4_libero10_bridge_cpu_r1/M2_leave_one_suite_out_test_libero_spatial/model.pt

Goal held-out detector:
  /data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v4_libero10_bridge_cpu_r1/M2_leave_one_suite_out_test_libero_goal/model.pt

LIBERO-10 held-out detector:
  /data/liuyu/layer2_outputs/provisional_cross_suite_20260621/mlp_matrix_v4_libero10_bridge_cpu_r1/M2_leave_one_suite_out_test_libero_10/model.pt
```

## GPU Policy

```text
GPU2: QUARANTINED
GPU3: NOT USED AS C+G PRIMARY
CUDA_VISIBLE_DEVICES: 1,5
render_gpu: 5
```

This uses a single worker to avoid conflicting ownership of GPU5 and to avoid GPU3 after the previous Xid 31.

## Acceptance

Declare:

```text
PROVISIONAL_LAYER123_MAINLINE = ENGINEERING_PASS
```

only if:

```text
planned_jobs = 24
complete_jobs = 24
failed_jobs = 0
duplicate parent-condition keys = 0
student trigger contract passes
arm preservation contract passes
video decode passes
telemetry length checks pass
v4 detector checkpoint SHA checks pass
no GPU Xid occurs during the run
```

This declaration would mean only:

```text
The provisional three-suite Layer1-to-Layer2-to-Layer3 pipeline runs end to end.
```

It would not mean:

```text
H2 scientifically frozen
Teacher ground truth finalized
cross-suite generalization scientifically confirmed
VIS superior to controls
attack effectiveness established
```
