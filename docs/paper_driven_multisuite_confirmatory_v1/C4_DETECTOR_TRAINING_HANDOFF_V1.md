# C4 Detector Training Handoff V1

Status: `C4_1_PARENT_RANDOM_FORMAL_TRAINING_PASS`

This document records the transition from closed data gates to detector training and separates smoke/candidate results from final paper evidence.

## Upstream closed evidence

```text
GATE_A1_LABEL_ARTIFACT = PASS
GATE_A2_FEATURE_BINDING = PASS
FORMAL_DETECTOR_DATASET_BUILD = PASS
C3_DETECTOR_TRAINING_SMOKE = PASS
C4_1_PARENT_RANDOM_FORMAL_TRAINING = PASS
```

## Frozen detector dataset identity

```text
dataset_root = /mnt/sdc/dty_user/openvla_attack_evidence/detector_dataset/formal_detector_dataset_d03560a
dataset_csv_sha256 = f7808c4ef2a74887689804758c131a19a7fecbbc0e5400bcc3322d08c796010a
split_csv_sha256 = df23607b3791e414d0e07900508c095bda6a190e8f6500502b056f0988e02673
state_index_sha256 = e4fafbb01e70418ec04b7dc19294b1f6b9c0b52ecc0d8aaa5b56997c3ba53691
```

Dataset counts:

```text
episodes = 2000
steps = 455646
DETECTOR_ELIGIBLE = 1350
DETECTOR_SAFETY = 650
```

## C4-1 parent-random detector candidate

```text
output_root = /mnt/sdc/dty_user/openvla_attack_evidence/detector_training/c4_balanced_parent_random_d03560a
population = DETECTOR_ELIGIBLE
split = parent_random_split_v1
seeds = 3
epochs = 20
batch_size = 512
best_seed = 2026070401
best_checkpoint = best_checkpoint.pt
best_checkpoint_sha256 = 5747a9c967b5b08f0e4b8fc8ba0cbf47c13533ffb5e347c38470e84efe17d79b
validation_selected_threshold = 0.95
best_val_f1 = 0.9219
best_test_f1 = 0.9372
```

Interpretation:

```text
C4-1 proves the formal detector dataset can train a valid detector and produce a loadable bundle candidate.
C4-1 is not a final paper main-table detector because it uses parent_random_split_v1.
```

## Non-actions preserved through C4-1

```text
OpenVLA = NOT_PERFORMED
LIBERO = NOT_PERFORMED
rollout = NOT_PERFORMED
attack = NOT_PERFORMED
exact_prefix_replay = NOT_PERFORMED
victim_inference = NOT_PERFORMED
paper_main_table = NOT_PERFORMED
```

## Required next gate: C4-2 bundle audit and safety calibration

C4-2 must audit the C4-1 detector bundle before any replay or attack work.

Required output root:

```text
/mnt/sdc/dty_user/openvla_attack_evidence/detector_training/c4_2_bundle_audit_d03560a
```

Required files:

```text
bundle_identity.json
checkpoint_identity.json
dataset_identity.json
threshold_identity.json
metrics_overall.json
metrics_by_suite.csv
metrics_by_task.csv
metrics_by_population.csv
safety_false_trigger_report.json
emission_rate_report.json
bundle_load_report.json
SHA256SUMS
SHA256SUMS.sha256
```

C4-2 PASS requires:

```text
checkpoint identity matches 5747a9c967b5b08f0e4b8fc8ba0cbf47c13533ffb5e347c38470e84efe17d79b
dataset identity matches the formal detector dataset hashes
bundle loads without ad-hoc code changes
threshold 0.95 reproduces reported metrics
neighbor thresholds 0.90 / 0.925 / 0.95 / 0.975 are reported
per-suite and per-task metrics are reported
DETECTOR_SAFETY false-trigger behavior is reported
no NaN/Inf
OpenVLA/LIBERO/rollout/attack remain NOT_PERFORMED
```

## After C4-2

Do not jump directly to attack replay. The next scientific training gates are:

```text
C4-3A Object task-held-out detector training
C4-3B Suite LOSO detector training
C4-3C threshold/freeze decision for the primary detector bundle
```

Only after a frozen detector bundle exists should the project move to:

```text
C5 detector-only exact-prefix replay
C6 attack/control matrix
```

## Paper interpretation guardrail

The parent-random detector can be reported as an engineering sanity check or candidate detector, not as the primary cross-task/cross-suite scientific result. The primary paper evidence must use task-held-out and/or suite-held-out splits plus replay outcomes.