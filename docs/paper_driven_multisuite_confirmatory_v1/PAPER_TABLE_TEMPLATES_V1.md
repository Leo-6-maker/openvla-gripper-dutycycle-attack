# Paper Table Templates V1

Status: PLANNING_ONLY

Each non-header row must map to `docs/paper_driven_multisuite_confirmatory_v1/EXPERIMENT_MATRIX_V2.csv`.

## Table 1: Cross-Suite Main Attack Results

| experiment_id | Suite | Method | eligible n | emit coverage | ITT CQFR | Official FR | delta open duty | Arm NAD | Linf actual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| P4_MAIN_OBJECT | Object | Clean/RAND/RANDOM_TIME/TMA/Ours |  |  |  |  |  |  |  |
| P4_MAIN_SPATIAL | Spatial | Clean/RAND/RANDOM_TIME/TMA/Ours |  |  |  |  |  |  |  |
| P4_MAIN_GOAL | Goal | Clean/RAND/RANDOM_TIME/TMA/Ours |  |  |  |  |  |  |  |
| P4_MAIN_LIBERO10 | LIBERO-10 | Clean/RAND/RANDOM_TIME/TMA/Ours |  |  |  |  |  |  |  |
| P4_MAIN_OBJECT/P4_MAIN_SPATIAL/P4_MAIN_GOAL/P4_MAIN_LIBERO10 | Macro | Macro average |  |  |  |  |  |  |  |

## Table 2: Detector Generalization

| experiment_id | Train regime | Test suite | Positive n | No-event n | Recall | Precision/F1 | +/-10 recall | False trigger | Correct abstain | Median error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2_DET_OBJECT_ONLY | Object-only | each suite |  |  |  |  |  |  |  |  |
| P2_DET_POOLED | Pooled | each suite |  |  |  |  |  |  |  |  |
| P2_DET_LOSO | LOSO | held-out suite |  |  |  |  |  |  |  |  |

## Table 3: Selectivity And Timing Mechanism

| experiment_id | Stage | Ours | RAND | Random time | TMA | Oracle | Arm preservation |
|---|---|---:|---:|---:|---:|---:|---:|
| P4_MECHANISM_SUBSET | detector emit |  |  |  |  |  |  |
| P4_MECHANISM_SUBSET | target objective achieved |  |  |  |  |  |  |
| P4_MECHANISM_SUBSET | executed OPEN duty |  |  |  |  |  |  |
| P4_MECHANISM_SUBSET | qpos/width response |  |  |  |  |  |  |
| P4_MECHANISM_SUBSET | detach/drop |  |  |  |  |  |  |
| P4_MECHANISM_SUBSET | CQ failure |  |  |  |  |  |  |

## Table 4: Detector And Attack Ablations

| experiment_id | Panel | Variant | Event recall | False trigger | CQFR | Runtime | Notes |
|---|---|---|---:|---:|---:|---:|---|
| P2_DET_OBJECT_ONLY | Detector | fixed time |  |  |  |  |  |
| P2_DET_OBJECT_ONLY | Detector | rule |  |  |  |  |  |
| P2_DET_OBJECT_ONLY | Detector | linear/main/teacher |  |  |  |  |  |
| P4_MECHANISM_SUBSET | Attack | RAND/shuffled/untargeted/TMA/ours/early/arm |  |  |  |  |  |

## Table 5: Official SR Versus Contact Quality

| experiment_id | Suite | Condition | Official success and CQ success | Official success and CQ failure | Official failure and CQ failure | Official failure and CQ success | Agreement | Kappa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P5_CONTACT_QUALITY | Object | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | Spatial | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | Goal | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | LIBERO-10 | TBD |  |  |  |  |  |  |
