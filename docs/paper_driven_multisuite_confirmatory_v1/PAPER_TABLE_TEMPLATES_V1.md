# Paper Table Templates V1

Status: PLANNING_ONLY

Each non-header row must map to `docs/paper_driven_multisuite_confirmatory_v1/EXPERIMENT_MATRIX_V2.csv`.

## Table 1: Cross-Suite Main Attack Results

| experiment_id | Suite | Method | eligible n | emit coverage | ITT CQFR | Official FR | delta open duty | Arm NAD | Linf actual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| P4_OBJECT_CLEAN | Object | Clean |  |  |  |  |  |  |  |
| P4_OBJECT_OURS | Object | Ours |  |  |  |  |  |  |  |
| P4_OBJECT_RAND_DIRECTION | Object | RAND_DIRECTION |  |  |  |  |  |  |  |
| P4_OBJECT_RANDOM_TIME | Object | RANDOM_TIME |  |  |  |  |  |  |  |
| P4_OBJECT_TMA_OPEN | Object | Adapted TMA-OPEN |  |  |  |  |  |  |  |
| P4_SPATIAL_CLEAN | Spatial | Clean |  |  |  |  |  |  |  |
| P4_SPATIAL_OURS | Spatial | Ours |  |  |  |  |  |  |  |
| P4_SPATIAL_RAND_DIRECTION | Spatial | RAND_DIRECTION |  |  |  |  |  |  |  |
| P4_SPATIAL_RANDOM_TIME | Spatial | RANDOM_TIME |  |  |  |  |  |  |  |
| P4_SPATIAL_TMA_OPEN | Spatial | Adapted TMA-OPEN |  |  |  |  |  |  |  |
| P4_GOAL_CLEAN | Goal | Clean |  |  |  |  |  |  |  |
| P4_GOAL_OURS | Goal | Ours |  |  |  |  |  |  |  |
| P4_GOAL_RAND_DIRECTION | Goal | RAND_DIRECTION |  |  |  |  |  |  |  |
| P4_GOAL_RANDOM_TIME | Goal | RANDOM_TIME |  |  |  |  |  |  |  |
| P4_GOAL_TMA_OPEN | Goal | Adapted TMA-OPEN |  |  |  |  |  |  |  |
| P4_LIBERO10_CLEAN | LIBERO-10 | Clean |  |  |  |  |  |  |  |
| P4_LIBERO10_OURS | LIBERO-10 | Ours |  |  |  |  |  |  |  |
| P4_LIBERO10_RAND_DIRECTION | LIBERO-10 | RAND_DIRECTION |  |  |  |  |  |  |  |
| P4_LIBERO10_RANDOM_TIME | LIBERO-10 | RANDOM_TIME |  |  |  |  |  |  |  |
| P4_LIBERO10_TMA_OPEN | LIBERO-10 | Adapted TMA-OPEN |  |  |  |  |  |  |  |

## Table 2: Detector Generalization

| experiment_id | Train regime | Test suite | Positive n | No-event n | Recall | Precision/F1 | +/-10 recall | False trigger | Correct abstain | Median error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P2_OBJECT_ONLY_ELIGIBLE | Object-only | eligible |  |  |  |  |  |  |  |  |
| P2_OBJECT_ONLY_SAFETY | Object-only | safety |  |  |  |  |  |  |  |  |
| P2_OBJECT_ONLY_MULTI_EVENT | Object-only | multi-event |  |  |  |  |  |  |  |  |
| P2_POOLED_ELIGIBLE | Pooled | eligible |  |  |  |  |  |  |  |  |
| P2_POOLED_SAFETY | Pooled | safety |  |  |  |  |  |  |  |  |
| P2_LOSO_ELIGIBLE | LOSO | eligible |  |  |  |  |  |  |  |  |
| P2_LOSO_SAFETY | LOSO | safety |  |  |  |  |  |  |  |  |
| P2_OBJECT_LOTO_ELIGIBLE | Object LOTO | eligible |  |  |  |  |  |  |  |  |

## Table 3: Selectivity And Timing Mechanism

| experiment_id | Stage | Ours | RAND | Random time | TMA | Oracle | Arm preservation |
|---|---|---:|---:|---:|---:|---:|---:|
| P4_MECH_SHUFFLED_GRADIENT | shuffled gradient |  |  |  |  |  |  |
| P4_MECH_UNTARGETED_PGD | untargeted PGD |  |  |  |  |  |  |
| P4_MECH_EARLY_SHIFT | early shift |  |  |  |  |  |  |
| P4_MECH_ARM_TARGETED | arm targeted |  |  |  |  |  |  |
| P4_MECH_COMMAND_OPEN_ORACLE | command-open oracle |  |  |  |  |  |  |

## Table 4: Detector And Attack Ablations

| experiment_id | Panel | Variant | Event recall | False trigger | CQFR | Runtime | Notes |
|---|---|---|---:|---:|---:|---:|---|
| P2_OBJECT_ONLY_ELIGIBLE | Detector | fixed time / rule / linear / main / teacher |  |  |  |  |  |
| P4_MECH_SHUFFLED_GRADIENT | Attack | shuffled |  |  |  |  |  |
| P4_MECH_UNTARGETED_PGD | Attack | untargeted |  |  |  |  |  |
| P4_MECH_EARLY_SHIFT | Attack | early |  |  |  |  |  |
| P4_MECH_ARM_TARGETED | Attack | arm targeted |  |  |  |  |  |

## Table 5: Official SR Versus Contact Quality

| experiment_id | Suite | Condition | Official success and CQ success | Official success and CQ failure | Official failure and CQ failure | Official failure and CQ success | Agreement | Kappa |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P5_CONTACT_QUALITY | Object | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | Spatial | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | Goal | TBD |  |  |  |  |  |  |
| P5_CONTACT_QUALITY | LIBERO-10 | TBD |  |  |  |  |  |  |
