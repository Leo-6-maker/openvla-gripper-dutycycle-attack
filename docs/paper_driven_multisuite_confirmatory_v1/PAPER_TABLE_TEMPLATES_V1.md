# Paper Table Templates V1

Status: PLANNING_ONLY

Each non-header row must map to `docs/paper_driven_multisuite_confirmatory_v1/EXPERIMENT_MATRIX_V1.csv`.

## Table 1: Cross-Suite Main Attack Results

| experiment_id | Suite | Method | n | ITT FR | CQFR | Official SR | No-emit | delta open duty | Arm NAD | Linf actual |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3_OBJECT_CLEAN | Object | Clean |  |  |  |  |  |  |  |  |
| P3_OBJECT_OURS | Object | Ours |  |  |  |  |  |  |  |  |
| P3_OBJECT_RAND_DIRECTION | Object | RAND |  |  |  |  |  |  |  |  |
| P3_OBJECT_RANDOM_TIME | Object | Random time |  |  |  |  |  |  |  |  |
| P4_CROSS_SUITE_ATTACK_MAIN | Spatial/Goal/LIBERO-10 | Main matrix |  |  |  |  |  |  |  |  |

## Table 2: Detector Generalization

| experiment_id | Train regime | Test suite | Positive n | No-event n | Event recall | +/-10 recall | False trigger | No-emit | Median error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| P2_DET_OBJECT_ONLY_OBJECT | Object-only | Object |  |  |  |  |  |  |  |
| P2_DET_OBJECT_ONLY_SPATIAL | Object-only | Spatial |  |  |  |  |  |  |  |
| P2_DET_OBJECT_ONLY_GOAL | Object-only | Goal |  |  |  |  |  |  |  |
| P2_DET_OBJECT_ONLY_LIBERO10 | Object-only | LIBERO-10 |  |  |  |  |  |  |  |
| P2_DET_POOLED_BY_SUITE | Pooled | each suite |  |  |  |  |  |  |  |
| P2_DET_LOSO_BY_SUITE | LOSO | held-out suite |  |  |  |  |  |  |  |

## Table 3: Selectivity And Timing Mechanism

| experiment_id | Target dimension | Timing | FR | delta gripper duty | Arm NAD | Interpretation |
|---|---|---|---:|---:|---:|---|
| P3_OBJECT_MECHANISM | Gripper | Detector window |  |  |  | Ours |
| P3_OBJECT_RAND_DIRECTION | Random direction | Same window |  |  |  | direction control |
| P3_OBJECT_RANDOM_TIME | Gripper | Random time |  |  |  | timing control |
| P3_OBJECT_MECHANISM | Gripper | Early shift |  |  |  | phase control |
| P3_OBJECT_MECHANISM | Command override | Oracle |  |  |  | physical upper bound |

## Table 4: Detector And Attack Ablations

| experiment_id | Variant | Event recall | False trigger | CQFR | Runtime | Notes |
|---|---|---:|---:|---:|---:|---|
| P5_ABLATION_DETECTOR_ATTACK | Fixed step |  |  |  |  |  |
| P5_ABLATION_DETECTOR_ATTACK | Rule-based |  |  |  |  |  |
| P5_ABLATION_DETECTOR_ATTACK | MLP |  |  |  |  |  |
| P5_ABLATION_DETECTOR_ATTACK | ProprioNoStep |  |  |  |  |  |
| P5_ABLATION_DETECTOR_ATTACK | w/o command |  |  |  |  |  |

## Table 5: Official SR Versus Contact Quality

| experiment_id | Suite | Condition | Official failure | Contact-quality failure | Missed failures | Agreement |
|---|---|---|---:|---:|---:|---:|
| P5_CONTACT_QUALITY_EVAL | Object | TBD |  |  |  |  |
| P5_CONTACT_QUALITY_EVAL | Spatial | TBD |  |  |  |  |
| P5_CONTACT_QUALITY_EVAL | Goal | TBD |  |  |  |  |
| P5_CONTACT_QUALITY_EVAL | LIBERO-10 | TBD |  |  |  |  |

