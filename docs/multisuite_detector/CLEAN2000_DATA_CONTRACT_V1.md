# CLEAN2000 Data Contract V1

## Source Corpus

| Suite | Target Episodes | Description |
|-------|----------------|-------------|
| LIBERO-Object (SC5) | 500 | Privileged LOTO corpus |
| LIBERO-Spatial | 500 | Cross-suite clean collection |
| LIBERO-Goal | 500 | Cross-suite clean collection |
| LIBERO-10 | 500 | Cross-suite clean collection |
| **Total** | **2000** | CLEAN2000 source corpus |

## Row Classification

Every source episode is classified into exactly one category:

| Category | Definition | Used For |
|----------|-----------|----------|
| `clean_success` | task_success=true by LIBERO check_success | Primary training |
| `clean_failure` | task_success=false | Safety/abstention evaluation |
| `schema_fail` | Missing required fields or schema violation | Excluded, counted in ledger |
| `infra_fail` | CUDA OOM, crash, incomplete trace | Excluded, counted in ledger |
| `telemetry_incomplete` | Missing steps, NaN features, non-contiguous step_idx | Excluded |
| `teacher_invalid` | No teacher label or label confidence below threshold | Abstention set |
| `mechanism_ineligible` | Unsupported mechanism type | Safety/abstention set |

## Primary Eligible Set

Must satisfy ALL:
```
clean_success = true
teacher_label_valid = true
telemetry_complete = true
schema_fail = false
mechanism_eligible = true
```

## Safety / Abstention Set

Everything in the source corpus that is legally collected but NOT in the Primary Eligible Set, excluding only schema_fail and infra_fail rows.

## Required Fields Per Episode

```
episode_key: unique identifier
parent_key: parent episode key (for paired designs)
suite: {libero_object, libero_spatial, libero_goal, libero_10}
task_id: integer
task_name: string
state_id: integer
eval_seed: integer
clean_success: boolean
mechanism_type: string
mechanism_eligible: boolean
teacher_label_valid: boolean
teacher_event_id: string | null
teacher_anchor_step: integer | null
teacher_window_start: integer | null
teacher_window_end: integer | null
teacher_confidence: float | null
feature_schema_sha256: string
source_manifest_sha256: string
artifact_inventory_sha256: string
n_steps: integer
n_valid_steps: integer
first_valid_step: integer
invalid_feature_steps: integer
```

## Forbidden Features

The following MUST NOT appear in model input features:
- normalized_step
- absolute timestep
- suite ID
- task ID
- state ID
- object identity
- teacher anchor/window
- object pose
- target pose
- object-target distance
- future timestep
- episode success/failure
- attack condition
- VIS/RAND/oracle outcome
- post-attack qpos/width
- manual outcome
