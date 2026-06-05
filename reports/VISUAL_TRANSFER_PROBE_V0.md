# Visual Transfer Probe V0

**Mode**: `dummy_visual`
**Rows**: 19
**Tasks**: 7
**LOTO feasible**: true

## Metrics

| Model | Bal Acc | Macro F1 | Neg Recall | MCC | TP/FP/FN/TN | Control FP |
|---|---:|---:|---:|---:|---|---:|
| always_positive | 0.5 | 0.3667 | 0.0 | 0.0 | 11/8/0/0 | 8/8 |
| task_key_only | 0.5511 | 0.5476 | 0.375 | 0.1086 | 8/5/3/3 | 5/8 |
| phase_candidate_role_only | 0.5 | 0.3667 | 0.0 | 0.0 | 11/8/0/0 | 8/8 |
| proprio_summary_only | 0.5 | 0.3667 | 0.0 | 0.0 | 11/8/0/0 | 8/8 |
| task_plus_phase | 0.5511 | 0.5476 | 0.375 | 0.1086 | 8/5/3/3 | 5/8 |
| task_plus_proprio_summary | 0.5511 | 0.5476 | 0.375 | 0.1086 | 8/5/3/3 | 5/8 |
| dummy_visual_only | 0.5 | 0.3667 | 0.0 | 0.0 | 11/8/0/0 | 8/8 |

## Boundary

Visual branch not scientifically evaluated. Dummy visual features are pipeline smoke only.
