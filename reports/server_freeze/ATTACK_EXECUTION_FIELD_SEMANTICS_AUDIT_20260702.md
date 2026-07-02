# Attack Execution Field Semantics Audit 2026-07-02

Status: FIELD_CONFLICT_BOUNDED.

The raw `attack_applied` field is preserved in the ledger but is not trusted as the sole attack-execution field. The Object summaries report `attack_applied=False` across all six conditions, while attack-frame and telemetry evidence show condition-specific execution evidence for RAND/TRUE/RANDOM_TIME/EARLY/ORACLE. CLEAN can have `mlp_triggered=True` with `attack_frames=0`; therefore `mlp_triggered != attack_applied`.

Lineage scan scope: `/mnt/sdc/dty_user/openvla_attack` and `/mnt/sdc/dty_user/table1_sota_execution_v1`.

Lineage rows written: 301.

| condition | planned | emitted | no emission | raw attack_applied true | attack_frames positive | multi-source confirmed | field conflict | success | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | 162 | 141 | 21 | 0 | 0 | 0 | 0 | 162 | 0 |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 0 | 0 | 141 | 141 | 141 | 0 | 141 |
| EARLY_SHIFT_T10 | 141 | 99 | 42 | 0 | 141 | 141 | 141 | 98 | 43 |
| RANDOM_TIME_V3 | 162 | 126 | 36 | 0 | 162 | 162 | 162 | 119 | 43 |
| RAND_T10 | 162 | 141 | 21 | 0 | 141 | 141 | 141 | 162 | 0 |
| TRUE_T10 | 162 | 141 | 21 | 0 | 141 | 141 | 141 | 21 | 141 |

Interpretation:

- `attack_applied_raw` is a retained raw field, not a sealed semantic source.
- `attack_frames_raw > 0` is treated as attack scheduling/execution support, not by itself as proof of nonzero perturbation.
- `CONFIRMED_MULTI_SOURCE` requires `attack_frames>0` plus independent telemetry such as nonzero perturbation norm or attack telemetry.
- Rows with raw false plus independent attack evidence are marked `FIELD_CONFLICT` in `attack_semantics_status`.

Artifacts:

- `tables/server_freeze/attack_execution_field_lineage.csv`
- `tables/server_freeze/object_frozen_master_ledger.csv`
- `tables/server_freeze/object_condition_summary.csv`
