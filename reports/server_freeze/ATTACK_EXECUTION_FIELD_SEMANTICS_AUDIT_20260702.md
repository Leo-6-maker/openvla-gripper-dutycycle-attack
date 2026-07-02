# Attack Execution Field Semantics Audit 2026-07-02

Status: ATTACK_EXECUTION_MULTI_SOURCE_CONFIRMATION_HOLD.

Revision 5 correction: telemetry/key-presence booleans are not independent perturbation evidence. CLEAN and no-emission RAND rows can contain telemetry columns and nonzero non-attack quantities while `attack_frames_raw=0`. Therefore emitted rows are no longer marked `CONFIRMED_MULTI_SOURCE`.

Raw writer status:

```text
RAW_ATTACK_APPLIED_WRITER = UNRECOVERABLE_AFTER_TARGETED_SEARCH
ATTACK_EXECUTION_FIELD_LINEAGE_TABLE = ATTACK_FIELD_SEARCH_INVENTORY
```

The raw `attack_applied` field is retained but not trusted as a semantic source. Until numeric per-step evidence is sealed (`max_linf_delta`, nonzero perturbation-frame count, attacked frame indices, actual image delta, condition-specific override, or per-step execution flag), the audit uses only frames-only support.

| condition | planned | emitted | no emission | attack_frames positive | multi-source confirmed | frames-only supported | field conflict | success | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | 162 | 141 | 21 | 0 | 0 | 0 | 0 | 162 | 0 |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 0 | 141 | 0 | 141 | 141 | 0 | 141 |
| EARLY_SHIFT_T10 | 141 | 99 | 42 | 141 | 0 | 141 | 141 | 98 | 43 |
| RANDOM_TIME_V3 | 162 | 126 | 36 | 162 | 0 | 162 | 162 | 119 | 43 |
| RAND_T10 | 162 | 141 | 21 | 141 | 0 | 141 | 141 | 162 | 0 |
| TRUE_T10 | 162 | 141 | 21 | 141 | 0 | 141 | 141 | 21 | 141 |

Artifacts:

- `tables/server_freeze/attack_execution_field_lineage.csv`
- `tables/server_freeze/object_frozen_master_ledger.csv`
- `tables/server_freeze/object_condition_summary.csv`
