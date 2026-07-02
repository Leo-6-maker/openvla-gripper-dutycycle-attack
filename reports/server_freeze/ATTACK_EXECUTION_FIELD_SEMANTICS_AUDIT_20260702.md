# Attack Execution Field Semantics Audit 2026-07-02

Status: PASS_WITH_LEDGER_RENAME_REQUIRED.

Review6 correction: subgroup metrics now separate emitted, no-emission, and frames-positive denominators. `emitted_fr` is `emitted_failure / emitted`; `no_emission_fr` and `frames_positive_fr` are separate fields.

Raw writer status:

```text
RAW_ATTACK_APPLIED_WRITER = UNRECOVERABLE_AFTER_TARGETED_SEARCH
ATTACK_EXECUTION_FIELD_LINEAGE_TABLE = ATTACK_FIELD_SEARCH_INVENTORY
```

The raw `attack_applied` field is retained but not trusted. Existing schema/key-presence columns are explicitly marked by `deprecated_not_execution_evidence=True`; new aliases clarify that these are key/schema presence, not execution evidence:

- `telemetry_schema_present`
- `perturbation_norm_key_present`
- `command_override_key_present`

| condition | planned | emitted | emitted_failure | emitted_fr | no_emission | no_emission_failure | no_emission_fr | frames_positive | frames_positive_fr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | 162 | 141 | 0 | 0.000000 | 21 | 0 | 0.000000 | 0 | UNKNOWN |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 141 | 1.000000 | 0 | 0 | UNKNOWN | 141 | 1.000000 |
| EARLY_SHIFT_T10 | 141 | 99 | 4 | 0.040404 | 42 | 39 | 0.928571 | 141 | 0.304965 |
| RANDOM_TIME_V3 | 162 | 126 | 25 | 0.198413 | 36 | 18 | 0.500000 | 162 | 0.265432 |
| RAND_T10 | 162 | 141 | 0 | 0.000000 | 21 | 0 | 0.000000 | 141 | 0.000000 |
| TRUE_T10 | 162 | 141 | 141 | 1.000000 | 21 | 0 | 0.000000 | 141 | 1.000000 |

Artifacts:

- `tables/server_freeze/object_frozen_master_ledger.csv`
- `tables/server_freeze/object_condition_summary.csv`
- `tables/server_freeze/rand_t10_episode_accounting.csv`
