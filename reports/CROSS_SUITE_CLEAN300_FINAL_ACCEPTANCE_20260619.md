# CROSS_SUITE_CLEAN300_FINAL_ACCEPTANCE_20260619

## Decision

```text
CROSS_SUITE_CLEAN300_COLLECTION = COMPLETE
CROSS_SUITE_CLEAN300_METADATA_RECONCILIATION = PASS
CROSS_SUITE_CLEAN300_DEEP_INTEGRITY = PASS
CROSS_SUITE_CLEAN300_FINAL_ACCEPTANCE = PASS
MORE_CLEAN_COLLECTION = NO_GO
LAYER1_OFFLINE_RESOLVER = GO_AFTER_DEEP_AUDIT
VIS_RAND_ATTACK = CONTINUE_NO_GO
```

## Frozen Provenance

```text
collector_source_commit = 63793972743f667c6a6bcc12e9700f322f261147
audit_tool_commit       = 2d3ff1c02995eb8090db4e3604d6fa1ad3f7a3dd
planned episodes        = 300
valid episodes          = 300
clean success           = 197
clean failure           = 103
```

Official output roots are listed in `evidence/manifests/cross_suite_clean300_root_registry.json`.
The authorized infra retry lineage is listed in `evidence/manifests/cross_suite_clean300_retry_lineage.csv`.

## Metadata Reconciliation Gate

```text
planned = 300
unique planned = 300
unique primary discovered = 300
missing = 0
extra = 0
duplicate primary = 0
replacement states = 0
schema invalid = 0
infra failed primary = 0
```

The final metadata audit is stored on the server at:

```text
/data/liuyu/audit_outputs/cross_suite_clean_300_final_metadata_300of300_20260619_2009
```

## Deep Integrity Gate

Deep-integrity audit output:

```text
/data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447
```

Observed deep checks:

```text
COMPLETE_VALID rows = 300
artifact mismatch nonzero = 0
missing required artifacts = 0
raw video missing = 0
overlay video missing = 0
agentview NPZ missing = 0
sim-state NPZ missing = 0
```

This is a metadata and artifact-integrity acceptance. It does not create Teacher labels and does not prove detector timing transfer or VIS/RAND attack effectiveness.

## Suite Summary

See `tables/cross_suite_clean300_final_summary.csv` for the frozen suite/GPU breakdown.

Totals:

```text
libero_10      100 episodes, 43 success, 57 clean failure, 58 detector emit
libero_goal    100 episodes, 78 success, 22 clean failure, 3 detector emit
libero_spatial 100 episodes, 76 success, 24 clean failure, 3 detector emit
```

## Allowed Claims

- The cross-suite CLEAN300 corpus has 300 planned canonical clean episodes and 300 accepted primary discovered episodes.
- Clean failures are retained in the denominator.
- The corpus is source-commit consistent with collector commit `6379397`.
- All accepted episodes passed metadata reconciliation and deep artifact integrity checks listed above.

## Forbidden Claims

- Do not claim Teacher timing labels are available.
- Do not claim Layer 2 zero-shot timing transfer has passed.
- Do not claim VIS/RAND or attack effectiveness.
- Do not use raw detector emit counts as timing correctness.
- Do not start more clean collection to improve the denominator.

## Next Gate

Proceed to Layer 1 offline resolver preregistration:

```text
configs/cross_suite_task_ontology_v1.yaml
reports/CROSS_SUITE_LAYER1_RESOLVER_PREREG.md
docs/schemas/cross_suite_teacher_label_schema_v1.md
```

The resolver must not read detector emit fields, attack outcomes, or hand-picked windows.
