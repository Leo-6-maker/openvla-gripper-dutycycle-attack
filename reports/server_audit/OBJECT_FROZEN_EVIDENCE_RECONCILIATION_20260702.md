# Object Frozen Evidence Reconciliation 2026-07-02

Object results are preserved as legacy empirical evidence under a partially recovered protocol.

Review6 corrections:

- subgroup failure rates use correct denominators;
- master ledger key/schema fields are explicitly deprecated as execution evidence;
- `attack_execution_not_applied` is reported in the condition summary;
- weak negative-search claims are demoted to `UNRESOLVED_NO_FORMAL_BINDING_EVIDENCE`;
- checkpoint/dataset SHA sets are recorded in `object_protocol_sha_sets.csv`.

See:

- `reports/server_freeze/ATTACK_EXECUTION_FIELD_SEMANTICS_AUDIT_20260702.md`
- `reports/server_freeze/RAND_T10_DIRECTION_CONTROL_AUDIT_20260702.md`
- `reports/server_freeze/OBJECT_ACTUAL_PROTOCOL_AUDIT_20260702.md`
