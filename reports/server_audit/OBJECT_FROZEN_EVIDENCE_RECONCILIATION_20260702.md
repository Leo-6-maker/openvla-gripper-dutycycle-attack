# Object Frozen Evidence Reconciliation 2026-07-02

Object results are preserved as legacy empirical evidence under a partially recovered protocol.

Revision 5 corrections:

- Raw `attack_applied` remains untrusted.
- Telemetry/key presence is not independent attack-execution evidence.
- Multi-source attack confirmation is HOLD.
- RAND_T10 final classification is `RAND_FIELD_CONFLICT_BOUNDED`.
- Object protocol source attribution now separates `value_source_*` from `binding_source_*`.

See:

- `reports/server_freeze/ATTACK_EXECUTION_FIELD_SEMANTICS_AUDIT_20260702.md`
- `reports/server_freeze/RAND_T10_DIRECTION_CONTROL_AUDIT_20260702.md`
- `reports/server_freeze/OBJECT_ACTUAL_PROTOCOL_AUDIT_20260702.md`
- `tables/server_freeze/object_frozen_master_ledger.csv`
- `tables/server_freeze/object_condition_summary.csv`
- `tables/server_freeze/object_protocol_provenance_ledger.csv`
