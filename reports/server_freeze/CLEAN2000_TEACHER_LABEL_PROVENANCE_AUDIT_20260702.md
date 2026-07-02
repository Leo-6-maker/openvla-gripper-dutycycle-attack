# CLEAN2000 Teacher Label Provenance Audit 2026-07-02

Status: CENSUS_READY_TIMING_TRAINING_HOLD.

CLEAN2000 cohort decomposition remains accepted:

```text
Outcome: CLEAN_SUCCESS=1392, CLEAN_FAILURE=608
Mechanism: MECHANISM_ELIGIBLE=1350, MECHANISM_INELIGIBLE=650
Cohort: PRIMARY_SUCCESS_ELIGIBLE=1043, ELIGIBLE_CLEAN_FAILURE=307, MECHANISM_INELIGIBLE_ABSTENTION=650
```

Required identity holds: 1043 + 307 + 650 = 2000.

Revision 5 source semantics:

- source_record_found: 2000
- source_positive_anchor_valid: 803
- source_no_event: 1197
- source_explicit_abstention: 650
- source_clean_failure_no_event: 276
- shared_fields_match: 0
- uncomparable_due_to_missing_fields: 2000

`canonical/source exact matches = 0/2000` is not used as a scientific mismatch claim because source confidence and event-id fields are missing. The scientifically relevant count is valid positive source timing coverage, currently `803/2000` in the extracted source table.

Official LIBERO registry reconciliation remains PASS: 40/40 exact BDDL matches.
