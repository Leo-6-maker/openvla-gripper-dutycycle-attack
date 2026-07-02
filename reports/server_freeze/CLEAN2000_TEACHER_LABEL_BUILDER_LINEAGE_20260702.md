# CLEAN2000 Teacher Label Builder Lineage 2026-07-02

Status: BUILDER_LINEAGE_UNRECOVERABLE_AFTER_TARGETED_SEARCH.

Revision 5 correction: this audit found source records and candidate source timing fields, but did not recover the exact CLEAN2000 canonical label builder script or the assignment code that wrote canonical `anchor=0`, `confidence=0.5`, `window=[0,10]`.

Source semantics summary:

```json
{
  "builder_lineage_status": "BUILDER_LINEAGE_UNRECOVERABLE_AFTER_TARGETED_SEARCH",
  "canonical_timing_usability": "FAIL_NOT_AUTHORIZED",
  "shared_fields_comparable": 803,
  "source_clean_failure_no_event": 276,
  "source_explicit_abstention": 650,
  "source_no_event": 1197,
  "source_positive_anchor_valid": 803,
  "source_record_found": 2000,
  "source_schema_valid": 1350,
  "total": 2000,
  "uncomparable_due_to_missing_fields": 2000
}
```

Interpretation:

- `source_record_found=2000` means a source row exists, not that a valid positive timing label exists.
- `source_positive_anchor_valid=803` is the current count of rows with non-negative source anchor/window in the availability table.
- `uncomparable_due_to_missing_fields=2000` because source confidence/event-id fields are unavailable in the current extracted table.
- CLEAN2000 remains authoritative as a census corpus, not as a timing-detector training set.

Artifacts:

- `tables/server_freeze/clean2000_teacher_source_availability.csv`
- `reports/server_freeze/clean2000_teacher_source_semantics_summary.json`
