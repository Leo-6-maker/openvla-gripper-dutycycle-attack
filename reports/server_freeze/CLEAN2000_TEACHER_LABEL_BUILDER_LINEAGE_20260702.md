# CLEAN2000 Teacher Label Builder Lineage 2026-07-02

Status: BUILDER_LINEAGE_UNRECOVERABLE_AFTER_TARGETED_SEARCH.

Review6 correction: source schema semantics are split. `source_timing_fields_present` means timing fields exist in a source row; `source_mechanism_eligible_schema_valid` is the mechanism-eligible source schema check. Explicit abstention rows are not mislabeled as invalid generic record schema.

Source semantics summary:

```json
{
  "builder_lineage_status": "BUILDER_LINEAGE_UNRECOVERABLE_AFTER_TARGETED_SEARCH",
  "canonical_timing_usability": "FAIL_NOT_AUTHORIZED",
  "crosstab": [
    {
      "cohort_class": "PRIMARY_SUCCESS_ELIGIBLE",
      "source_no_event": 271,
      "source_positive": 772,
      "total": 1043
    },
    {
      "cohort_class": "ELIGIBLE_CLEAN_FAILURE",
      "source_no_event": 276,
      "source_positive": 31,
      "total": 307
    },
    {
      "cohort_class": "MECHANISM_INELIGIBLE_ABSTENTION",
      "source_no_event": 650,
      "source_positive": 0,
      "total": 650
    }
  ],
  "source_clean_failure_no_event": 276,
  "source_explicit_abstention": 650,
  "source_mechanism_eligible_schema_valid": 1350,
  "source_no_event": 1197,
  "source_positive_anchor_valid": 803,
  "source_record_found": 2000,
  "source_timing_fields_present": 1350,
  "total": 2000,
  "unexplained_no_event": 0
}
```

CLEAN2000 remains authoritative as a census corpus, not as a timing-detector training set.
