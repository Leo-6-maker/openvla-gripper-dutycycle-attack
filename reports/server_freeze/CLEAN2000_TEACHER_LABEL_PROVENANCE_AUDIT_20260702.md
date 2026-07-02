# CLEAN2000 Teacher Label Provenance Audit 2026-07-02

Status: CENSUS_READY_TIMING_TRAINING_HOLD.

Accepted cohort decomposition:

```text
PRIMARY_SUCCESS_ELIGIBLE = 1043
ELIGIBLE_CLEAN_FAILURE = 307
MECHANISM_INELIGIBLE_ABSTENTION = 650
```

Source event crosstab now explains all source no-event rows:

```json
[
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
]
```

Summary:

- source_record_found: 2000
- source_positive_anchor_valid: 803
- source_no_event: 1197
- unexplained_no_event: 0
- source_timing_fields_present: 1350
- source_mechanism_eligible_schema_valid: 1350

Timing detector training remains not authorized.
