# CLEAN2000 Teacher Label Provenance Audit 2026-07-02

Status: CENSUS_READY_TIMING_TRAINING_HOLD.

CLEAN2000 cohort decomposition:

```json
{
  "outcome": {"CLEAN_FAILURE": 608, "CLEAN_SUCCESS": 1392},
  "mechanism": {"MECHANISM_ELIGIBLE": 1350, "MECHANISM_INELIGIBLE": 650},
  "cohort": {"ELIGIBLE_CLEAN_FAILURE": 307, "MECHANISM_INELIGIBLE_ABSTENTION": 650, "PRIMARY_SUCCESS_ELIGIBLE": 1043}
}
```

Required identity holds: 1043 + 307 + 650 = 2000.

`NON_PRIMARY_SAFETY_TOTAL=957` is a derived total only; it combines 307 eligible clean failures and 650 mechanism-ineligible abstentions and must not be treated as one label semantics class.

Official LIBERO registry reconciliation: 40/40 exact BDDL matches.

Artifacts:

- `tables/server_freeze/clean2000_episode_census.csv`
- `tables/server_freeze/clean2000_teacher_source_availability.csv`
- `tables/server_freeze/clean2000_task_registry_reconciliation.csv`
