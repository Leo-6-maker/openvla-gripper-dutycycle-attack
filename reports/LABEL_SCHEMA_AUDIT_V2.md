# Label Schema Audit V2

**Input**: `tables/object_phase_response_labels_v2.csv`
**Rows**: 0
**Verdict**: **FAIL**

## Blocking Issues

- `input_exists`: labels CSV not found: tables/object_phase_response_labels_v2.csv

## Warnings

- None.

## Notes

- Only `positive` and `negative` rows are train-eligible.
- Outcome/attack fields may exist as labels or audit metadata, but detector training must not use them as inputs.
- This script does not start rollout, VIS, GPU work, or server jobs.
