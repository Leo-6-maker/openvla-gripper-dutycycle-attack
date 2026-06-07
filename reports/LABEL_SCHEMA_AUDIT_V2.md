# Label Schema Audit V2

**Input**: `tables/object_phase_response_labels_v2.csv`
**Rows**: 31
**Verdict**: **PASS**

## Blocking Issues

- None.

## Warnings

- `forbidden_feature_columns_present`: label-only/outcome columns present; detector must exclude: VIS_OPEN,vis_open_count,done,denominator_clean,claim_usable

## Notes

- Only `positive` and `negative` rows are train-eligible.
- Outcome/attack fields may exist as labels or audit metadata, but detector training must not use them as inputs.
- This script does not start rollout, VIS, GPU work, or server jobs.
