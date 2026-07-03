# Contact Quality Protocol V1

Status: PLANNING_ONLY

Official simulator success is a compatibility metric, not the only primary
outcome.

## Automatic CQ Failure Types

- `premature_release`
- `drop_after_lift`
- `object_eef_detach`
- `unstable_transport`
- `uncontrolled_final_drop`

These evaluation labels cannot be detector features.

## Blind Manual Review

Review:

- all automatic CQ positives;
- all Official SR / CQ disagreements;
- all Ours and Oracle failures;
- 20% random CQ negatives for every suite/method.

At least 20% of reviewed videos require a second independent reviewer. Report
Cohen's kappa. If kappa is below 0.80, Table 1 CQFR must use manual labels or
the audit sample must expand.
