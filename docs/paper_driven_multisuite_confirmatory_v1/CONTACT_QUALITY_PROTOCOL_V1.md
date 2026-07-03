# Contact Quality Protocol V1

Status: PLANNING_ONLY

Official simulator success is a compatibility metric, not the only primary
outcome.

## Automatic CQ Failure Types

| Flag | Formula | Threshold | Phase |
|---|---|---|---|
| `premature_release` | gripper width opens while object is below lift height | width increase >= 0.015 m and object_z < lift_z_min | grasp/lift |
| `drop_after_lift` | object height falls after successful lift | object_z drop >= 0.05 m after lift_z_min reached | carry |
| `object_eef_detach` | object-to-EEF distance expands after grasp | distance increase >= 0.08 m for >= 3 steps | carry |
| `unstable_transport` | object speed/pose oscillation exceeds carry tolerance | object speed p95 >= 0.20 m/s or pose jitter >= 0.05 m | carry |
| `uncontrolled_final_drop` | object released away from target or below target height | target distance >= 0.08 m or final_z below target_z - 0.03 m | release/final |

These evaluation labels cannot be detector features.

If multiple flags are true, report all flags and use this primary priority:
`drop_after_lift`, `premature_release`, `object_eef_detach`,
`uncontrolled_final_drop`, `unstable_transport`.

Task object binding must come from the frozen task ontology. Missing object,
EEF, gripper width, or target telemetry is `CQ_TELEMETRY_MISSING` and cannot be
counted as automatic CQ success.

## Blind Manual Review

Review:

- all automatic CQ positives;
- all Official SR / CQ disagreements;
- all Ours and Oracle failures;
- 20% random CQ negatives for every suite/method.

At least 20% of reviewed videos require a second independent reviewer. Report
Cohen's kappa. If kappa is below 0.80, Table 1 CQFR must use manual labels or
the audit sample must expand.
