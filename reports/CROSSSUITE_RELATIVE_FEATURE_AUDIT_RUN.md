# CrossSuite Relative Feature Audit Run

Date: 2026-05-31

## Input

```text
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv
```

Rows:

- 87,474 timestep rows
- 400 episode keys

## Output

```text
tables/crosssuite_relative_feature_audit.csv
```

## Key Finding

Raw `eef_z` has large Object-to-Spatial/Goal shift. Causal relative-to-initial `eef_z` sharply reduces this mean shift.

Object mean absolute distance for `eef_z`:

| Suite | Raw distance | Relative-initial distance |
| --- | ---: | ---: |
| libero_spatial | 0.8671 | 0.0466 |
| libero_goal | 0.8678 | 0.0441 |
| libero_10 | 0.6269 | 0.0449 |
| libero_object | 0.0000 | 0.0000 |

This supports the hypothesis that relative EEF-z can reduce the largest observed cross-suite coordinate shift.

## Schema Caveat

The 2B dataset has usable `eef_z`, gripper, and action features, but `eef_x`, `eef_y`, `eef_vx`, and `eef_vy` are missing across all indexed episodes. `eef_vz` is mostly present but has expected first-step gaps.

## Gate XS-1

Result: PARTIAL PASS.

Relative `eef_z` reduces the main shift, but full `relative_eef_xyz` cannot be validated from this dataset because x/y are missing. CrossSuite-v2 full training is not approved from this audit alone.
