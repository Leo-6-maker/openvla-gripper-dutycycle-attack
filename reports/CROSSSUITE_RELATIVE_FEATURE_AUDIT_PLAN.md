# CrossSuite Relative Feature Audit Plan

## Purpose

Prepare CrossSuite-ProprioNoStep-v2 by auditing whether causal relative EEF features reduce the Object-to-Spatial/Goal distribution shift.

This is offline preparation only. It does not train a detector and does not run sus30.

## Script

`scripts/diagnostics/crosssuite_feature_transform_audit.py`

## Feature Transforms

- raw `eef_x`, `eef_y`, `eef_z`
- `relative_initial = eef_xyz - eef_xyz_initial`
- gripper qpos/width/command and action gripper convention stats

The relative transform uses only episode-initial values. It does not use future full-trajectory normalization.

## Metrics

- per-suite mean/std/min/max
- missing/NaN/zero rates
- Object-vs-suite mean absolute distance
- whether relative features reduce suite shift

## Gate

CrossSuite-v2 training should not start until:

- required proprio features and clean teacher labels are available
- relative features reduce distribution shift enough to justify a v2 detector
- Object baseline is frozen and protected
- suite-holdout and task-holdout split definitions are fixed
