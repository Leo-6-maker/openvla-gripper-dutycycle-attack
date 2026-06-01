# VIS Arm-Drift Diagnostic Plan

## Purpose

Measure whether VIS perturbations are gripper-targeted or simply induce generic action drift.

This diagnostic must not run rollout.

## Script

`scripts/diagnostics/vis_arm_drift_sweep.py`

The script defines the output schema for gripper-effect versus arm-drift comparisons. Real decode integration must use `debug["adv_inputs"]`.

## Loss Variants

- gripper objective only
- gripper objective plus arm preservation penalty
- gripper objective plus full action L2 penalty
- random same-norm baseline

## Lambda Values

- `0.1`
- `0.3`
- `1.0`
- `3.0`

## Candidate-Worthy Gate

A VIS configuration is candidate-worthy only if:

- gripper effect is meaningful
- arm L2 is controlled
- `abs(gripper_delta) / max(arm_L2, 1e-6) >= 1.0`
- random same-norm baseline is weaker
- perturbation budget is acceptable

If this gate fails, VIS rollout remains blocked.
