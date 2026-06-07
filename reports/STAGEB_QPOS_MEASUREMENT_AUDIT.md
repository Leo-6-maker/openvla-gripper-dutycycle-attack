# Stage-B Qpos Measurement Audit

**Date**: 2026-06-07
**Status**: qpos field confirmed broken — physical_response_label BLOCKED

## Findings

All 194 trace CSVs have `gripper_qpos` = 0.500000 for every step.
This is a measurement bug, not a zero physical response.

## Root Cause

The labeling script `run_stageb_vis_labeling.py` computes:

```python
qpos = env.sim.data.qpos
gripper_qpos = float((qpos[-2] + qpos[-1]) / 2.0)
```

The indices `[-2]` and `[-1]` are WRONG for LIBERO's MuJoCo qpos array.
They point to non-gripper joint(s) with constant value 0.5, not the actual
finger joints.

## Impact

- `qpos_delta` = 0.0 for ALL 75 paired summaries
- `physical_response_label` cannot be computed
- `has_qpos_response` is always 0

## Fix Needed

1. Identify correct gripper joint indices for LIBERO MuJoCo qpos array
2. Or use `gripper_width` from `env.sim.data.qpos` with correct index
3. Recompute qpos_delta from trace CSVs after fix

## Action

- **physical_response_label = BLOCKED_QPOS_MEASUREMENT**
- Do NOT train physical_response model until qpos is fixed
- command_susceptible_label is NOT affected (uses decoded_open_count, not qpos)
