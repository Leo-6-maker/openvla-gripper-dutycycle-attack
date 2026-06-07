# Fast VIS Policy-Only Audit v0

**Date**: 2026-06-05
**Status**: BLOCKED_MISSING_CACHED_OBS (8/8 candidates)

## Summary

Policy-only VIS audit targets verifying whether image PGD can flip OpenVLA gripper
to OPEN without stepping the LIBERO environment. This is a fast screening tool.

## Results

| Result | Count |
|--------|-------|
| OK | 0 |
| BLOCKED_MISSING_CACHED_OBS | 8 |

All 8 calibration candidates blocked: no cached RGB observation frames
available in existing Batch3/Batch3b trace CSVs.

## Root Cause

- Trace CSV files do not contain  column
- Manifest files do not contain  or 
- Existing outputs do not save agentview RGB frames as PNG

## Model-load validation

- Model: 
- GPU pair: 2,6 (device_map='auto', split 9GiB each)
- Load time: 13.5s
- Inference: OK
- PGD attack adapter: OK (19.3s per 10-step PGD)

## Proposed fix

In future clean/VIS rollouts, add:
1. Save agentview_image as PNG: 
2. Write  column in trace CSV
3. Write  /  in manifest

No fake results generated. This is a CORRECT blocker.
