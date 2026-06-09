# Stage-B v1.1 RC1 — Ready for Reachability Scan

**Date**: 2026-06-07
**Status**: RC1 provenance locked — proceeding to v1.1 clean reachability scan

## Locked Coordinates

| Field | Value |
|-------|-------|
| Commit | `105252f83627737f9e0209bac9c7d9ebdac9cb3e` |
| Branch | `exp/vis-prefix-margin-repair-20260603` |
| SHA table | 19 rows, 0 mismatches |
| Tests | 42 passed |
| source_snapshot_id exact check | PASS |
| Spec version | `openvla_libero_exec_spec_v1_20260607` |
| Trace version | `corrected_stageb_v1_1` |

## Quarantine

All pre-v1.1 labels quarantined. Only `corrected_stageb_v1_1` traces accepted.

## Next: Clean Reachability Scan

- 9 LIBERO object tasks × 3 seeds = 27 rollouts
- No VIS, no random perturbation
- Official prompt + official image preprocessing
- max_steps: 300
- 3 GPU pairs parallel
- Output: per-step records with gripper_qpos, env_action_6, decoded_open_bool

## After Scan

- Select 3 reachable windows from scan data
- Run corrected VIS smoke (VIS PGD20 + random Linf)
- Validate with postprocess + label builder
