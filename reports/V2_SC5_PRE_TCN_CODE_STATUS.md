# V2 SC5 Pre-TCN Code Status

**Branch:** exp/l2-sc5-data-census-tcn-v2-20260618
**Base:** bdccf36
**Date:** 2026-06-18

## Current Commit SHAs
- 018b15f: audit(layer12)
- c52b106: feat(layer1)
- 5d1f18d: feat(layer2)  
- bdccf36: fix(layer12)

## Current Data Source
- 81 clean-success episodes from milestone_2e2_object100_privileged_artifact_rich_20260527/runs/libero_object
- All from LIBERO-object pick-and-place tasks (10 tasks × ≤10 states)
- 100 total trajectories, 19 clean-fail, 81 clean-success
- All 81 clean-success have valid SC5 corridors
- No no-corridor episodes in current dataset

## Current Feature Schema
25D features: gripper_command, gripper_qpos, gripper_opening_proxy, eef_x/y/z/vx/vy/vz, action_dx/dy/dz/gripper, recent_close/open_streak, recent_gripper_flip_count, close_onset, time_since_close, eef_speed, eef_z_delta_since_close, qpos_delta_1, qpos_delta_3, opening_proxy_delta_3, opening_proxy_variance_5, eef_speed_variance_5

## Teacher Config
- Calibrated on 98 train paths (held-out excluded)
- SC5 rule: stable_carry_start + 5, K=10

## Current MLP Checkpoint
- 3 seeds at /data/liuyu/outputs/sc5_v4/sc5_mlp_s{1,2,3}.pt
- 6.6K params, 64 hidden dim, 25D input

## Current Gate D2
| Metric | Result | Target |
|--------|--------|--------|
| Coverage | 1.000 | ≥0.80 |
| False-early | 0.000 | ≤0.10 |
| Median abs error | 1.0 | ≤8 |
| K10 containment | 0.654 | ≥0.85 |

## Frozen Files
- src/gripper_attack/d5_frozen_*.py
- scripts/stageb/run_l3_d5_vis_temporal.py
- scripts/stageb/audit_l3_d5_vis_temporal_v3.py
- src/gripper_attack/attack_adapter.py
- Phase 3 command-hold artifacts

## Server Roots to Scan
- /data/liuyu/outputs/
- Any milestone_2* directories
- Any libero_* clean directories
