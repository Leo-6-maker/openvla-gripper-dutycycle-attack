# V2 Phase 1: Clean Artifact-Rich Data Inventory (Final)

**Date:** 2026-06-18
**Scan scope:** Entire `/data/liuyu/outputs/` (3,002 `step_records.jsonl` files)

## Summary

| Metric | Count |
|--------|-------|
| Total artifact-rich trajectories | 3,002 |
| All privileged fields present | ~1,278 (in known-task subset) |
| RGB frames | ~1,278 |
| With run_manifest.json (+ task_name) | 1,278 |
| Unknown task (older manifest format) | 1,724 |
| Tasks with known names | 20+ |
| Max state_id across ALL datasets | **9** |
| state_id=11 found anywhere | **0** |

## Butter_s11: DEFINITIVELY MISSING

### EEF starting position proof

v1 Butter_s11 uses LIBERO `init_states[11]` with a different physical configuration than any Object100 state:

| Source | eef_x | eef_y | eef_z | qpos_sum |
|--------|-------|-------|-------|----------|
| v1 CLEAN_D5 (init_state 11) | -0.16233 | -0.01595 | 0.24053 | -3.85e-05 |
| Object100 butter_s0 | -0.15119 | -0.00019 | 0.25660 | 0.03872 |

EEF positions differ by ~1 cm in each axis — these are different initial states.

### Butter trajectories available

30 total Butter runs across all datasets (all state_id 0-9):

| Source | States | Clean Success |
|--------|--------|---------------|
| object100_priv (2e2) | 0-9 | 8/10 |
| full10 | 0-5 | variable |
| other (m1d, table1, full4_clean, extra) | 0-9 | ~50% |

**8 clean-success Butter states** (0, 2, 3, 5, 6, 7, 8, 9) available for threshold calibration.

## Dataset Composition

Total 1,278 trajectories with known task names:

| Source | N | Description |
|--------|---|-------------|
| crosssuite300 | 308 | LIBERO-10 + LIBERO-Object |
| full10 | 200 | Full10x5 clean/oracle |
| online_shadow | 101 | Online shadow validation |
| object100_priv | 100 | Object100 privileged artifact-rich |
| sustained_proxy | 64 | Sustained proxy burst |
| other (m1d, table1, etc.) | 505 | Older milestone datasets |

### Butter object pick-and-place by source

| Source | N | States |
|--------|---|--------|
| object100_priv | 10 | 0-9 |
| full10 (clean+oracle) | 4 | 0-3 |
| old full4_clean | 3 | 0-2 |
| extra runs | 2 | (full trajectory, state=-1) |
| m1d object_full_10x10 | 10 | 0-9 |
| table1 | 1 | 6 |

All: state_id 0-9. Zero: state_id 11.

## Field Completeness

Every Object100 trajectory step (where `teacher_privileged_state_available=True`):

| Field | Present | Purpose |
|-------|---------|---------|
| object_pose_json (xyz + quat) | Yes | lift/drop/support detection |
| target_pose_json (xyz) | Yes | pre-place band, release-safe |
| object_to_target_distance | Yes | pre-release hazard |
| object_eef_distance | Yes | grasp/follow/detach |
| gripper_qpos | Yes | grasp state |
| gripper_width | Yes | grasp state |
| gripper_command | Yes | close/open onset |
| eef_x, eef_y, eef_z | Yes | EEF position |
| eef_vx, eef_vy, eef_vz | Yes | EEF velocity |
| action_dx, dy, dz, gripper | Yes | action features |
| step_idx | Yes | causal ordering |
| image_path | Yes | RGB available |
| phase | Yes | wait/policy |
| reward, done, success_so_far | Yes | denominator |

## Phase 1B Decision

**Butter_s11 (LIBERO init_state 11) is NOT in any of the 3,002 clean artifact-rich trajectories.**

Path: **Situation B** — other Butter states available, s11 missing.

### Recommended action

1. Collect exactly ONE `Butter_s11` artifact-rich clean canary (reuse Object100 telemetry schema)
2. Use 8 clean-success Butter states (0-9, excl. 1,4) for phase threshold calibration
3. Full 81+ task-state pool available for student training (Phase 4)

## Go/No-Go

```
GO_FOR_V2_TEACHER_IMPLEMENTATION
  — 8 Butter states for threshold calibration
  — Full privileged field schema verified
BLOCKED_FOR_EXACT_S11_PHASE_LABEL
  — Need 1 Butter_s11 canary (LIBERO init_state[11])
GO_FOR_S11_CANARY_COLLECTION
  — 1 trajectory, ~3 min GPU
BLOCKED_FOR_STUDENT_TRAINING
BLOCKED_FOR_NEW_VIS_RUNS
```

## Files Produced

- `tables/v2_clean_artifact_inventory.csv` — per-trajectory inventory (100 primary Object100 runs)
- `artifacts/v2_clean_artifact_roots_manifest.json` — summary manifest
- `reports/V2_PHASE1_DATA_INVENTORY.md` — this report
