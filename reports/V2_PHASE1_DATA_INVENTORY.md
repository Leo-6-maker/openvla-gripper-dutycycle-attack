# V2 Phase 1: Clean Artifact-Rich Data Inventory

**Date:** 2026-06-18
**Source:** `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527`
**Collection date:** 2026-05-28

## Summary

| Metric | Count |
|--------|-------|
| Total runs | 100 |
| Clean success | 81 |
| Clean fail | 19 |
| All privileged fields present | 100/100 |
| RGB frames | 100/100 |
| Tasks | 10 (all LIBERO-object pick-and-place) |
| States per task | 10 (state_id 0-9) |

## Butter_s11 Status

**Butter_s11: MISSING from all existing artifact-rich clean sources.**

Object100 contains Butter states 0-9 (8 clean-success, 2 fail). No other server directory contains Butter_s11 with full privileged telemetry.

Butter states available for threshold calibration:

| state | steps | success | priv | RGB |
|-------|-------|---------|------|-----|
| 0 | 151 | True | Yes | Yes |
| 1 | 290 | False | Yes | Yes |
| 2 | 208 | True | Yes | Yes |
| 3 | 243 | True | Yes | Yes |
| 4 | 290 | False | Yes | Yes |
| 5 | 177 | True | Yes | Yes |
| 6 | 235 | True | Yes | Yes |
| 7 | 160 | True | Yes | Yes |
| 8 | 180 | True | Yes | Yes |
| 9 | 150 | True | Yes | Yes |

## Field Completeness

Every Object100 trajectory contains (on steps where `teacher_privileged_state_available=True`):

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

## Classification by Task

| Task | Clean Success | States |
|------|---------------|--------|
| alphabet_soup | 8/10 | 0-9 |
| bbq_sauce | 9/10 | 0-9 |
| butter | 8/10 | 0-9 |
| chocolate_pudding | 9/10 | 0-9 |
| cream_cheese | 5/10 | 0-9 |
| ketchup | 8/10 | 0-9 |
| milk | 8/10 | 0-9 |
| orange_juice | 9/10 | 0-9 |
| salad_dressing | 9/10 | 0-9 |
| tomato_sauce | 8/10 | 0-9 |

## Phase 1B Decision

**Path: Situation B/C hybrid.**

- **Butter_s11 exact canary: MISSING** — needs one (1) artifact-rich clean collection
- **Other Butter states: AVAILABLE (8 clean-success × full privileged)** — sufficient for threshold distribution estimation
- **81 other task-state pairs: AVAILABLE** — sufficient for student training later

### Recommended action

1. Collect exactly ONE `Butter_s11` artifact-rich clean canary using the same privileged telemetry schema as Object100 (not 3-5 duplicates)
2. Use 8 clean-success Butter states for phase threshold calibration
3. Use full 81-task pool for student training (Phase 4)

## Go/No-Go

```
GO_FOR_V2_TEACHER_IMPLEMENTATION (uses existing 8 Butter states for calibration)
BLOCKED_FOR_EXACT_S11_PHASE_LABEL_UNTIL_CANARY_COLLECTED
GO_FOR_S11_CANARY_COLLECTION (1 trajectory, ~3 min GPU)
BLOCKED_FOR_STUDENT_TRAINING (Phase 4 gated behind Phase 3 command-hold)
BLOCKED_FOR_NEW_VIS_RUNS (Phase 5 gated behind Phase 4 student)
```

## Files Produced

- `tables/v2_clean_artifact_inventory.csv` — 100-row per-trajectory inventory
- `artifacts/v2_clean_artifact_roots_manifest.json` — summary manifest
- `reports/V2_PHASE1_DATA_INVENTORY.md` — this report
