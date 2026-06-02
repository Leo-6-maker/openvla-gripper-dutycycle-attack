# Visual V2 Data and Label Inventory

**Date**: 2026-05-30 | **Branch**: `exp/visual-v2-reranker-training-20260530`

## Data Sources

### 1. Full10 Frozen Visual Features

| Property | Value |
|----------|-------|
| Path | `milestone_2f_full10_frozen_visual_features_20260530/features/full10_frozen_visual_features.npy` |
| Shape | 38,730 × 2176 |
| dtype | float32 |
| NaN | 0 |
| Inf | 0 |
| Norm range | 75.2 — 102.7 |
| Encoder | DINOv2 + SigLIP fused |
| Join map | 38,730 rows (100% coverage) |
| Tasks | All 10 LIBERO Object tasks |
| States | 5 states per task (0-4) |

### 2. Object-100 Teacher-Labeled Dataset

| Property | Value |
|----------|-------|
| Path | `milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset_labeled.csv` |
| Rows | 18,875 |
| Columns | 57 |
| teacher_hazard=0 | 18,778 (99.5%) |
| teacher_hazard=1 | 97 (0.5%) |
| visual_feature_path | **EMPTY** — linkage not populated |

### 3. Proprio Feature Schema

13-dim per step: `gripper_command`, `gripper_qpos`, `gripper_width`, `eef_x`, `eef_y`, `eef_z`, `eef_vx`, `eef_vy`, `eef_vz`, `action_dx`, `action_dy`, `action_dz`, `action_gripper`

### 4. Label Columns Available

`teacher_phase`, `teacher_hazard`, `teacher_release_safe`, `teacher_confidence`, `teacher_window_start`, `teacher_window_end`, `teacher_anchor_step`

## Data Gaps

| Gap | Severity | Resolution |
|-----|----------|------------|
| Object-100 visual_feature_path empty | HIGH | Need to cross-reference Object-100 visual features at `milestone_2e3_object100_visual_features_openvla_20260527` |
| teacher_hazard severe imbalance (97/18778) | MEDIUM | Use weighted loss or oversampling |
| Full10 features not linked to teacher labels | MEDIUM | Extract teacher labels from Full10 step records or use Object-100 labels |

## Training Data Construction Options

### Option A: Object-100 labels + Object-100 visual features
- Labels: 18,875 rows with teacher_hazard
- Features: Need to extract from `milestone_2e3_object100_visual_features_openvla_20260527`
- Risk: visual_feature_path linkage needs fixing

### Option B: Full10 features + replay teacher labels
- Features: 38,730 frozen (ready)
- Labels: Extract teacher_hazard from step_records of Full10 clean/detector-clean rollouts
- Risk: teacher label quality depends on replay accuracy

### Option C: Full10 features only (recommended for smoke test)
- Use Full10 frozen features (38,730)
- Derive labels from step_records (teacher_hazard, teacher_phase from clean rollouts)
- Smaller scale, faster iteration, already linked via join map

## Recommendation

Start with **Option C** for smoke test, then scale to Option A if visual signal shows promise.
