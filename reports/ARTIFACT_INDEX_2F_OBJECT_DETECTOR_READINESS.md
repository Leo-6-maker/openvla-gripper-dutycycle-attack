# Artifact Index — 2F Object Detector Readiness

**Date**: 2026-05-29
**Status**: Large artifacts NOT committed to git

## Object-100 Data

| Artifact | Path | Size | SHA256 |
|----------|------|------|--------|
| Labeled dataset | `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/no_timestep_visual_proprio_student_dataset_labeled.csv` | 18,875 rows | Not committed |
| Teacher windows | `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/object100_teacher_window_labels.csv` | 100 rows | Not committed |
| Visual features | `/data/liuyu/outputs/milestone_2e3_object100_visual_features_openvla_20260527/features/` | 18,875 × 2176-dim | Not committed |
| Artifact-rich manifest | `/data/liuyu/outputs/milestone_2e2_object100_privileged_artifact_rich_20260527/tables/official_clean_artifact_rich_manifest.csv` | — | Not committed |

## Trained Models

| Model | Path | Params | SHA256 |
|-------|------|--------|--------|
| ProprioNoStep | `/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt` | 38,602 | `4b3f3d479d6bbb92b2bd15cffec0be587bf221dc81663aaff93e44afdd9c7b1f` |
| VisualNoStep | `.../models/VisualNoStep_frozen.pt` | 38,602 | Not verified |
| VisualProprioNoStep | `.../models/VisualProprioNoStep_fused.pt` | 38,602 | Not verified |

## Replay/Fusion Results

| Artifact | Path | Status |
|----------|------|--------|
| Replay comparison | `.../tables/object100_model_replay_comparison.csv` | Generated |
| Fusion calibration | `.../tables/object100_fusion_calibration_sweep.csv` | Generated |
| Best trigger policy | `.../tables/object100_best_trigger_policy.csv` | Generated |

## Shadow Validation

| Artifact | Path | Status |
|----------|------|--------|
| Shadow trigger log | `/data/liuyu/outputs/milestone_2f_object100_online_shadow_validation_20260527/shadow_logs/shadow_trigger_log.jsonl` | Generated |
| Shadow results | `.../tables/shadow_validation_results.csv` | Coverage 97.8%, all gates pass |

## GPU0,7 Rehearsal (Engineering Only)

| Artifact | Path | Status |
|----------|------|--------|
| Rehearsal log | `/data/liuyu/outputs/milestone_2f_gpu07_control_rehearsal_20260529/logs/rehearsal.log` | 6 episodes, 33% CUDA error rate |
| Rehearsal manifest | `.../rehearsal_manifest.json` | engineering_only=true |

## Running Workers

| Worker | Output Root | Progress |
|--------|-------------|----------|
| Goal-100 v2 | `/data/liuyu/outputs/milestone_2e5_goal100_parser_v2_privileged_rerun_20260527/` | 84/100 |
| L10-A (tasks 0-4) | `/data/liuyu/outputs/milestone_2e5_l10100_parser_v2_privileged_rerun_20260527/` | ~62 |
| L10-B (tasks 5-9) | Same root | ~62 |

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Object-100 clean SR | 81/100 (81%) |
| Teacher window coverage (clean success) | 81/81 (100%) |
| Teacher FP on failed episodes | 0/19 (0%) |
| ProprioNoStep offline coverage | 99.1% |
| ProprioNoStep offline AUROC | 0.969 |
| Shadow validation coverage | 97.8% |
| GPU0,7 rehearsal CUDA error rate | 33% |
| normalized_step in deployment path | 0 occurrences |
| Object/target pose in student input | 0 occurrences |
| Attack outcomes in detector training | 0 occurrences |

## Boundary Statement

No attack outcomes (VIS/random/oracle/manual) have been used to train or tune the detector. Privileged object/target state is teacher-label-only and does not enter deployed student features. Large artifacts (RGB frames, features, CSVs, models) are not committed to git. This index provides pointers for reproducibility.
