# Milestone 2D — Artifact-Rich Clean Runner for No-Timestep Online Detector

## Purpose

Prepare the codebase for artifact-rich clean rollouts that save per-timestep data required by the final no-timestep visual/proprio online detector. The current official eval runner produces only task-level summaries; this milestone adds step_records, RGB frames, gripper/EEF/action traces, and parallel-safe manifest shards.

## Why normalized_step Is Removed from Deployment Detector

- full_proprio coverage = 0.8706 (includes normalized_step — dev upper bound)
- no_normalized_step coverage = 0.7529 (deployment starting point)
- The deployment detector must use signals available on a real robot: visual, gripper, EEF, action history, task language
- normalized_step / absolute timestep are simulation-only and not transferable

## Current 2C/2C.1 Status

- 2C: proprio-only causal student trained, hazard F1 0.6743, AUROC 0.9874
- 2C.1: anti-timing validation passed (full > time-only, no-step > 0.3, label-shuffle = 0)
- Object R2: 80/100 under corrected PIL preprocessing

## Phase A Scope (Implemented)

- [x] Artifact-rich runner schema (configs/artifact_rich_official_eval_schema.yaml)
- [x] Deployment feature schema (configs/deployment_detector_feature_schema.yaml)
- [x] Artifact-rich runner script (scripts/run_official_eval_artifact_rich.py)
- [x] Parallel-safe aggregator (scripts/aggregate_artifact_rich_manifests.py)
- [x] Visual feature manifest scaffold (scripts/build_visual_feature_manifest.py)
- [x] No-timestep dataset exporter (scripts/export_no_timestep_visual_proprio_dataset.py)
- [x] Tests (25 passed)

## What Phase A Does NOT Run

- No GPU rollouts
- No attack conditions
- No detector training
- No visual encoder extraction

## Phase B Plan (Future)

After Spatial/Goal/L10 Table1 complete:
- Object smoke: bbq_sauce, cream_cheese, milk (artifact-rich)
- Verify clean SR preserved
- Verify step_records, image_paths complete
- Run teacher detector on artifact-rich data
- Build no-timestep dataset
- Begin Milestone 2E visual feature extraction
