# Milestone 2D Phase C.2 Artifact Index

## Output Root

```
/data/liuyu/outputs/milestone_2d_phase_c2_privileged_artifact_rich_object_smoke_20260527
```

## Key Tables

- `tables/official_clean_artifact_rich_manifest.csv` — 30-episode aggregated manifest
- `tables/privileged_state_coverage_by_task.csv` — Per-task privileged state coverage
- `tables/privileged_state_coverage_by_episode.csv` — Per-episode privileged state coverage
- `tables/sim_name_matching_summary.csv` — Object/receptacle body/site matching
- `tables/object_priv_teacher_window_labels.csv` — Teacher contact window labels
- `tables/object_priv_teacher_phase_labels.csv` — Teacher per-step phase labels
- `tables/object_priv_detector_coverage_by_task.csv` — Teacher detector coverage
- `tables/object_priv_detector_coverage_by_episode.csv` — Per-episode detector coverage
- `tables/no_timestep_visual_proprio_student_dataset_labeled.csv` — Labeled deployment dataset
- `tables/no_timestep_label_coverage_by_task.csv` — Label coverage by task
- `tables/no_timestep_deployment_feature_audit.csv` — Deployment feature leakage audit
- `tables/visual_feature_manifest.csv` — Visual feature manifest
- `tables/no_timestep_visual_proprio_student_dataset.csv` — Unlabeled deployment dataset
- `tables/shards/` — 3 worker shards

## Key Reports

- `reports/OBJECT_PRIVILEGED_ARTIFACT_RICH_SMOKE_SUMMARY.md` — Full summary
- `logs/c2_priv_milk_10_w0.log` — Milk worker log
- `logs/c2_priv_cream_cheese_10_w1.log` — Cream cheese worker log
- `logs/c2_priv_bbq_sauce_10_w2.log` — BBQ sauce worker log

## Episode Data

- 30 episode directories under `runs/libero_object/`
- Each contains: `run_manifest.json`, `episode_records.jsonl`, `step_records.jsonl`, `frames/`, `debug/sim_names.json`

## Counts

| Metric | Count |
|--------|-------|
| Episodes | 30 |
| Step records rows | 5,904 |
| Policy steps | 5,604 |
| RGB frames | 5,604 |
| Teacher phase labels | 5,604 |

## Git Boundary

Do not commit output directories, frames, CSVs, shards, or checkpoints to git.
Only source, tests, and documentation belong in the repository.
