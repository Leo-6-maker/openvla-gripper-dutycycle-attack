# Milestone 2D Phase B Artifact Index

## Output Root

```
/data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527/
```

## Key Tables

- `tables/official_clean_artifact_rich_manifest.csv` — Aggregated 30-episode manifest
- `tables/object_smoke_summary_by_task.csv` — Per-task SR summary
- `tables/object_smoke_feature_coverage_by_task.csv` — Feature coverage by task
- `tables/object_smoke_runtime_status.csv` — Per-state success/runtime status
- `tables/visual_feature_manifest.csv` — 5,755-row visual path manifest
- `tables/no_timestep_visual_proprio_student_dataset.csv` — 5,355-row deployment dataset
- `tables/artifact_completeness_summary.csv` — Completeness summary
- `tables/shards/` — 3 worker shards

## Key Reports

- `reports/OBJECT_ARTIFACT_RICH_SMOKE_SUMMARY.md` — Full smoke summary
- `reports/ARTIFACT_COMPLETENESS_AUDIT.md` — Artifact completeness audit
- `reports/VISUAL_FEATURE_MANIFEST_STATUS.md` — Visual feature status
- `reports/NO_TIMESTEP_DATASET_EXPORT_STATUS.md` — No-timestep export status
- `reports/ENVIRONMENT_FIX_AUDIT.md` — Environment fix documentation
- `reports/NEXT_ACTION_STATUS.md` — Status key-value pairs

## Episode Data

- 30 episode directories under `runs/libero_object/`
- Each contains: `run_manifest.json`, `episode_records.jsonl`, `step_records.jsonl`, `frames/`

## Counts

| Metric | Count |
|--------|-------|
| Episodes | 30 |
| Step records rows | 5,755 |
| RGB frames | 5,355 |

## Run Directories

- `runs/libero_object/pick_up_the_milk_and_place_it_in_the_basket_state0/` through `state9/`
- `runs/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_state0/` through `state9/`
- `runs/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_state0/` through `state9/`

## Git Boundary

Do not commit output directories, frames, CSVs, shards, or checkpoints to git.
Only source, tests, and documentation belong in the repository.
