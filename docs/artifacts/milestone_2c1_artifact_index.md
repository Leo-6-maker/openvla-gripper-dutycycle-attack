# Milestone 2C.1 Artifact Index

## Output Root

```bash
/data/liuyu/outputs/milestone_2c1_student_replay_ablation_20260527
```

## Tables

| File | Description |
|------|-------------|
| `tables/replay_comparison_overall.csv` | Overall replay metrics for all 9 model/threshold combinations |
| `tables/replay_comparison_by_suite.csv` | Replay metrics broken down by LIBERO suite |
| `tables/replay_comparison_by_mechanism.csv` | Replay metrics broken down by mechanism type |
| `tables/per_episode_trigger_audit.csv` | Per-episode trigger details for all models (133,909 bytes) |
| `tables/threshold_sweep_comparison.csv` | Full threshold sweep (hazard × release_safe grid) |
| `tables/time_only_baseline_metrics.csv` | Classification metrics for time-only baseline |
| `tables/no_normalized_step_ablation_metrics.csv` | Classification metrics for no-normalized-step ablation |
| `tables/label_shuffle_sanity_metrics.csv` | Classification metrics for label-shuffle sanity baseline |
| `tables/bad_cases_for_manual_review.csv` | Top 30 episodes with worst replay performance |
| `tables/trigger_overlap_teacher_vs_student.csv` | Teacher vs student trigger agreement analysis |

## Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/time_only_best_model.pt` | Best time-only baseline model (26,498 bytes) |
| `checkpoints/no_normalized_step_best_model.pt` | Best no-normalized-step ablation model (31,122 bytes) |
| `checkpoints/label_shuffle_best_model.pt` | Best label-shuffle sanity model (31,362 bytes) |

## Reports

| File | Description |
|------|-------------|
| `reports/STUDENT_REPLAY_COMPARISON.md` | Auto-generated replay comparison summary |

## Source Code

| File | Description |
|------|-------------|
| `scripts/run_milestone_2c1_replay_ablation.py` | Main script: causal replay + anti-timing ablation training/eval |

## Dependencies

- Input dataset: `/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv`
- Full proprio checkpoint: `/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt`
- Module: `src/utils/proprio_causal_student.py` (from Milestone 2C)

## Provenance

- Label source: `clean_only_teacher`
- Uses attack outcome: `false`
- Uses manual outcome: `false`
- Split mode: `task_id` (stratified by suite)
- Test episodes: 40
- Seed: 7

## Key Metrics (Full Proprio, Coverage-First)

| Metric | Value |
|--------|-------|
| Window coverage | 0.8706 |
| False early trigger rate | 0.0116 |
| Miss rate | 0.1294 |
| Mean latency to window start | -2.65 steps |
| Episode coverage rate | 1.0000 (34/34 episodes with window) |
| Mean trigger count per episode | 12.33 |

## Anti-Timing Verification

| Check | Status |
|-------|--------|
| Full proprio > time-only | PASS |
| No-step ablation retains performance | PASS |
| Label-shuffle collapses | PASS |
| False early rate low | PASS |
