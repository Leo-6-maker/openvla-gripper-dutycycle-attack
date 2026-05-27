# Milestone 2C.1 Reproducibility Guide

## Prerequisites

- Conda environment: `openvla_official_libero_20260525` (Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1)
- GPU: Not required (CPU-only training is sufficient and recommended)
- Input dataset: 87,474 clean timestep rows from Milestone 2B
- Full proprio checkpoint: best_model.pt from Milestone 2C

## Input Files

```bash
# Training dataset (400 episodes, 87,474 timestep rows)
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv

# Full proprio student checkpoint
/data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt
```

## Run Command

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524

python scripts/run_milestone_2c1_replay_ablation.py \
    --data_csv /data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv \
    --model_path /data/liuyu/outputs/milestone_2c_proprio_causal_student_20260526/checkpoints/best_model.pt \
    --output_root /data/liuyu/outputs/milestone_2c1_student_replay_ablation_20260527 \
    --split_mode task_id \
    --device cpu \
    --seed 7 \
    --epochs 50 \
    --batch_size 1024 \
    --coverage_hazard 0.1 \
    --coverage_release 0.3 \
    --conservative_hazard 0.5 \
    --conservative_release 0.5
```

## What the Script Does

1. Loads the training dataset and full proprio checkpoint
2. Evaluates the full proprio student in causal replay mode on the test split (40 episodes)
3. Computes a rule/proxy trigger baseline (gripper command/width/qpos heuristics)
4. Computes a teacher window oracle reference
5. Trains a time-only baseline MLP (only `normalized_step` + categorical features)
6. Trains a no-normalized-step ablation MLP (all features except `normalized_step`)
7. Trains a label-shuffle sanity MLP (shuffled `teacher_hazard` labels)
8. Evaluates all models in causal replay mode
9. Generates comparison tables and a summary report

## Expected Output

### Overall Metrics (coverage-first threshold)

| Model | Window Coverage | False Early | Miss Rate |
|-------|----------------|-------------|-----------|
| teacher_window | ~1.0000 | ~0.0000 | ~0.0000 |
| full_proprio_coverage | ~0.8706 | ~0.0116 | ~0.1294 |
| rule_proxy | ~0.0147 | ~0.0020 | ~0.9853 |
| time_only_coverage | ~0.4265 | ~0.0045 | ~0.5735 |
| no_normalized_step_coverage | ~0.7529 | ~0.0248 | ~0.2471 |
| label_shuffle | ~0.0000 | ~0.0000 | ~1.0000 |

### Ablation Classification Metrics

| Model | Phase Macro F1 | Hazard F1 | Hazard AUROC |
|-------|---------------|-----------|-------------|
| time_only | ~0.12 | ~0.00 | ~0.96 |
| no_normalized_step | ~0.67 | ~0.42 | ~0.95 |
| label_shuffle | ~0.69 | ~0.00 | ~0.50 |

## Reproducibility Notes

- The split is determined by `--seed 7` with `task_id` split mode. Different seeds will produce different train/val/test assignments.
- The full proprio checkpoint must match the split assignment (trained with `--split_mode task_id --seed 7`).
- CPU training takes approximately 10-20 minutes for all three ablation models (50 epochs each, early stopping typically triggers earlier).
- The rule/proxy trigger baseline is deterministic (no training required).
- The label-shuffle baseline uses `seed + 9999` for shuffling to avoid accidental correlation with the original labels.

## Expected Runtime

- Data loading + encoding: ~30 seconds
- Full proprio replay: ~2 seconds
- Rule/proxy replay: ~1 second
- Teacher window replay: <1 second
- Time-only training (50 epochs): ~2-3 minutes
- No-step training (50 epochs): ~3-5 minutes
- Label-shuffle training (50 epochs): ~3-5 minutes
- Total: approximately 10-20 minutes on CPU

## Verification

After running, verify:
1. `tables/replay_comparison_overall.csv` has 9 rows (one per model/threshold combination)
2. `tables/time_only_baseline_metrics.csv` shows hazard AUROC ≈ 0.96 but hazard F1 ≈ 0.00 (time-only cannot classify hazard well)
3. `tables/label_shuffle_sanity_metrics.csv` shows hazard AUROC ≈ 0.50 (random)
4. `reports/STUDENT_REPLAY_COMPARISON.md` shows all four anti-timing checks as PASS
