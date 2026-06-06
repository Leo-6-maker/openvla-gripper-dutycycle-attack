# Object100 Opportunity Localizer — Stage-A Results

**Dataset**: 294 rows, 74 positive, 24 features
**Eval**: Leave-task-out (9-fold)

## Model Performance (mean ± std across tasks)

| Model | AUROC | AUPRC | P@10 | P@20 | R@20 |
|---|---|---|---|---|---|
| LR | 0.9974 ± 0.0074 | 0.9962 ± 0.0107 | 0.8111 ± 0.1728 | 0.4259 ± 0.0510 | 1.0000 ± 0.0000 |
| RF | 0.9974 ± 0.0074 | 0.9962 ± 0.0107 | 0.8111 ± 0.1728 | 0.4259 ± 0.0510 | 1.0000 ± 0.0000 |
| TaskOnly | 0.5000 ± 0.0000 | 0.3035 ± 0.0310 | 0.2333 ± 0.0471 | 0.3037 ± 0.0105 | 0.7259 ± 0.1136 |

## Per-Task AUROC

| Task | LR AUROC | RF AUROC | TaskOnly | TimeOnly |
|---|---|---|---|---|
| alphabet_soup | 1.0 | 1.0 | 0.5 | nan |
| bbq_sauce | 1.0 | 1.0 | 0.5 | nan |
| butter | 1.0 | 1.0 | 0.5 | nan |
| cream_cheese | 1.0 | 1.0 | 0.5 | nan |
| ketchup | 1.0 | 1.0 | 0.5 | nan |
| milk | 0.9766 | 0.9766 | 0.5 | nan |
| orange_juice | 1.0 | 1.0 | 0.5 | nan |
| salad_dressing | 1.0 | 1.0 | 0.5 | nan |
| tomato_sauce | 1.0 | 1.0 | 0.5 | nan |

## RF Top-10 Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | n_window_frames | 0.3233 |
| 2 | eef_displacement | 0.1489 |
| 3 | qpos_delta_from_pre | 0.1126 |
| 4 | eef_z_trend | 0.1104 |
| 5 | grip_command_delta_from_pre | 0.0591 |
| 6 | gripper_qpos_range | 0.0427 |
| 7 | gripper_qpos_std | 0.0409 |
| 8 | eef_z_mean | 0.0381 |
| 9 | gripper_width_std | 0.0345 |
| 10 | gripper_qpos_max | 0.0178 |

## Gate Evaluation

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| Best model beats time-only | AUROC > 0.5000 | LR=0.9974 RF=0.9974 | YES |
| Best model beats task-only | AUROC > 0.5000 | LR=0.9974 RF=0.9974 | YES |
| AUROC >= 0.75 | 0.75 | LR=0.9974 RF=0.9974 | YES |
| Leave-task-out does not collapse | – | min LR=0.9766 min RF=0.9766 | OK |

## Gate Verdict: PASS

Proceed to Phase-2 VIS labeling plan.
