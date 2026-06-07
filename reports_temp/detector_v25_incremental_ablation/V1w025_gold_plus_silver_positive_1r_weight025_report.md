# Vulnerability-Ready Detector v1 - Diagnostic

**Train rows**: 33 (20 pos, 13 neg)
**Evaluation**: leave-one-task-out
**Sample weighting**: weighted_fit:sample_weight; baselines are unweighted evaluation rows.
**Prevalence baseline**: F1_pos=0.7547, F1_neg=0.0, balAcc=0.5

## Baselines

| Baseline | F1(pos) | F1(neg) | Bal Acc | MCC | TP/FP/FN/TN |
|----------|---------|---------|---------|-----|-------------|
| always_positive | 0.7547 | 0.0 | 0.5 | 0.0 | 20/13/0/0 |
| always_negative | 0.0 | 0.5652 | 0.5 | 0.0 | 0/0/20/13 |
| prevalence_random | 0.6364 | 0.2727 | 0.4654 | -0.076 | 14/10/6/3 |

## Feature-Set Ablation

| Feature Set | Model | F1(pos) | F1(neg) | Macro F1 | Bal Acc | Neg Recall | MCC |
|-------------|-------|---------|---------|----------|---------|------------|-----|
| A_task_key_only | LR | 0.0 | 0.5652 | 0.2826 | 0.5 | 1.0 | 0.0 |
| A_task_key_only | RF | 0.0 | 0.5652 | 0.2826 | 0.5 | 1.0 | 0.0 |
| B_phase_bin_only | LR | 0.5294 | 0.5 | 0.5147 | 0.5327 | 0.6154 | 0.0646 |
| B_phase_bin_only | RF | 0.0 | 0.5 | 0.25 | 0.4231 | 0.8462 | -0.315 |
| C_closed_pregrasp_gate | LR | 0.0 | 0.5652 | 0.2826 | 0.5 | 1.0 | 0.0 |
| C_closed_pregrasp_gate | RF | 0.0 | 0.5652 | 0.2826 | 0.5 | 1.0 | 0.0 |
| D_causal_safe | LR | 0.8182 | 0.6364 | 0.7273 | 0.7192 | 0.5385 | 0.4811 |
| D_causal_safe | RF | 0.7317 | 0.56 | 0.6459 | 0.6442 | 0.5385 | 0.293 |
| E_phase+causal | LR | 0.6667 | 0.4167 | 0.5417 | 0.5423 | 0.3846 | 0.0877 |
| E_phase+causal | RF | 0.7368 | 0.6429 | 0.6898 | 0.6962 | 0.6923 | 0.385 |
| F_task+phase | LR | 0.08 | 0.439 | 0.2595 | 0.3712 | 0.6923 | -0.3512 |
| F_task+phase | RF | 0.2963 | 0.5128 | 0.4046 | 0.4846 | 0.7692 | -0.0368 |
| G_task+phase+causal | LR | 0.6154 | 0.4444 | 0.5299 | 0.5308 | 0.4615 | 0.0608 |
| G_task+phase+causal | RF | 0.75 | 0.6154 | 0.6827 | 0.6827 | 0.6154 | 0.3654 |
| H_descriptor_upper | LR | 0.7907 | 0.6087 | 0.6997 | 0.6942 | 0.5385 | 0.413 |
| H_descriptor_upper | RF | 0.8 | 0.5714 | 0.6857 | 0.6808 | 0.4615 | 0.4122 |

## Task Split Warnings

- tasks with <2 train rows: cream_cheese,orange_juice

## Key Findings

1. **No model beats prevalence**: best balanced accuracy=0.7192.
2. **Negative recall is 0 for most models**: models predict all-positive or all-negative.
3. **v1 underpowered**: N=33 insufficient to learn vulnerability_ready.
4. **Controls needed**: stable/post_lock, far_too_early, pre_lock negatives.

## Verdict

Beats prevalence (balAcc=0.72)
Diagnostic only. NOT deployable.
