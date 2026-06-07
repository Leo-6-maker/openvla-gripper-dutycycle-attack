# Vulnerability-Ready Detector v1 - Diagnostic

**Train rows**: 22 (9 pos, 13 neg)
**Evaluation**: leave-one-task-out
**Prevalence baseline**: F1_pos=0.5806, F1_neg=0.0, balAcc=0.5

## Baselines

| Baseline | F1(pos) | F1(neg) | Bal Acc | MCC | TP/FP/FN/TN |
|----------|---------|---------|---------|-----|-------------|
| always_positive | 0.5806 | 0.0 | 0.5 | 0.0 | 9/13/0/0 |
| always_negative | 0.0 | 0.7429 | 0.5 | 0.0 | 0/0/9/13 |
| prevalence_random | 0.5 | 0.5833 | 0.547 | 0.0925 | 5/6/4/7 |

## Feature-Set Ablation

| Feature Set | Model | F1(pos) | F1(neg) | Macro F1 | Bal Acc | Neg Recall | MCC |
|-------------|-------|---------|---------|----------|---------|------------|-----|
| A_task_key_only | LR | 0.0 | 0.5806 | 0.2903 | 0.3462 | 0.6923 | -0.3922 |
| A_task_key_only | RF | 0.0 | 0.7429 | 0.3714 | 0.5 | 1.0 | 0.0 |
| B_phase_bin_only | LR | 0.4444 | 0.6154 | 0.5299 | 0.5299 | 0.6154 | 0.0598 |
| B_phase_bin_only | RF | 0.375 | 0.6429 | 0.5089 | 0.5128 | 0.6923 | 0.0271 |
| C_closed_pregrasp_gate | LR | 0.0 | 0.7429 | 0.3714 | 0.5 | 1.0 | 0.0 |
| C_closed_pregrasp_gate | RF | 0.0 | 0.7429 | 0.3714 | 0.5 | 1.0 | 0.0 |
| D_causal_safe | LR | 0.6957 | 0.6667 | 0.6812 | 0.7137 | 0.5385 | 0.4368 |
| D_causal_safe | RF | 0.25 | 0.5714 | 0.4107 | 0.4188 | 0.6154 | -0.1714 |
| E_phase+causal | LR | 0.5833 | 0.5 | 0.5417 | 0.5812 | 0.3846 | 0.1714 |
| E_phase+causal | RF | 0.4706 | 0.6667 | 0.5686 | 0.5684 | 0.6923 | 0.1398 |
| F_task+phase | LR | 0.125 | 0.5 | 0.3125 | 0.3248 | 0.5385 | -0.3699 |
| F_task+phase | RF | 0.1176 | 0.4444 | 0.281 | 0.2863 | 0.4615 | -0.4368 |
| G_task+phase+causal | LR | 0.4545 | 0.4545 | 0.4545 | 0.4701 | 0.3846 | -0.0598 |
| G_task+phase+causal | RF | 0.0 | 0.5333 | 0.2667 | 0.3077 | 0.6154 | -0.4512 |
| H_descriptor_upper | LR | 0.6957 | 0.6667 | 0.6812 | 0.7137 | 0.5385 | 0.4368 |
| H_descriptor_upper | RF | 0.2353 | 0.5185 | 0.3769 | 0.3803 | 0.5385 | -0.2446 |

## Task Split Warnings

- tasks with <2 train rows: cream_cheese,orange_juice

## Key Findings

1. **No model beats prevalence**: best balanced accuracy=0.7137.
2. **Negative recall is 0 for most models**: models predict all-positive or all-negative.
3. **v1 underpowered**: N=22 insufficient to learn vulnerability_ready.
4. **Controls needed**: stable/post_lock, far_too_early, pre_lock negatives.

## Verdict

Beats prevalence (balAcc=0.71)
Diagnostic only. NOT deployable.
