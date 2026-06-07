# Stage-B Transfer Scorer — Smoke-A (76 paired)

**EXPLORATORY ONLY — USES 1R PROVISIONAL LABELS — NOT FINAL DETECTOR**

**Paired rows**: 75, **Features matched**: 75, **Features**: 11

## Label Distribution

| Label | Count |
|---|---|
| cmd_susceptible | 20 |
| random_confounded | 14 |
| pending_negative_1r | 41 |

## Leave-Task-Out AUROC (mean ± std)

| Label | Model | AUROC |
|---|---|---|
| cmd_susceptible | LR | 0.4555 ± 0.1729 |
| cmd_susceptible | RF | 0.5833 ± 0.1687 |
| cmd_susceptible | TaskOnly | 0.5000 ± 0.0000 |
| cmd_susceptible | Shuffle | 0.4375 ± 0.1667 |

## Gate Check

- Best cmd_susceptible AUROC: 0.5833
- WEAK/NO SIGNAL — features insufficient or labels noisy
