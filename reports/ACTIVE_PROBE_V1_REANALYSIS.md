# Active Probe V1 Re-analysis Report

**Date**: 2026-06-07
**Input**: v0b step features (logit-level proxy, NOT decoded gripper)
**Windows**: 28

## Data Summary

| Label | N Pos | N Total |
|---|---|---|
| cmd_sus_k1 | 13 | 28 |
| cmd_sus_k3 | 11 | 28 |
| cmd_sus_k6 | 11 | 28 |
| cmd_sus_k10 | 11 | 28 |
| phys_bridge | 11 | 28 |
| phys_any | 11 | 28 |

## AUROC / AUPRC

| Score | Label | AUROC | AUPRC | P@1 | P@3 | P@5 |
|---|---|---|---|---|---|---|
| open_score_gain_max | cmd_sus_k1 | 0.4564 | 0.5567 | 1 | 0.6667 | 0.6 |
| open_score_gain_max | cmd_sus_k3 | 0.492 | 0.5406 | 1 | 0.6667 | 0.6 |
| open_score_gain_max | cmd_sus_k6 | 0.492 | 0.5406 | 1 | 0.6667 | 0.6 |
| open_score_gain_max | cmd_sus_k10 | 0.492 | 0.5406 | 1 | 0.6667 | 0.6 |
| open_score_gain_max | phys_bridge | 0.492 | 0.5406 | 1 | 0.6667 | 0.6 |
| open_score_gain_max | phys_any | 0.492 | 0.5406 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | cmd_sus_k1 | 0.4513 | 0.5249 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | cmd_sus_k3 | 0.4759 | 0.4931 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | cmd_sus_k6 | 0.4759 | 0.4931 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | cmd_sus_k10 | 0.4759 | 0.4931 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | phys_bridge | 0.4759 | 0.4931 | 1 | 0.6667 | 0.6 |
| open_score_gain_mean | phys_any | 0.4759 | 0.4931 | 1 | 0.6667 | 0.6 |
| open_token_count_total | cmd_sus_k1 | 0.5 | 0.5344 | 0 | 0.6667 | 0.4 |
| open_token_count_total | cmd_sus_k3 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_count_total | cmd_sus_k6 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_count_total | cmd_sus_k10 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_count_total | phys_bridge | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_count_total | phys_any | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_rate | cmd_sus_k1 | 0.5 | 0.5344 | 0 | 0.6667 | 0.4 |
| open_token_rate | cmd_sus_k3 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_rate | cmd_sus_k6 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_rate | cmd_sus_k10 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_rate | phys_bridge | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_token_rate | phys_any | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_streak | cmd_sus_k1 | 0.5 | 0.5344 | 0 | 0.6667 | 0.4 |
| open_dominant_streak | cmd_sus_k3 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_streak | cmd_sus_k6 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_streak | cmd_sus_k10 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_streak | phys_bridge | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_streak | phys_any | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_count | cmd_sus_k1 | 0.5 | 0.5344 | 0 | 0.6667 | 0.4 |
| open_dominant_count | cmd_sus_k3 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_count | cmd_sus_k6 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_count | cmd_sus_k10 | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_count | phys_bridge | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| open_dominant_count | phys_any | 0.5 | 0.384 | 0 | 0.3333 | 0.2 |
| token_flip_rate | cmd_sus_k1 | 0.4744 | 0.4903 | 0 | 0.3333 | 0.4 |
| token_flip_rate | cmd_sus_k3 | 0.5695 | 0.4878 | 0 | 0.3333 | 0.4 |
| token_flip_rate | cmd_sus_k6 | 0.5695 | 0.4878 | 0 | 0.3333 | 0.4 |
| token_flip_rate | cmd_sus_k10 | 0.5695 | 0.4878 | 0 | 0.3333 | 0.4 |
| token_flip_rate | phys_bridge | 0.5695 | 0.4878 | 0 | 0.3333 | 0.4 |
| token_flip_rate | phys_any | 0.5695 | 0.4878 | 0 | 0.3333 | 0.4 |

## Best per Label

- **cmd_sus_k1**: best AUROC=0.4513 (score=open_score_gain_mean, n_pos=13)
- **cmd_sus_k3**: best AUROC=0.5695 (score=token_flip_rate, n_pos=11)
- **cmd_sus_k6**: best AUROC=0.5695 (score=token_flip_rate, n_pos=11)
- **cmd_sus_k10**: best AUROC=0.5695 (score=token_flip_rate, n_pos=11)
- **phys_bridge**: best AUROC=0.5695 (score=token_flip_rate, n_pos=11)
- **phys_any**: best AUROC=0.5695 (score=token_flip_rate, n_pos=11)

## Conclusion

These results use LOGIT-LEVEL open_token_count as a proxy for decoded gripper actions.
v0b step features do NOT contain actual decoded gripper actions.
Step 2 (active_probe_v1_temporal.py) is needed for true decoded gripper streak analysis.
