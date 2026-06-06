# Online Window Feature Extraction Audit

**Date**: 2026-06-07
**Source**: ProprioNoStep shadow calibration clean traces

## Coverage

- Total labeled windows: 31
- Traces available: 10
- Traces missing: 21

### Missing
- butter s0
- alphabet_soup s3
- alphabet_soup s4
- alphabet_soup s8
- bbq_sauce s5
- bbq_sauce s9
- butter s3
- butter s5
- cream_cheese s4
- ketchup s2
- ketchup s5
- milk s4
- milk s5
- milk s9
- alphabet_soup s6
- ketchup s4
- milk s8
- orange_juice s2
- salad_dressing s5
- tomato_sauce s1
- tomato_sauce s3

## Feature Groups

| Group | Features | Source | Count |
|---|---|---|---|
| Gripper qpos | mean, std, min, max, at_start, range, is_closed, is_open | env state | 8 |
| Gripper action | open_count, close_count, open_rate, mean, std, switches, longest_streak | decoded clean action | 7 |
| Raw gripper | mean, std | model output (before normalize) | 2 |
| End-effector | displacement, velocity_mean, z_mean, z_std, z_trend | env state | 5 |
| ProprioNoStep | hazard_mean/max, above_001/003, release_mean, phase_mode | TCN detector | 6 |
| Temporal (pre→window) | qpos_delta, grip_action_delta | computed | 2 |
| Window position | start_frac, center_frac, len_frac, step_at_start, steps_remaining | computed | 5 |
| **Total** | | | **35** |

## Missing Features (require GPU forward pass)

The following features are NOT available in current traces and would require
a GPU clean-forward pass with model logit/hidden-state extraction:

- Gripper logits (open_score, open_margin, open_entropy)
- Action token entropy (per-dimension)
- Visual embedding (PCA-compressed hidden states)
- Top-K token overlap between clean and expected action

## Next Steps

1. Use these 35 proprio/action/position features for detector v0 baseline
2. If baseline is insufficient, run GPU feature extraction for logits+embeddings
3. Add traces for missing tasks (butter, tomato_sauce, orange_juice) from clean shadow runs
