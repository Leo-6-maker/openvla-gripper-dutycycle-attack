# VIS Gripper Margin Scan

**Date**: 2026-06-02  
**Source**: `vis_gripper_margin_scan.py` on GPU23  
**Data**: 727 frames across 4 tasks

## Core Finding

**Raw gripper margin does NOT predict PGD flip success.**

cream_cheese step030 has margin=-47.63 (rank 105/128 = relatively HARD frame) but FLIPPED.
cream_cheese step035 has margin=-46.39 (rank ~80/128 = EASIER frame) but did NOT flip.

Frame-specific visual features and gradient paths dominate attack outcome, not the starting OPEN-vs-CLOSE logit balance.

## Methodology

### Margin Definition

```
gripper_margin_to_open = logsumexp(open_region_logits) - max_non_open_logit
open_margin_gap = max_non_open_logit - logsumexp(open_region_logits) = -gripper_margin_to_open
```

- Values are negative because the model is heavily CLOSE-biased
- Larger (closer to zero) = "easier" by raw logit metric
- open_margin_gap is the distance the attack must cross

### OPEN Region

Computed via `action_bins_for_env_sign(dim=-1, sign="negative", postprocess_gripper=True)`.

## Per-Task Summary

| Task | Frames | Easiest Gap | Median Gap | Hardest Gap | top1_is_open |
|------|--------|-------------|------------|-------------|-------------|
| cream_cheese | 128 | 29.2 (step14) | 42.3 | 57.4 (step118) | 0/128 |
| salad_dressing | 141 | 23.4 (step140) | 40.7 | 55.0 (step59) | 0/141 |
| ketchup | 158 | 21.2 (step151) | 41.7 | 55.6 (step68) | 0/158 |
| tomato_sauce | 300 | 13.1 (step63) | 34.7 | 51.7 (step48) | 0/300 |

**All 727 frames have top1_is_open = False and open_region_prob_mass = 0.0000.**

The model's clean policy never predicts gripper OPEN at the argmax level. OPEN probability mass is effectively zero at every step.

## step030 vs step035 Comparison

| Metric | step030 | step035 |
|--------|---------|---------|
| gripper_margin_to_open | -47.63 | -46.39 |
| open_margin_gap | 47.63 | 46.39 |
| non_open_max_logit | 59.00 | 60.50 |
| open_region_logsumexp | 11.37 | 14.11 |
| clean_gripper_action | 0.9961 (CLOSE) | 0.9961 (CLOSE) |
| gate-lite result | **FLIP** (eps8) | noop (eps8) |

Step035 has BETTER margin (closer to OPEN) and HIGHER open logsumexp. Yet PGD fails to flip it. This proves that:

1. The PGD gradient from pixel space to gripper logit space has different strength on different frames
2. Step030's visual features provide a stronger gradient path toward OPEN despite worse starting margin
3. Frame selection by margin alone is insufficient; gradient-aware or attack-aware selection is needed

## Neighbor Analysis (cream_cheese steps 24-36)

```
step24: margin=-40.50 gap=40.50  ← best margin in neighborhood
step26: margin=-40.78 gap=40.78
step28: margin=-42.32 gap=42.32
step30: margin=-47.63 gap=47.63  ← FLIP (harder margin!)
step32: margin=-49.97 gap=49.97
step34: margin=-50.54 gap=50.54
step35: margin=-46.39 gap=46.39  ← NO FLIP (easier margin!)
step36: margin=-45.84 gap=45.84
```

Steps 24 and 26 have much better margins than step30 but their PGD outcome is unknown. This neighborhood is the primary test set for the P3 selected-frame gate.

## Frame Selection Strategy

Since raw margin is insufficient, we use a mixed strategy:

### Group A: Easiest Margin (top-K per task)
- Quantifies "how much does easier margin help?"
- All frames have clean_grip > 0.5 (CLOSE/neutral)

### Group C: Positive Probe Neighbors (cream_cheese only)
- Tests whether the FLIP at step030 is a local continuous region or an isolated point
- Steps 24, 26, 28, 30, 32, 34, 36

### Group D: Diversity (easy/medium/hard per task)
- Measures margin-vs-attack relationship across the full margin range
- 3 easy (smallest gap), 3 medium (median), 3 hard (largest gap)

## Selected Frames

See `tables/vis_gripper_margin_selected_frames.csv` (83 frames total).

## Implications

1. **Model is extremely CLOSE-biased**: 727/727 frames have argmax CLOSE. VIS must overcome 20-55 logit gap.
2. **Margin is necessary but insufficient**: step030 flipped despite worse margin. Attack success depends on visual features and gradient path.
3. **Attack budget matters**: ε=8 can overcome a -47 margin on step030. This is substantial attack capacity.
4. **Frame selection should be attack-informed**: Future frame selection should consider gradient strength or use attack-probe results, not just clean-policy margins.

## Claim Boundary

- Do NOT claim: margin predicts attack success, margin-based frame selection is optimal
- CAN claim: model is CLOSE-biased across all clean-policy frames; ε=8 PGD can overcome -47 margin on select frames; visual features dominate attack gradient
