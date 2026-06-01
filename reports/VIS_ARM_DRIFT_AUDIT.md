# VIS Arm Drift and Action Specificity Audit

**Date**: 2026-06-01 | **Phase**: E

## Summary

Arm drift was measured as L2 norm of the 6-dim arm action delta between clean and adversarial decode.

## Best Results (Lowest Arm L2 with Grip Flip)

| Task | Frame | Seeds | Grip Delta | Arm L2 | Ratio |grip|/arm | Verdict |
|------|-------|-------|-----------|--------|------|--------|
| cream_cheese | 0070 | 3/3 | +0.996 | **0.110-0.114** | 8.7 | **EXCELLENT** |
| tomato_sauce | 0130 | 3/3 | +0.996 | **0.131** | 7.6 | **EXCELLENT** |
| tomato_sauce | 0138 | 3/3 | +0.996 | **0.155-0.191** | 5.2-6.4 | **VERY GOOD** |
| cream_cheese | 0065 | 3/3 | +0.996 | 0.489 | 2.0 | Good |
| cream_cheese | 0080 | 3/3 | +0.996 | 0.597 | 1.7 | Acceptable |
| ketchup | 0098 | 10/10 | +0.996 | 0.839 | 1.2 | Elevated |
| cream_cheese | 0085 | 3/3 | +0.996 | 0.844 | 1.2 | Elevated |
| tomato_sauce | 0142 | 3/3 | +0.996 | 0.784-1.183 | 0.8-1.3 | High |
| tomato_sauce | 0150 | 3/3 | +0.996 | 0.922 | 1.1 | High |

## Evidence Against Arm-Drift-Only Hypothesis

**cream_cheese step_0075**: Arm L2 = 1.074 but grip delta = 0.0. The arm is heavily perturbed but the gripper token does NOT flip. This demonstrates that the PGD effect is **gripper-specific**, not just a generic action collapse caused by arm perturbation.

## Selectivity Pattern

The ratio |grip_delta|/arm_l2 varies from 0.8 to 8.7 across frames:
- Frames with low ratio (0.8-1.2): arm and gripper change together (less specific)
- Frames with high ratio (5-9): gripper changes independently of arm (highly specific)
- cream_cheese 0070 achieves the best trade-off: full grip flip with minimal arm disturbance

## Random Baseline Comparison

| Frame | Type | Arm L2 (targeted) | Arm L2 (random) | Grip Delta (targeted) | Grip Delta (random) |
|-------|------|-------------------|-----------------|----------------------|---------------------|
| ketchup_0098 | B | 0.839 | 0.000 | +0.996 | 0.000 |
| tomato_0134 | B | 0.178 | 0.000 | 0.000 | 0.000 |
| ketchup_0050 | B | 0.512 | 0.000 | -0.996 | 0.000 |

Random same-Linf perturbation produces essentially zero arm L2 and zero grip change — the targeted PGD is distinctly different from random perturbation.

## Gate E: PASS

- Gripper change is clear and directional (0.0→0.996, OPEN)
- Best frames show arm L2 as low as 0.11 with full grip flip
- cream_cheese_0075 proves the effect is not arm-drift-driven (high arm L2, no grip change)
- Random baseline produces no effect
- The effect is gripper-specific, not generic action collapse
