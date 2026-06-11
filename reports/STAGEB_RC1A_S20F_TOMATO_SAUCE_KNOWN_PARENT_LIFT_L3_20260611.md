# S20F tomato_sauce_s0_w70-80: Known-Parent Lift to Official/V4 Layer3

**Date:** 2026-06-11
**Runner:** `scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py` (V4-based)
**Claim level:** Known vulnerable parent lift, NOT v0.3.1 detector-selected

## Result: TASK_EFFECT_POSITIVE (2/3 seeds)

| Seed | Clean | RAND | VIS | VIS-RAND Δopen | Classification |
|------|-------|------|-----|----------------|----------------|
| 80 | 209✓ | 185✓ (open=3,str=2) | **280✗ timeout** (open=10,str=10) | +7 | TASK_EFFECT |
| 81 | 209✓ | 229✓ (open=5,str=5) | **280✗ timeout** (open=9,str=9) | +4 | TASK_EFFECT |
| 82 | 209✓ | 230✓ (open=8,str=7) | 175✓ (open=10,str=10) | +2 | CONTACT_WEAK |

Seeds 80 and 81 both show:
- Clean baseline success under S20d/V4 official runner
- Matched RAND control success (no task failure under random Linf perturbation)
- VIS timeout at 280-step horizon with sustained OPEN (10/10 and 9/9)
- VIS OPEN significantly stronger than RAND (Δ+7 and Δ+4)

Seed 82: RAND has high natural OPEN (8/10), VIS still stronger (10/10) but recovers.

## Attribution

Trace analysis of VIS80 post-window behavior:
- EEF converges (z-range decreases 0.066→0.056→0.010) → arm trajectory normal
- Gripper oscillates OPEN/CLOSE persistently (34.5% OPEN rate over 200 post-window steps)
- Model attempts task completion but cannot maintain stable grasp

**Classification: GRIPPER_CONTACT_ATTRIBUTED** — VIS-induced sustained OPEN prevents successful placement, not model collapse.

## Allowed Claim

```
A previously confirmed Layer1/2 vulnerable parent (tomato_sauce_s0_w70-80)
transfers to official/V4 full-episode evaluation and produces VIS-specific
task failure under matched random control (2/3 seeds confirmed).
```

## Forbidden Claims

- NOT v0.3.1 detector-selected
- NOT object-wide attack success
- NOT task-wide SR drop
- NOT LIBERO-wide generalization
- NOT Layer3 solved globally
