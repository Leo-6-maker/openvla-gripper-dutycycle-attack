# Window v2 Interim — Duration Confound Audit

**Date**: 2026-06-01

## Results

| Task | Window | Step | Attack | Success | Total Steps |
|------|--------|------|--------|---------|-------------|
| cream_cheese | det | 137 | 30 | **True** | 191 |
| cream_cheese | early | 107 | 30 | True | 163 |
| cream_cheese | vearly | 77 | 30 | True | 152 |
| cream_cheese | late | 167 | **0** | True | 163 |
| tomato_sauce | det | 196 | 30 | True | 245 |
| tomato_sauce | vearly | 136 | 30 | True | 219 |
| ketchup | early | 95 | 30 | True | 169 |
| salad | early | 70 | 30 | True | 131 |

## Key Finding

**30-step fixed burst does NOT cause failure** at any window position, including the detector window. Original sus30 had 153+ continuous trigger steps (attack_remaining reset by each new trigger), creating sustained open-grip that prevented task completion.

## Duration Confound

Window optimality cannot be claimed without disentangling:
1. **Onset timing**: detector selects the right phase (contact/transport)
2. **Duration**: sustained open-grip for 150+ steps is needed for high-sensitive failure

The detector window IS empirically critical — early windows (step 77-107 for cream_cheese) don't cause failure even with equivalent duration. But 30 steps is insufficient.

## Next: Window × Duration Ablation

Need to test various durations (30, 90, 150) at detector vs early windows to separate onset from duration effects.
