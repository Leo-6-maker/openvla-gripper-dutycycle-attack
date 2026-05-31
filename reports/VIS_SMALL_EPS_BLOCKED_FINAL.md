# VIS Small-Epsilon Blocked Final

Date: 2026-05-31

VIS remains blocked before rollout.

## Blocking Condition

The real one-frame loader now works and TokenPrefixPGD now respects the requested processor-pixel Linf budget. However, the valid-budget smoke did not produce a decoded gripper-token flip or decoded gripper-action change.

After the budget fix, the diagnostic reported:

```text
requested eps = 4/255 = 0.015686
observed perturbation_linf = 0.0078125
```

This means the previous budget bug is fixed for `processor_pixel_values_linf` semantics, but VIS remains blocked because there is still no decoded gripper effect.

## Evidence

From `tables/vis_token_flip_threshold_diagnostic.csv`:

- target CE: `32.0000 -> 15.9500`
- open-bin probability mass: `5.87e-13 -> 1.52e-07`
- close-bin probability mass: `0.999996 -> 0.987568`
- clean gripper token: `31872`
- adversarial gripper token: `31872`
- gripper token flipped: `false`
- clean gripper action: `0.0`
- adversarial gripper action: `0.0`
- gripper delta: `0.0`
- arm L2: `0.184442`
- perturbation Linf: `0.0078125`

## Gate Decision

VIS-1: FAIL.

Do not run:

- VIS rollout
- forced-window VIS micro
- detector-triggered VIS
- heavy arm-drift sweep

## Next Required Work

Improve VIS effectiveness under valid budget:

1. Keep `processor_pixel_values_linf` semantics unless a separate raw-image-space PGD path is implemented.
2. Improve objective/optimization without increasing arm drift.
3. Re-run one-frame loader smoke.
4. Only run a threshold sweep if decoded gripper token/action movement appears under valid budget.
