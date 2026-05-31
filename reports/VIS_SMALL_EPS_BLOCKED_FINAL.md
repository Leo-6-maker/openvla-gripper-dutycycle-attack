# VIS Small-Epsilon Blocked Final

Date: 2026-05-31

VIS remains blocked before rollout.

## Blocking Condition

The real one-frame loader now works, but the first real smoke did not produce a decoded gripper-token flip or decoded gripper-action change.

More importantly, the diagnostic reported:

```text
requested eps = 4/255 = 0.015686
observed perturbation_linf = 2.125
```

This means the current TokenPrefixPGD normalized `pixel_values` perturbation accounting/clamp is not a valid small-epsilon image-budget implementation for a VIS claim.

## Evidence

From `tables/vis_token_flip_threshold_diagnostic.csv`:

- target CE: `32.0000 -> 30.9197`
- open-bin probability mass: `5.87e-13 -> 1.76e-11`
- close-bin probability mass: `0.999996 -> 0.562177`
- clean gripper token: `31872`
- adversarial gripper token: `31872`
- gripper token flipped: `false`
- clean gripper action: `0.0`
- adversarial gripper action: `0.0`
- gripper delta: `0.0`
- arm L2: `0.054859`

## Gate Decision

VIS-1: FAIL.

Do not run:

- VIS rollout
- forced-window VIS micro
- detector-triggered VIS
- heavy arm-drift sweep

## Next Required Work

Fix or explicitly define TokenPrefixPGD perturbation-space semantics:

1. Decide whether epsilon is in raw image `[0, 1]`, processor-normalized pixel space, or another space.
2. Keep an fp32 master perturbation in the chosen space.
3. Avoid clamping normalized processor `pixel_values` to `[0, 1]` if those values are not raw pixels.
4. Re-run one-frame loader smoke.
5. Only then run a threshold sweep.
