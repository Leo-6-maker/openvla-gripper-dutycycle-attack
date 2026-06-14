# M3 log-ratio arm-v3 preregistration

## Objective

`autoregressive_prefix_gripper_target_token_logratio_arm_v3`

This is a fixed-frame development objective for Tomato step78.  It keeps the
same target token, frame, epsilon, step count, and official gates as v2, but
adds an arm-preservation CE penalty using the actual clean-generated arm prefix
as labels.

## Frozen Settings

- Target token: `31744`
- Target execution class: `CLIP_MEDIATED_OPEN`
- Frame: `tomato_sauce`, state `0`, clean absolute step `78`
- Epsilon: `6/255` in processor pixel-value space
- Steps: `20`
- Step size: `1.5 * epsilon / 20`
- Prefix refresh interval: `1`
- Surrogate score path: `cached_autoregressive_generate_v1`
- Arm preserve weight: `0.5`
- Arm gate: actual clean generated arm-prefix match `>=5/6`

## Gate

The v3 canary must satisfy all of the following before any multi-frame or
LIBERO rollout stage is allowed:

1. Strict route passes: no fallback, `TokenPrefixPGDAttacker`, `20` backwards.
2. Official exact 7-token generation and score invariant pass.
3. TRUE_PGD emits token `31744`.
4. TRUE_PGD actual clean arm-prefix match is at least `5/6`.
5. TRUE_PGD official target margin beats selected `RAND20`.
6. TRUE_PGD official target margin beats `SHUFFLED_GRAD_PGD20`.

## Allowed Claim

If the gate passes, this can support only fixed-frame target-token and
selectivity feasibility for the Tomato development frame.

## Forbidden Claim

Do not claim closed-loop Layer3, task effect, held-out transfer, or defense
validity from this canary.  No LIBERO rollout is part of this stage.
