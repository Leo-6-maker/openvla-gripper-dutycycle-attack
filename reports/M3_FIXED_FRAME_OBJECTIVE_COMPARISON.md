# M3 Fixed-Frame Objective Comparison

## Summary

This audit freezes the two completed Tomato step78 fixed-frame objective
results.

| Objective | Result |
| --- | --- |
| `autoregressive_prefix_gripper_target_token_cw_v1` | `RANDOM_NOT_BEATEN` |
| `autoregressive_prefix_gripper_target_token_logratio_v2` | `ARM_NONSELECTIVE` |

## Interpretation

The v1 CW objective produced an official `31744` gripper token and preserved
the clean arm prefix for TRUE_PGD (`6/6`), but it did not beat the matched
RAND20 control. The full RAND20 audit found that random selective `31744`
matches existed in the frozen candidate set (`13/20`).

The v2 log-ratio objective removed the v1 hinge saturation issue and strongly
increased the official target margin relative to RAND20 and shuffled-gradient
controls, but it did so with substantial arm-prefix drift (`2/6`).

Therefore, target controllability and arm selectivity have not yet been
achieved simultaneously on the Tomato step78 development frame.

## Claim Boundary

Allowed claim: the completed fixed-frame evidence supports neither v1 nor v2
as a gripper-selective true-PGD superiority result over matched controls.

Forbidden claims: no closed-loop critical-closure disruption, no task effect,
no held-out transfer, no detector-selected Layer3 success, and no solved
Layer3 pipeline.

No new GPU inference or LIBERO rollout was launched for this comparison.
