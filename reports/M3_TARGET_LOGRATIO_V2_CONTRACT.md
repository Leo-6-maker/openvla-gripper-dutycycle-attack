# M3 Target Log-Ratio V2 Contract

## Purpose

Phase D implements a non-saturating replacement for the step78 target-token
objective after the v1 canary showed that TRUE_PGD and RAND20 can both enter
the CW hinge zero-loss plateau.

This is a CPU-only code and contract PR. It does not add new real-model GPU
results, LIBERO rollout results, or any `TRUE_PGD > random` claim.

## Objective

`autoregressive_prefix_gripper_target_token_logratio_v2`

The target remains fixed:

- token: `31744`
- execution class: `CLIP_MEDIATED_OPEN`
- frame: Tomato clean step78
- epsilon: `6/255` in processor pixel-value space

The loss is:

```text
logsumexp(scores[j != 31744]) - scores[31744]
```

The optimized margin is:

```text
scores[31744] - logsumexp(scores[j != 31744])
```

Unlike v1, this objective has no hinge margin and no zero-loss plateau.

## Route Requirements

V2 keeps the strict true-PGD contract:

- `method=token_prefix_pgd`
- `strict_route=true`
- `allow_fallback=false`
- `target_token_id=31744`
- `target_execution_class=CLIP_MEDIATED_OPEN`
- `surrogate_score_path=cached_autoregressive_generate_v1`

The cached autoregressive surrogate path is required. Uncached full-context
surrogates are rejected for v2.

## Random Control

RAND20 selection must use the same target objective margin:

```text
surrogate_target_objective_margin
```

For v1 this equals the CW best-competitor margin. For v2 it is the
target-minus-competitor-logsumexp margin.

## Claim Boundary

Allowed claim: the repository now has a CPU-tested strict-route v2 objective
that removes the v1 hinge saturation issue while preserving the 31744 target
and cached surrogate path.

Forbidden claims: this PR does not show official fixed-frame transfer,
`TRUE_PGD > RAND20`, closed-loop critical-closure disruption, task effect, or
held-out transfer.
