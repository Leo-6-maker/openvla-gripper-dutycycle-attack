# M3 Log-Ratio V2 Seed81 Fixed-Frame Canary

## Result

`ARM_NONSELECTIVE`

The v2 log-ratio objective passed route, optimization, official-transfer, and
random-control margin gates on the Tomato step78 fixed frame, but it failed
the arm-selectivity gate.

This stops the v2 seed81 development canary before any fixed-frame panel,
closed-loop rollout, rescue, or held-out parent work.

## Primary Metrics

- stage commit: `2d602e28dd5d410490cfe93ccf4ddb082078da82`
- base merge SHA: `8ff6596d4089a66a7d3e2774d4471459841c608c`
- attack seed: `81`
- GPU mapping: `CUDA_VISIBLE_DEVICES=2,6`
- output dir: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/canary_logratio_v2_step78_2d602e2_seed81`
- clean token: `31872`
- delta0 token: `31872`
- TRUE_PGD token: `31744`
- RAND20 selected token: `31744`
- SHUFFLED_GRAD_PGD20 token: `31872`

Margins use the v2 target objective:

```text
scores[31744] - logsumexp(scores[j != 31744])
```

- delta0 official margin: `-0.25`
- TRUE_PGD official margin: `29.249469757080078`
- RAND20 best official margin: `6.0`
- SHUFFLED_GRAD_PGD20 official margin: `-0.5`

## Gates

- infra: `PASS`
- strict route: `PASS`
- no fallback: `PASS`
- exact 7 official tokens: `PASS`
- score invariant: `PASS`
- processor-space Linf <= 6/255: `PASS`
- optimization: `PASS`
- official transfer: `PASS`
- TRUE_PGD > RAND20 margin: `PASS`
- TRUE_PGD > SHUFFLED_GRAD_PGD20 margin: `PASS`
- arm selectivity: `FAIL`

The TRUE_PGD generated arm prefix was:

```text
[31938, 31870, 31938, 31882, 31999, 31915]
```

The clean arm prefix was:

```text
[31900, 31870, 31915, 31882, 31862, 31913]
```

Match count: `2/6`, below the preregistered `5/6` gate.

## Controls

RAND20 selected candidate:

- candidate id: `3`
- candidate seed: `1376924591`
- selection metric: `surrogate_target_objective_margin`
- selected margin: `6.0`

SHUFFLED_GRAD_PGD20 preserved the arm prefix at `6/6` but did not improve the
target margin and emitted `31872`.

## Claim Boundary

Allowed claim: on the fixed Tomato step78 development frame, log-ratio v2
TRUE_PGD can strongly increase the official 31744 objective margin and beat
the selected RAND20 and shuffled-gradient controls on margin, but only by
substantially changing the generated arm prefix.

Forbidden claims: no gripper-selective attack, no `TRUE_PGD > random` safety
claim, no fixed-frame panel success, no closed-loop critical-closure
disruption, no paired task effect, no held-out transfer, and no solved Layer3
pipeline.

No LIBERO rollout was launched.
