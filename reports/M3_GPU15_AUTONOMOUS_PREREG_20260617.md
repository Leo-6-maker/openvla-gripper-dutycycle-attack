# M3 GPU15 Autonomous Layer3 Preregistration

This stage is authorized for physical GPU pair `(1,5)` only, using `CUDA_VISIBLE_DEVICES=1,5`.

The run is infrastructure- and gate-driven. It may not reuse `(2,6)` fixed-frame margins or controls as formal matched controls. All matched conditions for the Tomato step78 screen must be regenerated under `(1,5)`.

Primary target:

```text
target token: 31744
epsilon: 6/255
objective family: target log-ratio plus clean arm-token preservation
arm gate: actual generated arm prefix match >= 5/6
strict route: true
fallback: forbidden
```

Finite state machine:

```text
S0_INIT
S1_GPU_QUAL
S2_CPU_BUILD
S3_TOMATO_SCREEN
S5_MULTI_PARENT
S6_ORACLE_CLOSED_LOOP
S8_FINALIZE
STOPPED / FAILED / COMPLETE
```

Tonight's first hard gate is Tomato step78. For each lambda in `[0.25, 0.5, 1.0, 2.0]`, the watcher runs a fresh `(1,5)` capture/preflight/canary using the existing hard-feasible V4 trajectory selection runner with the arm-preserve weight set to that lambda.

Tomato PASS requires:

```text
TRUE selected candidate exists
TRUE official gripper token == 31744
TRUE arm match >= 5/6
TRUE margin > RAND21 selected margin
TRUE margin > SHUFFLED selected margin
Linf <= 6/255
route audit PASS
fallback_used == false
exact 7-token audit PASS
```

Scientific failures are terminal for this campaign. The watcher may retry an infrastructure failure once, but may not change epsilon, target token, PGD steps, random-control selection metric, or window/frame after observing results.

If Tomato fails, the campaign finalizes with an L3-0 result and does not start multi-parent or closed-loop phases. If Tomato passes but the Layer1/2 G5 tag or handoff is unavailable, the campaign finalizes after Tomato and reports that downstream phases were not run.

No detector-triggered integration is authorized unless all fixed-frame and oracle gates pass and the required G5 tag is present with at least 90 minutes remaining.
