# Official V3 Detector V5 R2 stratified smoke

Status: `HOLD` (development gate only).

The fixed Fold-0 train subset contains 80 identities, exactly two per
suite/task, selected by frozen hash order.  Validation is the complete 200
identity Fold-0 validation set.  The run used the official A800 environment,
GPU4, FP32, seed `20260717`, and three epochs.  No protected split was read.

## Sealed results

- checkpoint SHA256:
  `ba680e67a9c971097ec200c47bc80e91afc1f762e28bc49112b8f7da222c9e7a`
- strict true-mixed episodes: `126`
- causal top-1 hit: `112/126 = 0.8888888889`
- pure-negative episodes: `3`
- causal online pure-negative abstention: `0/3 = 0.0`
- retrospective diagnostic abstention at 0.5: `1/3 = 0.3333333333`
- one-shot compliance: `true`
- total online emits: `172/200`

The same 507-episode FIT geometry gives the longest-window baseline a strict
true-mixed top-1 rate of `0.9289940828`; V5-A is below that shortcut baseline.
Consequently the frozen R2 smoke gate fails.  This is not an attack result or
a vulnerability measurement.

The failed first subset build, the pre-fix diagnostic retry, and the final
smoke root are all retained.  No full Fold-0 training, multi-seed run,
FIT-DEV, CAL, CHECK, or attack was started.

```text
V5_R2_STRATIFIED_SMOKE = HOLD
V5_A_FULL_FOLD0       = NOT RUN
FIT_DEV / CAL / CHECK = NOT READ
ATTACK                = NOT STARTED
```
