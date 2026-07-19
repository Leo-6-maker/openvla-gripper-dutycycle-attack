# Official V3 R7.2.1 Corrective Replay Audit — HOLD

Date: 2026-07-19  
PR: #87, Draft  
Reviewed evaluator commit: `bc841ad40b95b189abb39dc2d1e82da5a36777a8`

## Disposition

R7.2.1 closes the original R7.2 failures in the model forward path and threshold denominator. The reported scheduler results are useful development evidence:

```text
V5-A best scheduler hit = 3/26 = 0.1153846 at tau=0.1
V5-B best scheduler hit = 0/26
```

They support the provisional statement that the frozen Physics checkpoints are poor K10 scheduler candidates. They do not yet close the stronger representation-level statement that the checkpoints contain no K10 localization signal.

## Remaining blockers

1. `SOURCE_BINDING.json` records parent commit `fb9010e...`, although the evaluator first exists at `bc841ad...`.
2. The script manually reconstructs S1/K10 streams instead of using the official V5 loader and asserting K10 target parity.
3. Missing V5-B policy identity silently falls back to zero intent instead of failing.
4. `first_valid_dwell10` records `t-9`, which is retroactive and not deployable.
5. Score separation compares 26 inside samples with outside samples from up to 200 episodes instead of paired deltas on the same 26 feasible episodes.
6. Explicit outside-rankable, release/post-release, one-shot, containment and independent-ledger audit closure are missing.
7. No CPU tests or independent replay-root auditor accompany the evaluator.

The final paired representation diagnostic must report for each feasible episode:

```text
delta_i = max_score_inside_i - max_score_outside_i
```

and aggregate mean, median, count/rate `delta_i > 0`, raw best-step-in-corridor over 26, and feasible-start rank/percentile.

## R7.2.2 authorized scope

A narrow read-only closure replay is authorized. It may reuse the frozen A/B checkpoints and the same Fold-0 validation identities. It may not train, tune, select a threshold, read protected splits, run a simulator, or execute attacks.

The old R7.2 and R7.2.1 roots must be preserved unchanged. R7.2.2 must write a new root and a separate independent audit bundle.

## Promotion boundary

R7.3 may be authorized only after R7.2.2 establishes both scheduler transfer and paired raw-score representation transfer. If both remain weak, K10-specific training is warranted. If raw representation is informative but the scheduler is weak, the clean-only scheduler must be reconsidered first.

## Status

```text
R7_R1_FORMAL_ARTIFACT             = PASS
R7_R2_ORIGINAL_ROOT               = PRESERVE / INVALID
R7_R2_1_MODEL_FORWARD             = PASS
R7_R2_1_THRESHOLD_DENOMINATORS    = PASS
R7_R2_1_SCHEDULER_REPLAY          = PROVISIONAL PASS
R7_R2_1_SOURCE_BINDING            = HOLD
R7_R2_1_OFFICIAL_LOADER_PARITY    = HOLD
R7_R2_1_REPRESENTATION_DIAGNOSTIC = HOLD
R7_R2_1_BASELINE_CAUSALITY        = HOLD
R7_R2_1_FORMAL_ARTIFACT           = HOLD
R7_R2_2_CLOSURE_REPLAY            = AUTHORIZED — READ ONLY
R7_R3_TRAINING                    = HOLD
R7_R4_EXACT_PREFIX                = HOLD
R7_R5_ATTACK_CANARY               = HOLD
FIT_DEV / CAL / CHECK             = NOT READ
CS200_ATTACK                      = NOT STARTED
```
