# Official V3 R7.2.1 Corrective Replay Audit — HOLD

Date: 2026-07-19  
PR: #87, Draft  
Reviewed commit: `bc841ad40b95b189abb39dc2d1e82da5a36777a8`

## Disposition

R7.2.1 closes the original R7.2 P0 failures in the model forward path and metric denominator:

- repository `CausalMultimodalVulnerabilityRanker` is used;
- checkpoint loading is strict;
- V5-B consumes the sealed policy-intent root;
- the repository `V5OneShotScheduler` is used;
- Fold-0 validation contains 200 unique identities, including 26 K10-feasible episodes;
- threshold denominators are no longer expanded by the nine-point sweep.

The reported scheduler results are therefore useful development evidence:

```text
V5-A best scheduler hit = 3/26 = 0.1153846 at tau=0.1
V5-B best scheduler hit = 0/26
```

They support the provisional statement that the frozen Physics checkpoints are poor K10 scheduler candidates. They do not yet close the stronger representation-level statement that the checkpoints contain no K10 localization signal.

## Remaining blockers

### P0 — source binding is factually wrong

`SOURCE_BINDING.json` hard-codes:

```text
git_commit = fb9010e49ac05c28aa3e0e259ac7f1df9fbad412
```

but the corrective evaluator first exists at:

```text
bc841ad40b95b189abb39dc2d1e82da5a36777a8
```

The server root named with `bc841ad` is therefore not correctly bound to its executing commit. The old root must remain immutable and be marked provisional. A new root must derive the runtime commit dynamically and fail if the worktree is dirty.

### P0 — official data-loader parity is not closed

The corrective script imports but does not use `load_v5_episode`/`load_v5_episodes`. It manually reads `student_input_records.jsonl` and K10 labels, uses `valid` with a permissive default, and uses K10 `candidate_close` as the scheduler gate.

The final replay must load the model inputs and candidate/valid streams through the same official V5 loader used by training and the existing evaluator. K10 labels must be joined only as the target. It must assert exact identity, step-count, candidate-close, and window/segment parity rather than silently assuming them.

### P0 — V5-B intent consumption is not fail-closed

When an identity is absent from the policy index, the current helper falls back to an all-zero intent stream. The final replay must raise immediately for any missing identity, missing step, invalid intent row, or normalization mismatch.

### P1 — causal baseline timestamp is retroactive

The `first_valid_dwell10` baseline detects that dwell reaches 10 at current step `t` but records the emission at `t-9`. An online detector cannot emit into the past. The deployable baseline must emit at current step `t`. A retrospective start-of-window value may be reported only as future-informed diagnostics.

### P1 — score separation is not paired

The current score diagnostic compares:

- inside maxima from the 26 feasible episodes; and
- outside maxima from up to all 200 episodes.

This is not a paired separation test. The final representation diagnostic must operate only on the same 26 feasible episodes and report, per episode:

```text
delta_i = max_score_inside_i - max_score_outside_i
```

Required aggregates are mean delta, median delta, count/rate `delta_i > 0`, raw best-step-in-corridor count over 26, and feasible-start rank/percentile among causal candidate steps. This separates representation transfer from scheduler transfer.

### P1 — required safety metrics are incomplete

The final ledger must explicitly report:

```text
outside-rankable emits
release/post-release emits
one-shot compliance
K10 containment
false-early emits
late/outside-corridor emits
first-feasible-start delay
no-corridor abstention
```

Equivalent quantities must still be written under these names and independently recomputed from the ledger.

### P1 — no tests or independent replay-root auditor

The submitted commit adds one evaluator script only. Acceptance requires CPU tests for:

- one denominator per threshold;
- strict checkpoint loading;
- missing V5-B intent fails;
- official-loader/K10 parity mismatch fails;
- dwell-10 emits at current time, not `t-9`;
- paired score separation uses only feasible episodes;
- one-shot and outside-rankable gates.

A read-only independent auditor must verify root seals, source binding, 200-identity closure, 3,600 candidate-threshold ledger rows, aggregate recomputation, and the absence of protected or attack reads.

## R7.2.2 authorized scope

A narrow read-only closure replay is authorized. It may reuse the frozen A/B checkpoints and the same 200 Fold-0 validation identities. It may not train, tune, select a threshold, read protected splits, run a simulator, or execute attacks.

The old R7.2 and R7.2.1 roots must be preserved unchanged. R7.2.2 must write a new root and a separate independent audit bundle.

## Promotion boundary

R7.3 may be authorized only after R7.2.2 establishes both:

1. **scheduler transfer:** official one-shot scheduler K10 hit/precision/abstention curves; and
2. **representation transfer:** paired raw-score localization on the 26 feasible episodes.

If both remain weak, K10-specific detector training is warranted. If raw representation is informative but the scheduler is weak, R7.3 must first reconsider the clean-only scheduler rather than assuming the encoder/ranker lacks signal.

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
