# Official V3 Fresh40 V5 R1/R2 Review

## Review scope

This review starts from handoff commit `f07fb43c79d16339abf3f79d177792a2b72e7717` and preserves every historical source, checkpoint, prediction, and shadow root. The remediation code is on `codex/fresh40-v5-canary-20260728` at `404c5279e9d15edea22a0754dd884582880c83d2d`, with no protected-split or attack input access.

The f07 engineering chain is reproducible, but its scientific canary must remain `NOT_EVALUATED_DUE_TO_CONFOUNDS`: the Teacher is a recorded-telemetry proxy, and the original shadow metric divided one-shot emits by step-level critical labels.

## Confirmed f07 P0 findings

1. `critical_only` and the historical `three_head` path allowed untrained heads to affect emission. The frozen remediation equations are now explicit:

   - `critical_only`: `candidate AND physical_criticality`;
   - `three_head`: `candidate AND physical_criticality AND NOT instability AND gripper_closing_state`;
   - `full_five`: all five declared predicates.

2. The old `critical_recall_among_known_true` was not event recall. It used a one-shot emit numerator against a known-critical timestep denominator.

3. The Teacher has no exact object-gripper contact pairs. The source contains global contact information and distance proxies; `physical_criticality` and `safe_release` therefore remain development-only labels.

4. The proxy label distribution is unsuitable for a scientific learnability claim: physical criticality has 10,715 UNKNOWN of 11,710 steps, instability has 16 TRUE steps, and safe release has no exact contact semantics.

## R1 remediation

Commit `404c5279e9d15edea22a0754dd884582880c83d2` adds:

- frozen active-head equations and inactive-head mutation tests;
- head-level AUROC, AUPRC, recall, precision, and balanced accuracy;
- candidate-gate ceiling;
- event-level positive recall, known-negative event false-positive rate, and latency;
- explicit UNKNOWN handling (`unknown_as_negative=false`, UNKNOWN excluded from event denominators);
- strict event masking when any step in the candidate event is UNKNOWN;
- development-only prediction schema version 2.

Official environment validation:

```text
py_compile = PASS
pytest n5/phase4_fresh40/tests/test_fresh40_v5_contracts.py = 9 passed, 0 failed
```

The test set includes the inactive-head mutation contract and a partial-UNKNOWN event denominator test.

## R2 oracle ladder

The ladder reuses only the already sealed full-five checkpoint and the sealed eight-identity development split. It is diagnostic only; Teacher fields are written to the oracle bundle but are not Student or scheduler runtime inputs.

```text
development identities = 8
development steps      = 2,363
candidate true coverage ceiling = 87 / 106 = 0.8207547169811321
candidate events       = 85
UNKNOWN-excluded events = 78
known positive events  = 1
known negative events  = 6
```

Because only one positive event remains after strict UNKNOWN exclusion, `1/1` oracle recall is not a scientific PASS and cannot support a detector claim.

| Ladder | Known positive event recall | Known negative event FPR | Interpretation |
|---|---:|---:|---|
| Teacher critical + candidate | 1/1 | 0/6 | diagnostic oracle only; denominator is insufficient |
| predicted critical | 0/1 | 0/6 | current critical score did not select the known positive event |
| predicted critical + Teacher auxiliary veto | 0/1 | 0/6 | auxiliary oracle does not rescue the observed event |
| full predicted one-shot | 0/1 | 0/6 | no evidence of a valid scientific detector result |

The variant shadow replays use the same strict event denominator. Their known-event recall is `0/1` with `0/6` known-negative FPR; the emitted rows are mostly UNKNOWN-label diagnostics, not scored negatives. Candidate-gate and UNKNOWN coverage therefore prevent promotion.

## Sealed development evidence

All new roots passed recursive `SHA256SUMS` and `SHA256SUMS.sha256` verification, and all output roots are non-overwriting. The historical f07/d477/7ea roots remain untouched.

| Bundle | Root | `SHA256SUMS.sha256` |
|---|---|---|
| critical checkpoint, critical-only replay | `fresh40_v5_shadow_r1_critical_ckpt_criticalonly_404c_20260728T080000Z` | `e48880e387e838b6f3d00dba63b47d64fa37f28f92455ab1349320585f5905ce` |
| full checkpoint, critical-only equation | `fresh40_v5_shadow_r1_full_ckpt_criticalonly_404c_20260728T080000Z` | `a08929ae6ec416bfaac0805cda96b212f1679cd411bba20657daab49bae48c84` |
| full checkpoint, three-head equation | `fresh40_v5_shadow_r1_full_ckpt_threehead_404c_20260728T080000Z` | `70d91ad275e46fb3fe40fb6d4d0fe8432f34a8ef0f06404a5e3b897e4b41b474` |
| full checkpoint, full-five equation | `fresh40_v5_shadow_r1_full_ckpt_fullfive_404c_20260728T080000Z` | `499522833be0ac1fdd3e50359d58475d2e3c7d2b97463c0ce757c4cc63382f84` |
| oracle ladder | `fresh40_v5_oracle_ladder_full5_404c_20260728T080000Z` | `eb0c4c1663a388b41da132b86523c218f7eff15d23f68f89a4ea21dd9ae8f691` |

The development dataset manifest SHA is `6f2a6bfe0071aa56aa671f3586f4c7dfaf64c228e77c6c7a372db455af0b910a`; the full checkpoint SHA is `fe9173c96387610b4292105727c7574df298dd2878e97648e37b8cc9418d0fb8`.

## Gate decision

```text
V5_R1_CONTRACT_REMEDIATION       = PASS
V5_R2_ORACLE_LADDER              = HOLD_INSUFFICIENT_KNOWN_EVENT_COVERAGE
FRESH40_SCIENTIFIC_CANARY        = NOT_EVALUATED_DUE_TO_CONFOUNDS
V5_R3_RETRAINING                 = NOT RUN
FRESH670                         = BLOCKED
PROTECTED_READS                  = 0
OPENVLA_INFERENCE                = NOT RUN
STUDENT_TRAINING_THIS REMEDIATION = NOT RUN
STUDENT_DEVELOPMENT_SHADOW       = RUN ON SEALED DEV SPLIT ONLY
ROLLOUT                          = NOT RUN
ATTACK                           = NOT RUN
```

The next scientifically meaningful step is corrected contact-pair Teacher canary data. Do not promote these proxy labels, do not run Fresh670, and do not infer observability limits from this R2 result.
