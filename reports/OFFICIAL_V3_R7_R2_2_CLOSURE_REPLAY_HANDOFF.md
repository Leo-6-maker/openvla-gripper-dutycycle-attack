# R7.2.2 Closure Replay Handoff

**Date:** 2026-07-19  
**PR:** [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)  
**Execution commit:** `33f5dc3488546ea9b9bbddb7152998c3fb560566`

## Executive Summary

R7.2.2 closes the load-bearing HOLD items from the R7.2.1 audit. Using the official `load_v5_episodes` with Physics Teacher V2.1, fail-closed policy-intent coverage, auto-detected git identity, deployable dwell baselines, and paired per-episode representation diagnostics, the corrected replay shows weak transfer on the sealed Fold-0 validation population:

1. **Scheduler transfer is weak** — V5-A best recall 0.1154 (3/26), V5-B 0.0000 (0/26).
2. **Representation transfer is weak** — paired delta analysis on the same 26 positive episodes gives negative mean delta (V5-A: -0.1149, V5-B: -0.0886), with only 2/26 and 3/26 episodes respectively having a higher peak score inside the K10 start corridor than outside.

This supports K10-specific clean-only detector development. It does not establish that proprioception cannot solve the target, that K10 is attack-vulnerability ground truth, or that the result transfers beyond Fold-0 FIT validation.

## R7.2.1 Audit Closure

| HOLD Item | R7.2.1 | R7.2.2 |
|---|---|---|
| Wrong commit in SOURCE_BINDING | hardcoded parent | execution commit auto-detected |
| Manual input loading | hand-rolled | official `load_v5_episodes` |
| V5-B intent fallback | silent zeros | validation identity coverage required |
| Dwell baseline | retroactive `t-9` | current causal time `t` |
| Score diagnostic | non-paired | paired on the same 26 episodes |
| Tests/auditor | absent | CPU tests and sealed artifact auditor |

## Results

### Population

| Metric | Value |
|---|---:|
| Fold-0 validation identities | 200 |
| K10-feasible episodes | 26 |
| No-corridor episodes | 174 |
| Candidate-close parity | 200/200 |
| Step-count parity | 200/200 |

### V5-A

| Threshold | Recall | Precision | Hits | Emits | No-corridor abstention |
|---|---:|---:|---:|---:|---:|
| 0.1 | **0.1154** | 0.3333 | 3/26 | 9 | 0.9828 |
| 0.2 | 0.0769 | 0.4000 | 2/26 | 5 | 0.9828 |
| 0.3 | 0.0385 | 0.3333 | 1/26 | 3 | 0.9885 |
| 0.4–0.8 | 0.0000 | 0.0000 | 0/26 | 1–3 | 0.9885–0.9943 |
| 0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

### V5-B

| Threshold | Recall | Precision | Hits | Emits | No-corridor abstention |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.2–0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

### Paired representation diagnostics

| Diagnostic | V5-A | V5-B |
|---|---:|---:|
| mean Δ = max-inside − max-outside | -0.1149 | -0.0886 |
| reported upper-middle order statistic | -0.0249 | -0.0143 |
| Δ > 0 | 2/26 | 3/26 |
| Best rankable step inside K10 starts | 2/26 | 3/26 |
| Mean best-feasible rank | 30.7 | 36.2 |

### Baselines

| Baseline | Recall | Precision | Hits | Emits |
|---|---:|---:|---:|---:|
| First candidate-close | 0.0000 | 0.0000 | 0/26 | 200 |
| First valid dwell≥10, emit at `t` | 0.0000 | 0.0000 | 0/26 | 200 |

## Server Artifacts

```text
Replay root:
OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_33f5dc3_20260719
SHA256SUMS = 13e8338ed6681dc23fd4f991070ba2caf0dcd1280b6314efe3e740b743f15dab

Audit root:
OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_AUDIT_33f5dc3_20260719
SHA256SUMS = 8d038783f398c701a525d2f40af6cf93725d7f69b727ad9af3c475a22a1235c5
Status = PASS
```

Preserved unchanged:

```text
R7.2   = INVALID
R7.2.1 = PROVISIONAL
```

## Audit Interpretation

The sealed auditor verifies artifact integrity, schemas, population closure, ledger row counts and ledger/metric consistency. It is not a second independent neural-network implementation.

`evaluator_file_blob_sha256` stores the 40-hex Git blob object ID returned by `git hash-object`; it is a valid source identifier but is not a SHA-256 digest.

Auxiliary safety fields in the replay are not promoted as primary endpoints. The load-bearing evidence is scheduler K10 recall/precision and the paired raw-score localization analysis.

## Development Gate Status

```text
R7_R1_FORMAL_ARTIFACT          = PASS
R7_R2_ORIGINAL_ROOT            = PRESERVE / INVALID
R7_R2_1_ROOT                   = PRESERVE / PROVISIONAL
R7_R2_2_CORE_REPLAY            = PASS
R7_R2_2_ARTIFACT_INTEGRITY     = PASS
R7_R2_2_TRANSFER_CONCLUSION    = PASS — WEAK ON FOLD-0

R7_R3_K10_SPECIFIC_TRAINING    = AUTHORIZED UNDER FROZEN PROTOCOL
R7_R4_EXACT_PREFIX             = HOLD
R7_R5_ATTACK_CANARY            = HOLD
FIT_DEV/CAL/CHECK              = NOT READ
CS200                          = NOT READ
ATTACK_EXECUTED                = FALSE
```

R7.3 must follow `protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md` and stop after the two authorized candidates and their sealed Fold-0 audit bundles are submitted.
