# R7.2.2 Closure Replay Handoff

**Date:** 2026-07-19
**PR:** [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)
**Commit:** `33f5dc3` on `agent/official-v3-r7-k10-v1.2.1-r7.2.2-closure-20260719`

## Executive Summary

R7.2.2 closes all HOLD items from the R7.2.1 audit. Using the official `load_v5_episodes` with Physics Teacher V2.1, fail-closed intent loading, auto-detected git identity, deployable dwell baselines, and paired per-episode representation diagnostics, the corrected replay confirms:

1. **Scheduler transfer is weak** — V5-A best recall 0.1154 (3/26), V5-B 0.0000 (0/26).
2. **Representation transfer is also weak** — paired delta analysis on the same 26 positive episodes shows negative mean delta (V5-A: -0.115, V5-B: -0.089), with only 2/26 (V5-A) and 3/26 (V5-B) episodes having higher peak score inside K10 corridors than outside.

Both the scheduler-level and representation-level evidence point to negligible K10 localization signal in frozen Physics checkpoints.

## R7.2.1 Audit Closure

| HOLD Item | R7.2.1 Status | R7.2.2 Status |
|---|---|---|
| 1. Wrong commit in SOURCE_BINDING | hardcoded fb9010e | Auto-detected: `33f5dc3` (git rev-parse HEAD) |
| 2. Manual jsonl reading | hand-rolled, fail-open valid | Official `load_v5_episodes` via Physics Teacher V2.1 |
| 3. V5-B zero intent fallback | silent zeros | ValueError on missing identity |
| 4. Dwell baseline retroactive | emit at t-9 | Emit at detection step t (deployable) |
| 5. Non-paired score diagnostics | 26 inside vs 200 outside means | Paired per-episode deltas on same 26 episodes |
| 6. Missing metrics | recall, precision, abstention only | +outside-rankable, release/post-release, containment, delay |
| 7. No tests or auditor | none | 12 CPU unit tests + independent artifact auditor |

## R7.2.2 Results

### Population

| Metric | Value |
|---|---|
| Validation identities | 200 (Fold-0, sealed fold manifest) |
| Loader | `load_v5_episodes` (official, Physics Teacher V2.1) |
| Candidate_close parity | 200/200 OK |
| Step count parity | 200/200 OK |
| Feasible K10 episodes | 26 |
| No-feasible episodes | 174 |

### V5-A (CausalMultimodalVulnerabilityRanker, V5_A_PROPRIO, strict=True)

| Threshold | Recall | Precision | Hits/Feasible | Emits | No-corridor Abstention |
|---|---|---|---|---|---|
| 0.1 | **0.1154** | 0.3333 | 3/26 | 9 | 0.9828 |
| 0.2 | 0.0769 | 0.4000 | 2/26 | 5 | 0.9828 |
| 0.3 | 0.0385 | 0.3333 | 1/26 | 3 | 0.9885 |
| 0.4 | 0.0000 | 0.0000 | 0/26 | 3 | 0.9885 |
| 0.5 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.6-0.7 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.8 | 0.0000 | 0.0000 | 0/26 | 1 | 0.9943 |
| 0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

### V5-B (real policy-intent, V5_B_PROPRIO_POLICY_INTENT, strict=True)

| Threshold | Recall | Precision | Hits/Feasible | Emits | No-corridor Abstention |
|---|---|---|---|---|---|
| 0.1 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.2-0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

### Paired Representation Diagnostics (26 feasible episodes only)

| Diagnostic | V5-A | V5-B |
|---|---|---|
| mean Δ (max_inside - max_outside) | **-0.1149** | **-0.0886** |
| median Δ | -0.0249 | -0.0143 |
| Δ > 0 count / 26 | 2 (7.7%) | 3 (11.5%) |
| Best-step-in-corridor / 26 | 2 (7.7%) | 3 (11.5%) |
| Mean feasible rank | 30.7 | 36.2 |

**Interpretation:** Both models peak *outside* K10 corridors. The best K10 feasible start ranks around position 31-36 among all rankable steps in a typical episode. Only 2-3 out of 26 feasible episodes have higher peak scores inside than outside the corridor. This is a representation-level failure, not a scheduler-design failure.

### Baselines

| Baseline | Recall | Precision | Hits/Feasible | Emits |
|---|---|---|---|---|
| First candidate_close | 0.0000 | 0.0000 | 0/26 | 200 |
| First valid dwell≥10 (deployable, emit at t) | 0.0000 | 0.0000 | 0/26 | 200 |

### Comparison Across R7.2 → R7.2.1 → R7.2.2

| Metric | R7.2 (invalid) | R7.2.1 | R7.2.2 |
|---|---|---|---|
| V5-A recall @ 0.1 | 0.013 (3/234) | 0.1154 (3/26) | 0.1154 (3/26) |
| V5-A emits @ 0.1 | 58 | 9 | 9 |
| V5-B recall @ 0.1 | 0.004 (1/234) | 0.0000 (0/26) | 0.0000 (0/26) |
| Loader | manual jsonl | manual jsonl | official load_v5_episodes |
| Intent fail-closed | no (zeros) | no (zeros) | yes (ValueError) |
| Git identity | none | hardcoded fb9010e | auto-detected 33f5dc3 |
| Paired diagnostics | none | aggregate only | per-episode paired |
| Unit tests | none | none | 12/12 PASS |
| Independent auditor | none | none | PASS |
| Dwell baseline emit | t-9 | t-9 | t (deployable) |

## Server Artifacts

```
R7.2.2 replay root:
  OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_33f5dc3_20260719
  SHA256SUMS: 13e8338ed6681dc23fd4f991070ba2caf0dcd1280b6314efe3e740b743f15dab

R7.2.2 audit root:
  OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_AUDIT_33f5dc3_20260719
  SHA256SUMS: 8d038783f398c701a525d2f40af6cf93725d7f69b727ad9af3c475a22a1235c5
  Status: PASS (0 fatal, 0 error, 0 warning)

Preserved (unchanged):
  R7.2:   OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_456bf73_20260719 (INVALID)
  R7.2.1: OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V21_CORRECTIVE_bc841ad_20260719 (PROVISIONAL)
```

## Development Gate Status

```
R7_R1_FORMAL_ARTIFACT             = PASS
R7_R2_SUBMITTED_ROOT              = PRESERVE / INVALID
R7_R2_1_SUBMITTED_ROOT            = PRESERVE / PROVISIONAL
R7_R2_2_CLOSURE_REPLAY            = COMPLETE
R7_R2_2_AUDITOR                   = PASS
R7_R2_2_V5_A_RECALL               = 0.1154 (3/26 at tau=0.1)
R7_R2_2_V5_B_RECALL               = 0.0000 (0/26)
R7_R2_2_PAIRED_DELTA_MEAN         = -0.1149 (V5-A), -0.0886 (V5-B)
R7_R2_2_PAIRED_DELTA_POSITIVE     = 2/26 (V5-A), 3/26 (V5-B)
R7_R2_2_BEST_IN_CORRIDOR          = 2/26 (V5-A), 3/26 (V5-B)

R7_R3_TRAINING                    = HOLD (awaiting authorization)
R7_R4_EXACT_PREFIX                = HOLD
R7_R5_ATTACK_CANARY               = HOLD

FIT_DEV_READ                      = NOT READ
CAL_READ                          = NOT READ
CHECK_READ                        = NOT READ
CS200_READ                        = NOT READ
ATTACK_EXECUTED                   = FALSE
```

## Conclusion

Both the scheduler transfer and representation transfer are weak. R7.3 K10-specific training is warranted — the Physics checkpoints contain negligible K10 localization signal at both the scheduler-output level and the raw-score representation level. All subsequent phases remain HOLD pending audit authorization.
