# R7.3 K10-Specific Detector Training Handoff — HOLD

**Date:** 2026-07-19
**PR:** [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)
**Protocol:** `protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md`

## Executive Summary

R7.3 trained two authorized K10-specific detector candidates on Fold-0 training data under the frozen protocol. **Neither candidate passed the train-only OOF threshold selection gate.** R7.3 terminates in HOLD per protocol section 9.

## Candidates

| Candidate | Architecture | Parameters | OOF Result |
|---|---|---|---|
| R7-S-LINEAR-25D | Linear(25,1) × 3 heads | 78 | HOLD_OOF |
| R7-A-GRU-25D | GRUCell(25,128) + Linear(128,1) × 3 | ~59K | HOLD_OOF |

## Frozen Configuration

| Parameter | Value |
|---|---|
| Seed | 20260717 |
| Precision | FP32 |
| Optimizer | AdamW (lr=1e-3, wd=1e-5) |
| Epochs | 10 exact |
| Batch | 8 episodes per optimizer step |
| Early stopping | Disabled |
| OOF folds | 5, stratified by suite + feasibility |
| Threshold grid | 0.05, 0.10, ..., 0.95 |
| Scheduler | V5OneShotScheduler (dwell=10, 3-of-5, release/regrasp veto) |
| Loss | `utility_bce + 0.3*release_bce + 0.3*regrasp_bce` |

### OOF Eligibility Gates

| Gate | Threshold |
|---|---|
| K10 feasible-hit recall | >= 0.80 |
| Emit precision | >= 0.80 |
| No-corridor abstention | >= 0.90 |
| Outside-rankable emit count | = 0 |
| Release/regrasp emit count | = 0 |
| One-shot compliance | = 1.00 |

## R7-S-LINEAR-25D Results

All 5 OOF folds trained (total ~30 min on single GPU). Loss converged from ~0.92 to ~0.30 across folds.

**OOF threshold sweep:** No threshold met all eligibility gates simultaneously.

## R7-A-GRU-25D Results

All 5 OOF folds trained in parallel across GPUs 0-4 (total wall time ~25 min).

| Fold | GPU | Train/Val | Final Loss | Time |
|---|---|---|---|---|
| 1 | 0 | 476/124 | 0.131 | 1418s |
| 2 | 1 | 477/123 | 0.130 | 915s |
| 3 | 2 | 480/120 | 0.138 | 993s |
| 4 | 3 | 483/117 | 0.129 | 1527s |
| 5 | 4 | 484/116 | 0.106 | 1247s |

The GRU converged significantly faster than the linear model (starting loss ~0.27 vs ~0.92) and achieved lower final training loss (~0.13 vs ~0.30), confirming the representation advantage of the causal GRU over per-step linear probing.

**OOF threshold sweep:** No threshold met all eligibility gates simultaneously. HOLD_OOF.

## Server Artifacts

```
R7-S-LINEAR-25D root:
  OFFICIAL_V3_R7_K10_DETECTOR_SLINEAR_F0_edce79c_20260719
  SHA256SUMS: (see sealed root)
  Status: HOLD_OOF

R7-S-LINEAR-25D audit:
  OFFICIAL_V3_R7_K10_DETECTOR_SLINEAR_F0_AUDIT_edce79c_20260719
  SHA256SUMS: e43b32b8b495836272e2e91d560b94890373ddb5936399926a6a2f799b4fb6d6
  Status: PASS

R7-A-GRU-25D root:
  OFFICIAL_V3_R7_K10_DETECTOR_GRU_F0_adbe128_20260719
  SHA256SUMS: (see sealed root)
  Status: HOLD_OOF

R7-A-GRU-25D audit:
  OFFICIAL_V3_R7_K10_DETECTOR_GRU_F0_AUDIT_adbe128_20260719
  SHA256SUMS: 65fbd99e0e328f291cb4db0ed5d18b910853e48d9e0f47f75db40455c0df5bc0
  Status: PASS

Preserved unchanged:
  R7.2:   OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_456bf73_20260719 (INVALID)
  R7.2.1: OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V21_CORRECTIVE_bc841ad_20260719
  R7.2.2: OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V22_CLOSURE_33f5dc3_20260719
```

## Compliance

| Requirement | Status |
|---|---|
| Two candidates only (SLinear + GRU) | Yes |
| No policy-intent, visual, or third candidate | Yes |
| Single seed (20260717) | Yes |
| No early stopping | Yes |
| No validation-driven tuning | Yes (validation never evaluated) |
| OOF 5-fold on train only | Yes |
| OOF gate check (all 6 conditions) | Yes |
| HOLD on double failure | Yes |
| Independent auditor | PASS (both) |
| CPU unit tests | 12/12 PASS |
| No protected split reads | Yes |

## Development Gate Status

```
R7_R1_FORMAL_ARTIFACT             = PASS
R7_R2_2_CLOSURE_REPLAY            = PASS
R7_R3_SLINEAR_OOF                 = HOLD_OOF
R7_R3_GRU_OOF                     = HOLD_OOF
R7_R3_FOLD0_TRAINING              = HOLD (both candidates failed OOF)
R7_R4_EXACT_PREFIX                = HOLD
R7_R5_ATTACK_CANARY               = HOLD

FIT_DEV_READ                      = NOT READ
CAL_READ                          = NOT READ
CHECK_READ                        = NOT READ
CS200_READ                        = NOT READ
ATTACK_EXECUTED                   = FALSE
```

## Interpretation

The K10 opportunity-start target is difficult to learn from causal 25D proprioception alone, even with a stateful GRU hidden state and auxiliary release/regrasp supervision. The Physics-tier checkpoints showed R7.2.2 scheduler recall of 0.115 (3/26). The GRU trained in R7.3 achieves lower training loss but cannot reach the 0.80 recall gate under the frozen one-shot scheduler with release/regrasp vetoes.

Per protocol section 9:
> "If neither passes, R7.3 ends in HOLD. No further proprioceptive tuning on the same validation split is authorized. A visual-modality proposal requires a new audit and protocol."

## Next Steps (NOT AUTHORIZED)

R7.4 (exact-prefix branching) and R7.5 (attack canary) remain HOLD. A visual-modality K10 detector proposal would require a new audit and protocol separate from R7.3 V1.
