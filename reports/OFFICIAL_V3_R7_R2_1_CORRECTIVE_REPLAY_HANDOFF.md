# R7.2.1 Corrective Offline Replay Handoff

**Date:** 2026-07-19
**PR:** [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)
**Commit:** `008a2f9` on `agent/official-v3-r7-k10-v1.2.1-r7.2.1-corrective-20260719`

## Executive Summary

R7.2.1 corrects all P0 findings from the R7.2 audit. Using the official `CausalMultimodalVulnerabilityRanker` with `strict=True` loading, real sealed policy-intent for V5-B, and the official `V5OneShotScheduler`, the corrected replay confirms: **frozen V5 Physics checkpoints do not transfer to K10 opportunity detection.**

V5-A achieves 0.1154 recall (3/26) at the lowest threshold (tau=0.1) with the official scheduler, but score diagnostics show the model has zero internal separation favoring K10 corridors (best-in-corridor rate 0.01, mean score higher outside corridor than inside). V5-B achieves 0.0000 recall across all thresholds with real policy intent. Both baselines score 0.0000.

These corrected results are consistent with the original qualitative conclusion that Physics-tier checkpoints provide negligible K10 localization signal, but the quantitative values differ substantially from the invalid R7.2 submission.

## R7.2 Audit Closure

| Finding | R7.2 Status | R7.2.1 Status |
|---|---|---|
| P0-1: Denominator 9x inflated | 26→234 | Fixed: per-threshold 26 |
| P0-2: Non-official model | ad-hoc V5PhysicsGRU, sigmoid gate, no Tanh | Fixed: CausalMultimodalVulnerabilityRanker |
| P0-3: V5-B zero intent | intent_dummy = zeros | Fixed: sealed policy-intent root + 9D normalization |
| P0-4: strict=False loading | silent miss | Fixed: strict=True |
| P0-5: Wrong scheduler | candidate_close AND utility >= threshold | Fixed: V5OneShotScheduler (dwell=10, 3-of-5, vetoes) |
| P0-6: No lineage closure | hard-coded states, silent skip, default valid=True | Fixed: sealed fold manifest, no skips, step-count parity |
| P0-7: No episode ledger | aggregate only | Fixed: 3600-row episode_threshold_ledger.jsonl |

## R7.2.1 Results

### Population

| Metric | Value |
|---|---|
| Validation identities | 200 (Fold-0, states 0-4) |
| Feasible K10 episodes | 26 |
| No-feasible episodes | 174 |
| Thresholds | 9 (0.1-0.9) |
| Missing/skipped | 0 |

### V5-A (CausalMultimodalVulnerabilityRanker, V5_A_PROPRIO)

| Threshold | Recall | Precision | Hits/Feasible | Emits | No-corridor Abstention |
|---|---|---|---|---|---|
| 0.1 | 0.1154 | 0.3333 | 3/26 | 9 | 0.9828 |
| 0.2 | 0.0769 | 0.4000 | 2/26 | 5 | 0.9828 |
| 0.3 | 0.0385 | 0.3333 | 1/26 | 3 | 0.9885 |
| 0.4 | 0.0000 | 0.0000 | 0/26 | 3 | 0.9885 |
| 0.5 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.6 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.7 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.8 | 0.0000 | 0.0000 | 0/26 | 1 | 0.9943 |
| 0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

Score diagnostics: best_in_corridor_rate=0.0100, mean_max_inside=0.0558, mean_max_outside=0.0875

### V5-B (CausalMultimodalVulnerabilityRanker, V5_B_PROPRIO_POLICY_INTENT, real intent)

| Threshold | Recall | Precision | Hits/Feasible | Emits | No-corridor Abstention |
|---|---|---|---|---|---|
| 0.1 | 0.0000 | 0.0000 | 0/26 | 2 | 0.9885 |
| 0.2-0.9 | 0.0000 | — | 0/26 | 0 | 1.0000 |

Score diagnostics: best_in_corridor_rate=0.0150, mean_max_inside=0.0347, mean_max_outside=0.0910

### Causal Baselines

| Baseline | Recall | Precision | Hits/Feasible | Emits |
|---|---|---|---|---|
| First candidate_close | 0.0000 | 0.0000 | 0/26 | 200 |
| First valid dwell≥10 | 0.0000 | 0.0000 | 0/26 | 200 |

### Comparison: R7.2 vs R7.2.1

| Metric | R7.2 (invalid) | R7.2.1 (corrected) |
|---|---|---|
| V5-A recall @ 0.1 | 0.013 (3/234) | 0.1154 (3/26) |
| V5-A emits @ 0.1 | 58 | 9 |
| V5-B recall @ 0.1 | 0.004 (1/234) | 0.0000 (0/26) |
| V5-B emits @ 0.1 | 42 | 2 |
| V5-A model | ad-hoc V5PhysicsGRU | CausalMultimodalVulnerabilityRanker |
| V5-B intent | zeros(1, T, 9) | sealed clean_policy_intent_9d |
| Scheduler | candidate_close AND utility>=tau | V5OneShotScheduler (dwell/persistence/vetoes) |

## Interpretation

1. **Corrected results still show near-zero K10 transfer.** V5-A recall improved from 0.013 to 0.1154 after the fixes, but this is still far below the 0.80 development gate. The official scheduler is dramatically more selective (9 emits vs 58 at tau=0.1), which improves precision (0.33 vs 0.05) but keeps recall low because the model's peak utility scores fall outside K10 corridors.

2. **Score diagnostics confirm no localization signal.** Both V5-A and V5-B show *negative* score separation: the mean maximum utility score is higher outside K10 corridors than inside. The best-step-in-corridor rate (0.01 for V5-A, 0.015 for V5-B) is at chance level. The Physics ranking objective (predicting causal utility tiers) does not produce representations that peak at K10 feasible starts.

3. **V5-B with real intent performs worse than V5-A.** The policy intent features do not help K10 detection — V5-B emits near-zero at all thresholds. This is the opposite of what the broken R7.2 suggested (which showed V5-B with 42 dummy-intent emits).

4. **Baselines confirm K10 starts occur late.** Both first-close (always before K10) and first-dwell-10 baselines score 0.0000 recall, confirming K10 feasible starts require conditions (stable_grasp + manipulation_active) established well after initial gripper close.

## Server Artifacts

```
Root: OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_V21_CORRECTIVE_bc841ad_20260719
SHA256SUMS: e7cc17002929e7f377fcb64293da785d24f636aad3d8f6899c371e924b43c10d

Previous R7.2 root (preserved, invalid):
  OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_456bf73_20260719
  SHA256SUMS: f13a83efec6b2431507ceaef3376c30e0ba0091beb5a2d4c03457ee32b205751
```

## Output Files

| File | Description |
|---|---|
| MANIFEST.json | Population, thresholds, candidates summary |
| SOURCE_BINDING.json | Full lineage digests for all 7 source roots |
| threshold_metrics.csv | Per-threshold metrics (A/B × 9 rows) |
| episode_threshold_ledger.jsonl | 3600 rows (200 eps × 2 models × 9 thresholds) |
| baseline_episode_ledger.jsonl | 200 rows, per-episode baseline details |
| baseline_metrics.csv | Baseline summary rows |
| score_diagnostics.csv | Per-model score separation diagnostics |
| AUDIT.json | Self-audit checklist (all P0/P1 PASS) |
| commands.txt | Exact CLI invocation |
| SHA256SUMS + SHA256SUMS.sha256 | Root seal |

## Development Gate Status

```
R7_R1_FORMAL_ARTIFACT           = PASS
R7_R2_SUBMITTED_ROOT            = PRESERVE / INVALID_FOR_SCIENTIFIC_CLAIMS
R7_R2_1_CORRECTIVE_REPLAY       = COMPLETE
R7_R2_1_ROOT_SHA256SUMS         = e7cc1700...
R7_R2_1_V5_A_RECALL             = 0.1154 (3/26 at tau=0.1)
R7_R2_1_V5_B_RECALL             = 0.0000 (0/26 at all thresholds)
R7_R2_1_BASELINE_RECALL         = 0.0000
R7_R2_1_SCORE_SEPARATION        = NEGATIVE (both models)

R7_R3_TRAINING                  = HOLD (awaiting authorization)
R7_R4_EXACT_PREFIX              = HOLD
R7_R5_ATTACK_CANARY             = HOLD

FIT_DEV_READ                    = NOT READ
CAL_READ                        = NOT READ
CHECK_READ                      = NOT READ
CS200_READ                      = NOT READ
ATTACK_EXECUTED                 = FALSE
SOURCE_ARTIFACT_MUTATION        = 0
PROTECTED_SPLIT_READS           = 0
ATTACK_OR_MANUAL_OUTCOME_READS  = 0
```

## Next Steps (NOT AUTHORIZED)

R7.3 training of K10-specific detectors remains the logical next step. The corrected replay confirms that Physics-tier checkpoints provide negligible K10 signal, but the qualitative question of whether K10-specific training can achieve usable recall remains unanswered. All subsequent phases (R7.3, R7.4, R7.5) remain HOLD pending independent audit authorization.
