# R7 Clean Opportunity Offline Replay Handoff

**Date:** 2026-07-19
**PR:** [#87](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/87)
**Branch:** `agent/official-v3-r7-k10-v1.2.1-20260719`

## Executive Summary

R7.1 produced sealed K10 opportunity labels (109/800 FIT episodes with feasible K=10 gripper-critical starts). R7.2 replayed frozen V5 Physics checkpoints against these labels. Neither V5-A (proprio-only) nor V5-B (proprio+policy-intent) transferred to K10 detection — both achieved near-zero recall (0.013 and 0.004). The first-eligible-close baseline also scored 0.0 recall, confirming that K10 feasible starts occur after stable_grasp+manipulation_active is established, well past the initial close.

**Conclusion:** Frozen Physics checkpoints do not provide K10 localization signal. R7.3 training of new K10-specific detectors is warranted.

---

## R7.1: K10 Label Closure

| Metric | Value |
|---|---|
| FIT identities | 800/800 |
| Feasible K10 episodes | 109 (13.6%) |
| Total feasible starts | 7,399 |
| Per-suite feasible: L10 | 88 |
| Per-suite feasible: Goal | 9 |
| Per-suite feasible: Object | 3 |
| Per-suite feasible: Spatial | 9 |
| Independent auditor | PASS |
| Protected reads | 0 |
| Attack/manual reads | 0 |
| Source mutation | 0 |

### Server artifacts

```
Teacher root:  OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V21_7e876c2_20260719
Teacher SHA:   18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da
Label root:    OFFICIAL_V3_R7_K10_OPPORTUNITY_LABELER_V1_2_1_8e4f5ff_20260719
Label SHA:     d7defce8d3551725537a15837d9a916aa8ee5cdf012e71828da810665810fb56
Audit root:    OFFICIAL_V3_R7_K10_V121_INDEPENDENT_AUDIT_8e4f5ff_20260719
Audit status:  PASS
Labeler commit: 8e4f5ff727d279ccd3a3566f46e6b94034a8a54b
```

---

## R7.2: Offline Replay Methods

### Replayed checkpoints

| Checkpoint | Variant | Training | Architecture |
|---|---|---|---|
| V5-A | V5_A_PROPRIO | Physics tier, 160-ID subset, 10 epochs, FP32 | 25D proprio → GRU128 → utility+release+regrasp |
| V5-B | V5_B_PROPRIO_POLICY | Physics tier, 160-ID subset, 10 epochs, FP32 | 25D proprio + 9D policy intent → dual GRU + gate + fusion |

### Evaluation protocol

- Population: Fold-0 validation identities (states 0-4, 200 episodes, 234 threshold-expanded)
- One-shot scheduler: emit at first step where utility score >= threshold within candidate_close window
- Threshold sweep: [0.1, 0.2, ..., 0.9] — diagnostic only, no threshold selected
- K10 target: `is_feasible_start` from sealed K10 label root
- No training, no threshold selection, no protected split reads

### Causal baselines

- **First-eligible-close**: emit at first `candidate_close=True` step in the episode

### Metrics reported

- K10 feasible-hit recall
- Emit precision
- No-corridor abstention
- Per-threshold hit/emit/false counts

---

## R7.2 Results

### V5-A (Physics proprio)

| Threshold | Recall | Precision | Hits/Feasible | Emits |
|---|---|---|---|---|
| 0.1 | 0.013 | 0.052 | 3/234 | 58 |
| 0.2 | 0.013 | 0.079 | 3/234 | 38 |
| 0.3 | 0.000 | 0.000 | 0/234 | 23 |
| 0.4 | 0.000 | 0.000 | 0/234 | 10 |
| 0.5 | 0.000 | 0.000 | 0/234 | 7 |
| 0.6-0.9 | 0.000 | — | 0/234 | 1-6 |

### V5-B (Physics proprio + policy intent)

| Threshold | Recall | Precision | Hits/Feasible | Emits |
|---|---|---|---|---|
| 0.1 | 0.004 | 0.024 | 1/234 | 42 |
| 0.2 | 0.004 | 0.033 | 1/234 | 30 |
| 0.3 | 0.004 | 0.050 | 1/234 | 20 |
| 0.4 | 0.004 | 0.067 | 1/234 | 15 |
| 0.5-0.9 | 0.000 | — | 0/234 | 4-12 |

### First-eligible-close baseline

| Metric | Value |
|---|---|
| Recall | 0.000 |
| Precision | 0.000 |
| Hits/Feasible | 0/26 |
| Emits | 200 |

### Interpretation

1. **Both V5 Physics checkpoints show essentially zero transfer to K10.** The Physics tier ranking task (predicting utility/release/regrasp scores) is a different objective from K10 burst feasibility detection. The utility scores do not correlate with K10 feasible starts.

2. **The first-eligible-close baseline always emits too early.** K10 feasible starts require stable_grasp >= 0.5 AND (lift >= 0.3 OR support_removed >= 0.3 OR target_progress > 0.05), conditions that are met well after the initial gripper close. The first candidate_close step is never a feasible start.

3. **R7.3 training is warranted.** New detectors trained directly on the K10 burst_feasible target are needed to provide localization signal.

---

## Development Gate Status

```
R7_R1_K10_LABELS                = PASS (sealed, independently audited)
R7_R2_OFFLINE_REPLAY            = PASS (complete, 0 protected reads)
R7_R2_V5_A_K10_RECALL           = 0.013 (near-zero, no transfer)
R7_R2_V5_B_K10_RECALL           = 0.004 (near-zero, no transfer)
R7_R2_BASELINE_RECALL           = 0.000 (first-close never hits K10)
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

---

## Server Artifacts

```
V5-A checkpoint:
  OFFICIAL_V3_DETECTOR_V5_PHYSICS_A_SMOKE_F0_S20260717_7fe4963_20260719
  checkpoint seal: 77ac6ac0ce0edc2d8be51477ff671180a6e4b6b26c9102b28a068cafcb352ac9

V5-B checkpoint:
  OFFICIAL_V3_DETECTOR_V5_PHYSICS_B_SMOKE_F0_S20260717_7fe4963_20260719
  checkpoint seal: c44a05ceca9130ff147ab1e1290461319309eadd0bdcef27c05e1c6d3e2a4a97

R7.2 replay root:
  OFFICIAL_V3_R7_K10_V5_OFFLINE_REPLAY_456bf73_20260719
  SHA256SUMS: f13a83efec6b2431507ceaef3376c30e0ba0091beb5a2d4c03457ee32b205751
```

---

## Next Steps (NOT AUTHORIZED)

R7.3 would train at most two K10-specific detector candidates (R7-SLinear and R7-A-GRU) on Fold-0 training data against the sealed K10 labels. R7.4 would build the exact-prefix branching system. R7.5 would run a FIT-only command-OPEN + VIS joint canary on 16 pre-registered parents. All remain HOLD pending independent audit authorization.
