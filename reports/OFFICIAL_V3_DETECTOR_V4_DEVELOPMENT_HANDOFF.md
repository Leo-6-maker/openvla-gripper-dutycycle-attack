# Detector V4 Development Handoff

**Date:** 2026-07-18
**Branch:** `agent/official-v3-detector-v4-development-20260718`
**PR:** https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/85
**Status:** DETECTOR_V4_FIT_DEVELOPMENT = HOLD (architectural bottleneck identified)

## Executive Summary

Detector V4 development completed through Stage 3 (1-fold x 1-seed GPU smoke). The infrastructure — candidate window dataset, Teacher V2 labels, causal feature derivation, training pipeline, and evaluation — is fully operational. Three model configurations (A+L2, B+L2, B+L4) were tested on Fold 0.

**Critical finding: The shared-GRU architecture cannot distinguish valid retention close from invalid close because both share the same "close activity" hidden representation.** Dynamic features (View B: +8 gripper dynamics features) and pairwise ranking loss (L4) show modest improvement but do not solve the fundamental bottleneck.

## What was built

### Protocol configs (sealed before training)
- `configs/DETECTOR_V4_DEVELOPMENT_PROTOCOL_V1.json`
- `configs/DETECTOR_V4_TEACHER_PROTOCOL_V1.json`
- `configs/DETECTOR_V4_FEATURE_PROTOCOL_V1.json`
- `configs/DETECTOR_V4_BASELINE_PROTOCOL_V1.json`
- `configs/DETECTOR_V4_DECISION_CONFIG_V1.json`

### Candidate window dataset
- **Server root:** `OFFICIAL_V3_DETECTOR_V4_CANDIDATE_WINDOWS_V1_5e27d7c`
- **SHA256SUMS:** `999f48041680b9709b894b974e5b409c2cc9d782b3509c8e8714552fb1602de6`
- 1386 candidate windows from FIT 800 episodes
- Classification: 512 VALID_RETENTION, 503 PREMATURE_RELEASE, 352 UNSTABLE_MULTI_CLOSE, 9 CLOSE_WITHOUT_SUPPORT, 10 SUPPORT_WITHOUT_RETENTION
- All 6 B3 false-trigger episodes correctly classified as hard-negatives
- Teacher V2 labels at step level: candidate_close, valid_retention, critical_retention_window, false_trigger_veto, release_imminent, hard_negative_category

### Feature derivation
- View A: Original 25D (same as B3)
- View B: 33D (25D + 8 causal gripper dynamics: delta qpos, delta2 qpos, command deviation, close dwell, time since close onset, recent close count, opening trend, command variance)
- View C: 39D (View B + 6 causal EEF dynamics: velocity, acceleration, vertical velocity, stability, displacement since close onset, action consistency)
- All features strictly causal (no future leakage, no privileged info)

### Model candidates
- Candidate A: View A (25D) + GRU128 + criticality + veto + release_imminent
- Candidate B: View B (33D) + GRU128 + criticality + veto + release_imminent
- Candidate C: View C (39D) + GRU128 + hazard + valid_retention + veto + release_hazard (not yet tested)

### Loss ablations
- L0: Masked BCE (control, same as B3)
- L2: L0 + hard-negative veto BCE with class weighting
- L4: L2 + candidate-window pairwise ranking loss

## GPU smoke results (Fold 0, seed 20260717, cuda:1)

| Run | View | Loss | Crit Hit Rate | Hard-Neg False Emit | Final Loss |
|-----|------|------|---------------|---------------------|------------|
| A+L2 | 25D | L2 | 0.989 | 3/3 (1.0) | 0.108 |
| B+L2 | 33D | L2 | 0.995 | 3/3 (1.0) | 0.102 |
| B+L4 | 33D | L4 | 0.989 | 3/3 (1.0) | 0.120 |

All models achieve criticality hit rates close to the close-only baseline (~0.989-0.995).
None suppress hard-negative false emissions.

### Score analysis

Hard-negative criticality scores (B+L4 vs B+L2):
- `libero_goal/task_07/state_04`: crit max 0.894 (was 0.955 with B+L2) — L4 helps
- `libero_goal/task_09/state_03`: crit max 0.724 (was 0.576 with B+L2) — borderline
- `libero_object/task_05/state_04`: crit max 0.884 (was 0.934 with B+L2) — L4 helps

Veto head analysis:
- Veto probabilities are high (0.84-0.99) on BOTH valid-retention AND hard-negative close windows
- The shared GRU hidden state encodes "close activity exists" — the veto linear head cannot learn the opposite polarity from the same representation
- The veto BCE loss is fundamentally at odds with the criticality BCE loss on the shared hidden state

## Root cause: shared GRU architecture limitation

The GRU processes features → produces hidden state. All linear heads read from this same hidden state.

For close-window steps:
- Criticality head should output: HIGH for valid, LOW for invalid
- Veto head should output: LOW for valid, HIGH for invalid

But the hidden state at ALL close-window steps looks similar ("close activity present"). Two linear layers cannot produce opposing behaviors from identical input vectors. The gradient from one head fights the gradient from the other.

**The pairwise ranking loss (L4) is the most promising direction** because it operates on the criticality scores directly (pushing valid scores above invalid scores) without requiring a separate veto head. However, the current 25D/33D features do not provide enough signal for the GRU to learn this distinction via ranking alone.

## Recommended next steps

### Option 1: Separate veto pathway (most likely to work)

Replace shared GRU with branched architecture:
- GRU_1: processes close-related features → criticality head (detects "close exists")
- GRU_2: processes quality-related features (delta qpos, EEF stability, dwell time) → veto head (detects "close quality is valid")
- EMIT = criticality > tau_crit AND veto < tau_veto

This prevents gradient conflict because each GRU specializes in one task.

### Option 2: Replace GRU with TCN + attention

Temporal Convolution Network with self-attention over close windows:
- TCN encodes per-step features with local receptive field
- Self-attention over the close window captures duration, stability, and internal structure
- Pooling produces a single quality score per window

### Option 3: View C + L4 with higher ranking weight

The EEF dynamics (stability, displacement) should provide the strongest quality signal. Testing View C+L4 with `ranking_weight=2.0` might push hard-negative scores below 0.5 without architectural changes.

### Option 4: Accept close-baseline parity and move to scheduler

If the Detector cannot beat the close baseline on selectivity, the project could pivot to:
- Accept close-only as the candidate gate (it already has ~99% T10 coverage)
- Focus R&D on the scheduler: one-shot, persistence, and timing logic
- Use B3/V4 predictions as a soft score for scheduler ranking rather than hard gate

## Boundaries respected

```
CLEAN_2000                     = IMMUTABLE (no modifications)
FIT/S1/Teacher/B3 roots        = IMMUTABLE (no modifications)
B3_25D 12-run results          = IMMUTABLE (no overwrite)
FIT-DEV states 20-23           = NOT READ
CAL states 24-26               = NOT READ
CHECK states 27-29             = NOT READ
CS200 states 30-49             = NOT READ (preparation only)
Attack rollout                 = NOT STARTED
New output in new directories  = YES (all V4 artifacts in fresh roots)
```

## Server artifact index

```
OFFICIAL_V3_DETECTOR_V4_CANDIDATE_WINDOWS_V1_5e27d7c    999f4804...
OFFICIAL_V3_DETECTOR_V4_SMOKE_A_L2_F0_S20260717         (checkpoint.pt + eval/)
OFFICIAL_V3_DETECTOR_V4_SMOKE_B_L2_F0_S20260717         (checkpoint.pt + eval/)
OFFICIAL_V3_DETECTOR_V4_SMOKE_B_L4_F0_S20260717         (checkpoint.pt + eval/)
```

## Mutation declaration

```
GITHUB MUTATION          = 0 (clean commits, no force push)
SOURCE ARTIFACT MUTATION = 0
CHECKPOINT MUTATION      = 0
PREDICTION MUTATION      = 0
FULL-FIT REFIT           = NOT STARTED
CAL / CHECK              = NOT READ
ATTACK                   = NOT STARTED
FIT-DEV                  = HOLD UNTIL USER AUTHORIZATION
```
