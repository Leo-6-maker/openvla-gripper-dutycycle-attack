# Next Steps to Fully Connect the Pipeline

**Date**: 2026-06-06  
**Based on**: Join audit (77 rows, 7 fully joined), connection eval (5 modes)

---

## Current State

```
Pipeline connectivity:
  Phase ←→ Vuln ←→ Mechanism
   24      58       58      rows covered
    7 rows fully joined (phase+vuln+mech)

Bottlenecks:
  1. Phase detector: only 3/9 tasks (cream_cheese, ketchup, salad_dressing)
  2. Vuln detector: 22 valid eval rows (9 pos, 13 neg) — underpowered
  3. Labels: 7/9 positives are "claim_usable" — need calibration v2 confirmation
  4. Negatives: 9/13 are "no_action_bridge" — wrong mechanism for physical detector
```

---

## Priority Plan

### P0: Complete Pipeline Contract & Join Table ✓ DONE

- [x] `END_TO_END_DETECTOR_PIPELINE_CONTRACT.md` — 5 stages defined
- [x] `end_to_end_detector_join_table.csv` — 77 rows, all sources joined
- [x] `END_TO_END_DETECTOR_JOIN_AUDIT.md` — coverage report
- [x] `END_TO_END_DETECTOR_CONNECTION_EVAL.md` — 5 modes evaluated
- [x] `VULNERABILITY_LABEL_TARGET_REVIEW.md` — label definition clarified
- [x] `PROPOSED_FINAL_DETECTOR_ARCHITECTURE.md` — architecture proposed

### P1: Run Config-Matched Calibration v2

**Why**: 7/9 gold positives are "claim_usable" — need 1R vs 3R match to confirm.
**Candidates**: `vis_1r_vs_3r_calibration_v2_candidates.csv` (10 rows: 5 pos, 5 neg)
**GPU**: Needs 1 pair for ~6 hours
**Blocked by**: All 3 GPU pairs busy on overnight adaptive
**When**: After overnight depletes or one pair is freed
**Output**: Confirmed positives with matched 1R+3R == OPEN agreement

### P2: Run Clean-Control 3R Confirmation

**Why**: 22 clean controls are "candidate-derived" — need 3R to confirm they're true negatives.
**Candidates**: `clean_rollout_control_negative_candidates.csv` (12 candidates)
**Goal**: >=6 confirmed control negatives
**GPU**: Needs 1 pair for ~4 hours
**Blocked by**: Same as P1
**Output**: Confirmed negatives for detector v3 training

### P3: Split Labels by Mechanism Type

**Why**: Current labels mix physical_bridge (positives) with no_action_bridge (negatives).
A mechanism-pure detector needs mechanism-pure labels.
**CPU only**: No GPU needed
**Blocked by**: Nothing (can do now)
**Output**: `tables/labels_v3_mechanism_split.csv`
- physical_bridge_positive subset (target: positive for physical detector)
- no_action_bridge_positive subset (target: positive for policy detector — currently 0 rows)
- clean_control_negative subset
- confirmed_negative subset

### P4: Train Mechanism-Separate Detectors

**Why**: Single detector on mixed negatives over-predicts on no_action_bridge (FPR=0.462).
**CPU only**: No GPU needed
**Blocked by**: P1 (positives need confirmation), P2 (negatives need confirmation)
**Output**:
- `detector_v3_physical_bridge/` — trained on physical_bridge_positive + confirmed_negative + clean_control
- `detector_v3_policy_sensitivity/` — trained on no_action_bridge_positive (when available)

### P5: Deploy Mechanism-Aware Decision Layer

**Why**: Mode C (mechanism routing) is the recommended approach — separates alarm from confirmation.
**CPU only**: Script that takes join table row and outputs risk_category + action.
**Blocked by**: P3 (need mechanism-split labels), P4 (need trained detectors)
**Output**: `scripts/deploy_mechanism_aware_decision_layer.py`

### P6: Phase Detector — Extend or Replace

**Why**: Current phase detector covers 3/9 tasks, hazard_score=0 for ALL vulnerability windows.
Options:
  a. Retrain on attack-relevant windows (VIS traces, not clean rollouts)
  b. Train a new phase detector on all 9 object tasks
  c. Accept limited coverage (3 tasks) and use only for audit

**CPU+GPU**: Training needs GPU; inference is CPU
**Blocked by**: Decision on phase detector's role
**Recommendation**: Option (c) for now — use as audit tool for covered tasks.

---

## Dependency Graph

```
P0 (contract+join) ✓ DONE
    │
    ├── P3 (split labels) ── can start NOW (CPU)
    │       │
    │       ▼
    ├── P1 (calibration v2) ── blocked by GPU availability
    │       │
    │       ▼
    ├── P2 (clean-control 3R) ── blocked by GPU availability
    │       │
    │       ├── P4 (train mechanism-separate detectors) ── CPU
    │       │       │
    │       │       ▼
    │       └── P5 (deploy decision layer) ── CPU
    │
    └── P6 (phase detector extend/replace) ── needs decision
```

---

## What NOT to Do

- Do NOT train detector v3 on current mixed labels
- Do NOT use phase detector as hard gate
- Do NOT promote 1R results to silver without calibration v2
- Do NOT claim pipeline is "connected" until P1+P2+P5 are done
- Do NOT interrupt overnight adaptive (P0: keep running)

---

## Target End State

A deployable vulnerability detector pipeline that:
1. Takes a (task, state, window) candidate
2. Extracts causal features (Stage B)
3. Scores vulnerability (Stage C)
4. Classifies mechanism (Stage D)
5. Routes to alarm / confirmation / suppress / review (Stage E)
6. Phase detector provides mechanism audit context, NOT a hard gate

**Estimated time to end state**: 2-4 weeks, depending on GPU availability for P1+P2.
