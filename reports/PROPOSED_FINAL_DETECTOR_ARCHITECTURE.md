# Proposed Final Detector Architecture

**Date**: 2026-06-06  
**Status**: PROPOSAL — not yet implemented

---

## Architecture Diagram

```
Observation Sequence
(clean rollout: proprio, action, qpos)
        │
        ▼
┌─────────────────────────────────────────┐
│         FEATURE BUILDER                 │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ D_causal_safe descriptors       │   │
│  │ - gripper width stats           │   │
│  │ - qpos delta over window        │   │
│  │ - action delta over window      │   │
│  │ - vis_open_count (if VIS trace) │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Phase Context (optional)        │   │
│  │ - hazard_score_mean/max         │   │
│  │ - release_safe_score_mean/min   │   │
│  │ - predicted_phase               │   │
│  │ - phase_confidence              │   │
│  │ [Available: 3/9 tasks]          │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Task Identity (optional)        │   │
│  │ - task_key encoding             │   │
│  └─────────────────────────────────┘   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       VULNERABILITY DETECTOR            │
│                                         │
│  Primary: LR on D_causal_safe           │
│  (V0_gold_only, BalAcc=0.714)           │
│                                         │
│  Future: multi-task detector            │
│  - Head 1: physical_bridge risk         │
│  - Head 2: policy/action sensitivity    │
│                                         │
│  Output: vulnerability_score ∈ [0,1]    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     MECHANISM-AWARE DECISION LAYER      │
│                                         │
│  Inputs:                                │
│  - vulnerability_score                  │
│  - mechanism_type (from taxonomy)       │
│  - provenance_status                    │
│  - phase context (if available)         │
│                                         │
│  Decision table:                        │
│  ┌──────────────────────────────────┐   │
│  │ vuln ∧ physical_bridge          │   │
│  │ → ALARM (physical_vuln_risk)    │   │
│  ├──────────────────────────────────┤   │
│  │ vuln ∧ mechanism_unclear        │   │
│  │ → NEEDS_3R_CONFIRMATION         │   │
│  ├──────────────────────────────────┤   │
│  │ vuln ∧ no_action_bridge         │   │
│  │ → POLICY_ACTION_RISK            │   │
│  ├──────────────────────────────────┤   │
│  │ ¬ vuln ∧ clean_control          │   │
│  │ → SUPPRESS                      │   │
│  ├──────────────────────────────────┤   │
│  │ ¬ vuln ∧ label_missing          │   │
│  │ → MANUAL_REVIEW                 │   │
│  ├──────────────────────────────────┤   │
│  │ provenance ∈ {infra, polluted,  │   │
│  │   gpu37_probation}              │   │
│  │ → EXCLUDE_FROM_TRAIN            │   │
│  └──────────────────────────────────┘   │
│                                         │
│  Output: risk_category + action         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       GUARD / EXPERIMENT SCHEDULER      │
│                                         │
│  ALARM → schedule 3R confirmation       │
│  NEEDS_3R → add to calibration queue    │
│  SUPPRESS → skip                        │
│  MANUAL_REVIEW → flag for human         │
│  EXCLUDE → mark provenance, skip train  │
└─────────────────────────────────────────┘
```

---

## What This Architecture Explicitly Rejects

### 1. Phase Detector as Hard Pre-Vulnerability Gate

```
REJECTED:  phase_gate(vuln) → if not hazard: return safe
REASON:    0/7 positive recall in POC. Phase detector trained on clean
           hazard phases; vulnerability windows are orthogonal.
```

### 2. Single Detector for All Mechanisms

```
REJECTED:  one LR for all positives regardless of mechanism
REASON:    physical_bridge and no_action_bridge are different phenomena.
           Mixing them produces a detector that predicts "any VIS failure"
           rather than "gripper vulnerability."
```

### 3. Phase Detector as Hard Cascade Input

```
REJECTED:  phase_pred → feature → vuln_detector
REASON:    Phase features (hazard_score) are zero for all vulnerability
           windows in current data. Adding them as features adds noise
           without signal.
```

---

## Why Phase Detector Is a Separate Module

The phase detector (ProprioNoStep TCN) measures a different construct:
- **Phase detector**: "Is this window in a physically hazardous phase during CLEAN execution?"
- **Vulnerability detector**: "Does VIS perturbation cause task failure in this window?"

These are orthogonal by design:
- A window can be physically safe (no hazard) but VIS-vulnerable (policy/action sensitive)
- A window can be physically hazardous but VIS-robust (gripper control is stable despite perturbation)

The phase detector's value is as a **mechanism auditor**:
- Does physical_bridge occur during a hazard phase? → confirms physical mechanism
- Does clean control window coincide with hazard phase? → validates control selection
- Does no_action_bridge occur in safe phase? → confirms policy mechanism

---

## Current Coverage Gaps

| Component | Status | Gap |
|-----------|--------|-----|
| Feature builder | EXISTS (D_causal_safe) | Missing: phase features for 6/9 tasks |
| Vuln detector | EXISTS (V0_gold LR, BalAcc=0.714) | Underpowered (22 rows, 9 pos) |
| Mechanism taxonomy | EXISTS (60 rows) | 14 mechanism_unclear need audit |
| Decision layer | PROPOSED (not implemented) | Needs integration script |
| Guard/scheduler | PARTIAL (parallel_overnight, watcher) | No mechanism-aware routing |
| Phase detector | EXISTS (3 tasks, CPU) | Out-of-domain for 6/9 tasks |
| Phase features | EXISTS (387 step scores) | All zero for vulnerability windows |

---

## Implementation Sequence

1. **Merge mechanism taxonomy into decision layer** (CPU-only, script)
2. **Calibration v2 confirms 7 claim_usable positives** (needs GPU pair)
3. **Clean-control 3R confirmation** (needs GPU pair)
4. **Split labels: physical vs policy** (CPU-only)
5. **Retrain detector on mechanism-pure labels** (CPU-only)
6. **Deploy mechanism-aware routing** (CPU-only)
7. **Phase detector: retrain on attack windows, or extend coverage** (CPU+GPU)
