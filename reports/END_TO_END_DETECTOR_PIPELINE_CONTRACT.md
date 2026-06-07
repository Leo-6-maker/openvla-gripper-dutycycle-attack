# End-to-End Detector Pipeline Contract

**Version**: 1.0  
**Date**: 2026-06-06  
**Status**: AUDIT — do not implement without review

## Purpose

Define the intended contract for each pipeline stage so that phase detector,
vulnerability detector, mechanism taxonomy, and label policy compose into a
single auditable decision chain. This document defines outputs, not
implementation.

---

## Pipeline Overview

```
Observation sequence (clean rollout proprio/action)
    │
    ▼
┌─────────────────────────────────────┐
│ Stage A: Phase Detector Inference   │
│ Model: ProprioNoStep TCN (CPU)      │
│ Covers: cream_cheese, ketchup,      │
│         salad_dressing only         │
│ Output: phase context features      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage B: Vulnerability Feature      │
│          Extraction                 │
│ Builds: D_causal_safe descriptors,  │
│   qpos/command/width/action stats,  │
│   optional phase features           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage C: Vulnerability Detector     │
│          Scoring                    │
│ Model: LR (canonical: V0_gold_only  │
│   D_causal_safe)                    │
│ Output: vulnerability_score, pred   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage D: Mechanism Taxonomy         │
│ Classifies: physical_bridge,        │
│   no_action_bridge, clean_control,  │
│   mechanism_unclear, infra/manual   │
│ Output: mechanism_type + evidence   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Stage E: Final Risk Interpretation  │
│ Routes by mechanism × vuln_score    │
│ Output: risk_category + action      │
└─────────────────────────────────────┘
```

---

## Stage A: Phase Detector Inference

### Input
| Field | Source | Format |
|-------|--------|--------|
| `task_key` | episode identity | string |
| `state_id` | episode identity | int |
| `window_start` | candidate window | int (step index) |
| `window_end` | candidate window | int |
| Proprio/action sequence | clean rollout trace | array of (step, qpos, action) |

### Output
| Field | Type | Description |
|-------|------|-------------|
| `predicted_phase` | string | phase_bin_proxy value |
| `phase_confidence` | float | model confidence |
| `hazard_score_mean` | float | mean hazard score over window steps |
| `hazard_score_max` | float | max hazard score over window steps |
| `release_safe_score_mean` | float | mean release-safe score |
| `release_safe_score_min` | float | min release-safe score |
| `phase_detector_source` | string | `proprionostep_tcn_cpu` or `checkpoint_agg` |
| `phase_inference_status` | string | `ok`, `out_of_domain`, `model_missing`, `no_data` |

### Constraints
1. Model ONLY covers cream_cheese, ketchup, salad_dressing. All other tasks → `phase_inference_status = out_of_domain`.
2. Phase detector MUST NOT be used as a hard gate (see Section: Hard Gate Rejection).
3. Output is **context features**, not a decision.

---

## Stage B: Vulnerability Feature Extraction

### Input
| Field | Source |
|-------|--------|
| task_key, state_id, window | candidate identity |
| Clean rollout trace | qpos/action sequence |
| Phase features (from Stage A) | optional join |

### Output
| Field | Type | Description |
|-------|------|-------------|
| `D_causal_safe` features | float vector | gripper width, qpos delta, action delta descriptors |
| `qpos_opening_delta` | float | max qpos change in window |
| `vis_open_count` | int | VIS frames with open gripper |
| `task_failure` | bool | did task fail |
| `action_bridge_status` | bool | action→gripper bridge detected |
| `phase_features` | float vector (optional) | hazard_score_mean/max, predicted_phase encoding |

---

## Stage C: Vulnerability Detector Scoring

### Input
| Field | Source |
|-------|--------|
| Feature vector | Stage B output |
| Detector variant | canonical: V0_gold_only / LR / D_causal_safe |

### Output
| Field | Type | Description |
|-------|------|-------------|
| `vulnerability_score` | float [0,1] | probability of vulnerability |
| `vulnerability_pred` | int {0,1} | binary prediction |
| `detector_variant` | string | which model produced this |
| `prediction_status` | string | `ok`, `out_of_domain`, `model_missing` |

### Constraints
1. Model is LR trained on 22 gold rows (9 pos, 13 neg). This is underpowered.
2. Detector v3 is BLOCKED until confirmed negatives >=6 + calibration v2 PASS.
3. Current performance (canonical): TP=8/9, FP=6/13, BalAcc=0.714.

---

## Stage D: Mechanism Taxonomy

### Input
| Field | Source |
|-------|--------|
| All evidence from Stages A-C | |
| VIS trace metadata | qpos_delta, vis_open_count, task_failure |
| Label provenance | gold_v2, 1R_screening, calibration |

### Output
| Field | Type | Description |
|-------|------|-------------|
| `mechanism_type` | enum | see below |
| `qpos_delta` | float | measured gripper opening |
| `vis_open_count` | int | count of VIS frames with opening |
| `task_failure` | bool | task outcome |
| `action_bridge_status` | bool | action→physical coupling |
| `no_action_bridge_flag` | bool | task failure without physical opening |
| `provenance_status` | string | gold/1R/calibration/infra |

### Mechanism Types
| Type | Definition | Current Count |
|------|-----------|---------------|
| `physical_bridge_positive` | VIS → gripper opens → task fails | 9 |
| `command_or_token_positive` | VIS → token/action perturbation → task fails | 0 (not yet classified) |
| `no_action_bridge_positive` | VIS → task fails but gripper does NOT open | 0 (positives are all physical) |
| `clean_control_negative` | Clean/random rollout, no attack, no failure | 22 |
| `confirmed_negative` | 3R VIS confirmed no opening, no failure | 4 (physical_task_negative) |
| `mechanism_unclear` | evidence insufficient to classify | 14 |
| `infra_or_manual` | Infra failure, polluted, manual override | 8 |

---

## Stage E: Final Risk Interpretation

### Input
All outputs from Stages A-D.

### Output
| Field | Type | Description |
|-------|------|-------------|
| `final_risk_score` | float [0,1] | composite risk |
| `final_risk_category` | enum | risk classification |
| `recommended_action` | enum | what to do next |

### Risk Categories
| Category | Condition | Interpretation |
|----------|-----------|---------------|
| `physical_vulnerability_risk` | vuln_high AND physical_bridge evidence | Gripper-level vulnerability confirmed |
| `policy_action_vulnerability_risk` | vuln_high AND no_action_bridge | Policy/action sensitivity, needs 3R confirmation |
| `clean_control_low_risk` | vuln_low AND clean_control | Expected negative, no action |
| `mechanism_unclear_manual_review` | evidence missing | Human review required |
| `needs_3R_confirmation` | 1R result only | Run 3R VIS to confirm |
| `exclude_from_train` | infra/polluted/manual | Never use in training |

### Recommended Actions
| Action | When |
|--------|------|
| `alarm` | physical_vulnerability_risk, confirmed |
| `suppress` | clean_control_low_risk |
| `manual_review` | mechanism_unclear |
| `needs_3R_confirmation` | 1R-only provenance |
| `exclude_from_train` | infra/polluted/probation |

---

## Hard Gate Rejection

### Why phase gate was rejected

The phase detector (ProprioNoStep TCN) was tested as a hard pre-vulnerability gate:
- Phase detector says "hazard" → pass to vulnerability detector
- Phase detector says "safe" → suppress (never evaluate for vulnerability)

**Result**: 0/7 positive recall on covered subset (all vulnerability positives were in "safe" phases).

**Root cause**: The phase detector was trained on clean physical hazard phases. Vulnerability windows that are policy/action-sensitive (no_action_bridge) or early physical bridges exist in phases the model classifies as "safe." The signals are orthogonal.

### Rule
Phase detector MUST NOT be used as a hard gate. It can only:
1. Provide context features for the vulnerability detector
2. Stratify clean controls by physical hazard phase
3. Audit mechanism consistency (does physical_bridge align with hazard phase?)
4. Modulate risk scores (soft factor, never zero)
