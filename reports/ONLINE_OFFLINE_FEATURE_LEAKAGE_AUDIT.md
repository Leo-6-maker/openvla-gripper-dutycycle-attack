# Online/Offline Feature Leakage Audit

**Date**: 2026-06-06 15:37
**Purpose**: Identify all columns that would NOT be available at deployment inference time

---

## Category Definitions

| Category | Name | Availability |
|----------|------|-------------|
| **A** | online_available_before_attack | Available from clean rollout only — OK for deployment |
| **B** | offline_after_attack_only | Requires running VIS attack first — NOT available at inference |
| **C** | label_oracle | Ground truth label — NEVER available at inference |
| **D** | metadata | Audit/experiment metadata — not a feature |

---

## Feature Set Leakage Analysis

| Feature Set | Online Cols | Offline Leaks | Oracle Leaks | Risk |
|-------------|-------------|---------------|--------------|------|
| A_task_key_only | task_key | none | none | CLEAN |
| B_phase_bin_only | phase_bin_proxy, predicted_phase | none | none | CLEAN |
| C_closed_pregrasp_gate | phase_bin_proxy, phase_is_critical | none | none | CLEAN |
| D_causal_safe | NONE | qpos_opening_delta, vis_open_count, action_bridge_confounded, label_action_bridge, label_physical_response, label_task_failure, qpos_label, done | none | BLOCKED_LEAKAGE_RISK |
| E_phase+causal | phase_bin_proxy | qpos_opening_delta, vis_open_count, action_bridge_confounded | none | BLOCKED_LEAKAGE_RISK |
| F_task+phase | task_key, phase_bin_proxy, predicted_phase | none | none | CLEAN |
| G_task+phase+causal | task_key, phase_bin_proxy | qpos_opening_delta, vis_open_count, action_bridge_confounded | none | BLOCKED_LEAKAGE_RISK |
| H_descriptor_upper | NONE | qpos_opening_delta, vis_open_count, qpos_label, label_action_bridge, label_physical_response | none | BLOCKED_LEAKAGE_RISK |

---

## Critical Finding

### BLOCKED: 4/8 feature sets use offline-only or oracle columns

These feature sets CANNOT be used in a deployed online detector:

- **D_causal_safe**: offline leaks: `qpos_opening_delta, vis_open_count, action_bridge_confounded, label_action_bridge, label_physical_response, label_task_failure, qpos_label, done`. 
- **E_phase+causal**: offline leaks: `qpos_opening_delta, vis_open_count, action_bridge_confounded`. 
- **G_task+phase+causal**: offline leaks: `qpos_opening_delta, vis_open_count, action_bridge_confounded`. 
- **H_descriptor_upper**: offline leaks: `qpos_opening_delta, vis_open_count, qpos_label, label_action_bridge, label_physical_response`. 

### CLEAN: 4/8 feature sets use only online-available columns

- **A_task_key_only**: task_key
- **B_phase_bin_only**: phase_bin_proxy, predicted_phase
- **C_closed_pregrasp_gate**: phase_bin_proxy, phase_is_critical
- **F_task+phase**: task_key, phase_bin_proxy, predicted_phase

---

## The Central Leakage Problem

The **D_causal_safe** and **H_descriptor_upper** feature sets — which produced
the best detector performance (BalAcc=0.714, posRecall=0.889) — are built on
columns that are ONLY available AFTER running a VIS attack:

- `qpos_opening_delta` — measured from VIS trace, not available before attack
- `vis_open_count` — count of VIS frames with gripper open
- `action_bridge_confounded` — whether action bridge was confounded during VIS
- `label_action_bridge` — oracle label of action bridge from VIS outcome
- `label_physical_response` — oracle label of physical response

**This means the current "best" detector (V0_gold D_causal_safe LR) is trained
on oracle features. It would be useless at deployment time because these features
don't exist before running an attack.**

This is a FUNDAMENTAL leakage: the detector is being evaluated on features that
encode the attack outcome, making it an outcome classifier rather than a
vulnerability predictor.

---

## Online-Safe Feature Whitelist

The ONLY columns that can be used in a deployed online detector:

| Column | Source | Description |
|--------|--------|-------------|
| `task_key` | env | Task identity |
| `state_id` | env | Episode state ID |
| `window_start` | env | Window start step |
| `window_end` | env | Window end step |
| `predicted_phase` | ProprioNoStep | Phase bin prediction |
| `phase_confidence` | ProprioNoStep | Phase confidence |
| `hazard_score_mean` | ProprioNoStep | Mean hazard score |
| `hazard_score_max` | ProprioNoStep | Max hazard score |
| `release_safe_score_mean` | ProprioNoStep | Mean release-safe |
| `release_safe_score_min` | ProprioNoStep | Min release-safe |
| `phase_bin_proxy` | heuristic | Heuristic phase bin |
| `qpos_phase_class` | heuristic | Clean qpos phase class |

**Plus**: any features derived from clean rollout observations:
- Clean qpos trajectory statistics
- Clean action trajectory statistics
- Clean gripper width statistics
- Clean proprioception statistics

**Explicitly NOT allowed**:
- Any VIS attack outcome (qpos_delta, vis_open_count, done, task_failure)
- Any oracle label (label_status, mechanism_type, taxonomy)
- Any post-hoc classification (physical_bridge_positive, no_action_bridge)
- Any provenance/confirmation status (3R confirmed, calibration status)

---

## Detection of Current Leakage in Training Data

The detector_v27_phase_aware_dataset.csv contains these leaked columns
as part of the training features. The model learns:

```
f(task_key, qpos_opening_delta, vis_open_count, ...) → vulnerable?
```

But at deployment:
```
f(task_key, ???, ???, ...) → can't compute!
```

The `qpos_opening_delta` is only known AFTER running VIS. At deployment,
we need to predict vulnerability BEFORE running an attack.

---

## Impact on Detector v3

**Detector v3 is BLOCKED on feature leakage as well as label readiness.**

Before training v3:
1. Define an online-safe feature set (only A columns)
2. Rebuild the training dataset with ONLY online-available features
3. Accept that online performance will be LOWER than offline (oracle) performance
4. The gap between online and offline performance IS the leakage penalty

---

## Recommended Feature Set for Deployable Detector

```
Online-Safe Feature Set (proposed):
  - task_key (one-hot or embedding)
  - Clean qpos statistics over window: mean, std, min, max, delta
  - Clean action statistics over window: mean, std, delta
  - Clean gripper_width statistics over window
  - Phase features (where ProprioNoStep available): hazard_score_mean/max,
    release_safe_score_mean/min, predicted_phase, phase_confidence
  - Window position relative to episode (normalized step index)
```

This feature set contains NO attack outcomes and NO oracle labels.
It is the only basis for a deployable detector.

---

## Column Classification Table (90 columns classified)

| Category | Count |
|----------|-------|
| A — online_available_before_attack | 29 |
| B — offline_after_attack_only | 21 |
| C — label_oracle | 13 |
| D — metadata | 27 |

