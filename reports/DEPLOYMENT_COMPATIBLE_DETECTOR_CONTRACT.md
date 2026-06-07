# Deployment-Compatible Detector Contract

**Date**: 2026-06-06
**Status**: FALSIFICATION — current detector uses offline-only features

---

## Critical Falsification Finding

The current "best" detector (V0_gold/LR/D_causal_safe, BalAcc=0.714) is trained on
features that are **only available AFTER running a VIS attack**:

| Feature | Category | Available at Deploy? |
|---------|----------|---------------------|
| `qpos_opening_delta` | offline (B) | NO — measured from VIS trace |
| `vis_open_count` | offline (B) | NO — counted from VIS frames |
| `action_bridge_confounded` | offline (B) | NO — from VIS attack analysis |
| `label_action_bridge` | oracle (C) | NO — label from VIS outcome |
| `label_physical_response` | oracle (C) | NO — label from VIS outcome |
| `done` | offline (B) | NO — task outcome under VIS |

**This means the detector is an ATTACK OUTCOME CLASSIFIER, not a vulnerability predictor.**
It would be useless at deployment — these features don't exist before running an attack.

---

## Deployment Pipeline Contract

### Input (Online — Available at Deployment)

```
clean_rollout_sequence:
  - observations[t-window_start : t-window_end]  (proprio, action, qpos)
  - task_key (string)
  - state_id (int, from env)
  - window_start, window_end (int, step indices)
```

### Allowed Online Features

| Feature | Source | Category |
|---------|--------|----------|
| `task_key` | env | A |
| Clean qpos statistics (mean, std, min, max, delta) | clean rollout | A |
| Clean action statistics (mean, std, delta) | clean rollout | A |
| Clean gripper_width statistics | clean rollout | A |
| Window position (normalized step / episode_length) | derived | A |
| `predicted_phase` | ProprioNoStep (if available) | A |
| `phase_confidence` | ProprioNoStep (if available) | A |
| `hazard_score_mean` | ProprioNoStep (if available) | A |
| `hazard_score_max` | ProprioNoStep (if available) | A |
| `release_safe_score_mean` | ProprioNoStep (if available) | A |
| `release_safe_score_min` | ProprioNoStep (if available) | A |
| `qpos_phase_class` | heuristic (clean qpos) | A |
| `phase_bin_proxy` | heuristic (clean rollout) | A |

### Explicitly Disallowed (Offline or Oracle)

| Column | Reason |
|--------|--------|
| `qpos_opening_delta` | Requires VIS trace |
| `vis_open_count` | Requires VIS trace |
| `action_bridge_confounded` | Post-hoc analysis |
| `label_action_bridge` | Oracle |
| `label_physical_response` | Oracle |
| `label_task_failure` | Oracle / VIS outcome |
| `done` | VIS outcome |
| `task_failure` | VIS outcome |
| `qpos_label` | From VIS outcome |
| `qpos_delta` | From VIS trace |
| `mechanism_type` | Oracle label |
| `taxonomy` | Oracle label |
| `label_status` | Oracle label |
| `label_source` | Oracle label |
| `provenance_status` | VIS outcome |
| `status` | VIS run status |

---

## Online Pipeline

```
Observation sequence (clean rollout only)
    │
    ├─→ Phase Context Module (ProprioNoStep, 3/9 tasks)
    │     Output: hazard_score, release_safe_score, predicted_phase
    │
    ├─→ Feature Builder (online-safe)
    │     Output: clean_qpos_stats, clean_action_stats, phase_features
    │
    ▼
┌─────────────────────────────────────────┐
│  ONLINE VULNERABILITY DETECTOR          │
│  Input: ONLY online-safe features (A)   │
│  Output: vulnerability_score ∈ [0,1]    │
│  Training: on labels from VIS outcomes  │
│            (offline labels for training  │
│             are OK; features must be     │
│             online-available)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  ONLINE DECISION LAYER                  │
│                                         │
│  vuln_score > threshold:                │
│    → FLAG_FOR_VIS_TESTING               │
│    → NEVER "alarm" without VIS confirm  │
│                                         │
│  vuln_score low:                        │
│    → SUPPRESS                           │
│                                         │
│  vuln_score medium + uncertainty:        │
│    → NEEDS_EXPLORATION                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  OFFLINE CONFIRMATION (post-VIS)        │
│                                         │
│  Run VIS attack → measure outcome       │
│  Classify mechanism from VIS trace      │
│  Update label: confirmed positive/neg   │
│  Feed back into training data           │
└─────────────────────────────────────────┘
```

---

## Key Design Rule

```
ONLINE: predict vulnerability BEFORE attack using ONLY clean rollout
OFFLINE: confirm vulnerability AFTER attack using VIS trace + mechanism audit

The detector MUST NOT use offline features as inputs.
The offline analysis CAN use VIS outcomes for label building.
```

---

## What This Means for Detector v3

1. **D_causal_safe features are BLOCKED** — they contain offline-only columns
2. **Online-safe features are weaker** — expect lower performance than current 0.714 BalAcc
3. **A_task_key_only is the only CLEAN baseline** — task identity as the sole predictor (BalAcc likely ~0.5)
4. **Phase features can add signal** — but only for 3/9 tasks
5. **Clean rollout statistics** need to be engineered as the primary feature source

### Recommended v3 Training (post calibration + confirmed negatives)

```
Feature set: Online-Safe
  - task_key (one-hot)
  - clean_qpos_stats (mean, std, min, max, delta over window)
  - clean_action_stats (mean, std, delta over window)
  - clean_gripper_width_stats (mean, std, delta over window)
  - window_position (normalized step / episode_length)
  - phase_features (hazard_score_mean/max, release_safe_score_mean/min) WHERE available
  - phase_bin_proxy (heuristic)

Target: vulnerability label from 3R VIS (offline — used for training only)
Training: LR or RF on mechanism-pure labels
Evaluation: LOTO by task
```

---

## Gap: Online vs Offline Performance

The difference between online-safe and offline-leaked detector performance
IS the information content of the VIS outcome itself.

If D_causal_safe (offline) → BalAcc = 0.714
And A_task_key_only (online) → BalAcc = ?
Then the gap = leakage penalty

We must measure this gap explicitly and report it.
