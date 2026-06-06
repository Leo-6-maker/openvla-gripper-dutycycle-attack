# Online Clean-Forward Window Detector — Pipeline Contract

**Date**: 2026-06-07
**Status**: Draft v0
**Replaces**: ProprioNoStep-as-detector, no-env Active Probe surrogate

---

## 1. Objective

Build a detector that, from **clean rollout observations only** (no VIS attack outcome),
identifies gripper-contact-phase windows where a **low-budget inference-time VIS PGD attack**
(budget ≤ 20 steps) is likely to induce sustained decoded gripper OPEN.

The detector output feeds a **trigger policy** that decides when to launch a VIS attack
within a budget-constrained online deployment.

---

## 2. Allowed Online Features

All features must be computable from **causal history only** (current and past timesteps).
No future information. No VIS attack outcome. No task success.

### 2.1 Explicitly Allowed

| Feature group | Source | Description |
|---|---|---|
| **Proprioception** | env state | gripper qpos, width, EEF position/velocity, joint positions |
| **Action history** | decoded clean actions | past N action vectors (7-dim: 6 arm + 1 gripper) |
| **Action statistics** | window aggregations | gripper_open_rate, gripper_std, arm_velocity, arm_displacement |
| **Gripper logits** | model forward pass | logsumexp(open_tokens), max(non_open_tokens), open_margin, open_entropy |
| **Action token entropy** | model output | per-dimension entropy of discrete action tokens |
| **Visual embedding** | model vision encoder | PCA-compressed or pooled visual hidden states |
| **Temporal statistics** | sliding window | rolling mean/var/max of all above features |
| **Task context** | language instruction | one-hot task embedding or text embedding |
| **Window position** | derived | step_index / episode_length, window_start / window_end |
| **Phase heuristics** | heuristic | qpos-based OPEN/CLOSE detector, contact proximity proxy |

### 2.2 Explicitly Forbidden (Leakage)

| Forbidden feature | Reason |
|---|---|
| VIS attack outcome (vis_open_count, qpos_delta) | Online unavailable — would require running the attack |
| Task success / done | Causal leakage — success is determined after the episode |
| Manual mechanism label (physical_bridge, taxonomy) | Human annotation not available online |
| Future timestep features | Causal leakage |
| 3R (3-repeat) confirmation result | Requires offline analysis |
| Attack trace fields (token_flip_rate from VIS, PGD loss curves) | Would require running the attack |
| ProprioNoStep hazard score as feature | Proven 0/7 POC recall; not useful for pre-grasp windows |
| Full VIS trace per-frame data | Online unavailable |

---

## 3. Label Definitions

### 3.1 command_susceptible_label

**Definition**: A window is command-susceptible if a **targeted VIS PGD attack**
(PGD20+, prefix_locked_gripper_open_margin, eps=6/255) induces **sustained decoded gripper OPEN**
that a **matched random Linf perturbation does NOT reproduce**.

**Positive criteria** (all must hold):
1. VIS targeted open_count >= 6 (of 18 attack frames)
2. VIS targeted longest_open_streak >= 3
3. Matched random open_count < VIS targeted open_count - 3 (contrast)
4. Provenance: clean (no infra/manual/polluted contamination)
5. Attack valid: PGD optimization converged, no OOM/infra error

**Negative criteria** (at least one):
1. Valid VIS attack produces open_count < 3 AND longest_open_streak < 2
2. Matched random reproduces similar OPEN as targeted (no contrast)
3. Provenance clean

**Excluded from train/eval**:
- Polluted traces (VIS contaminated)
- Pending 1R failures (unconfirmed)
- Infra errors
- Manual-merge rows with unresolved conflicts

### 3.2 physical_bridge_label

**Definition**: A window has physical bridge if command_susceptible AND
the induced OPEN actually causes **measurable gripper qpos opening** (qpos_delta >= 0.02, qpos_label = strong/weak).

This is a **downstream label** — the detector's primary target is command_susceptible.
physical_bridge is evaluated separately as a harder, deployment-relevant outcome.

### 3.3 Label provenance tiers

| Tier | Criteria | Use |
|---|---|---|
| **Gold** | 3R confirmed, random contrast passed, clean provenance | train + eval |
| **Silver** | 1R positive with random contrast, clean provenance | train only (no eval) |
| **Bronze** | 1R positive without random contrast | exploratory only |
| **Reject** | Polluted, pending 1R failure, infra error, manual conflict | excluded |

---

## 4. Window Definition

- **Window**: contiguous step range [ws, we] in a clean LIBERO Object rollout
- **Window source**: existing labeled windows from `object_phase_response_labels_v2.csv`
- **Window constraint**: we - ws >= 4 (minimum 5 steps for temporal features)
- **Window phase**: predominantly pre-grasp or grasp-formation phases
- **Maximum steps**: 300 (LIBERO Object episode length)

---

## 5. Train/Eval Split

### 5.1 Leave-Task-Out (primary)

Train on 7 tasks, eval on 1 held-out task. Rotate through all 8 LIBERO Object tasks.
Report mean ± std across tasks.

### 5.2 Leave-State-Out (secondary)

Train on 80% of states (seeds), eval on 20% held-out. All tasks represented in both splits.

### 5.3 Holdout constraint

- Gold-label windows only in eval
- Silver-label windows in train only
- No window appears in both train and eval

---

## 6. Budget Metrics

### 6.1 Attack Budget

| Budget tier | PGD steps | Restarts | Effective cost | Use case |
|---|---|---|---|---|
| Micro | 5 | 1 | 5 | Canary/probe |
| Low | 10 | 1 | 10 | Online candidate screen |
| Medium | 20 | 1 | 20 | Confirmation |
| Full | 40 | 3 | 120 | Offline label generation |

### 6.2 Detector Efficiency Metrics

- **Budget reduction** = (windows attacked with detector) / (windows attacked without detector)
- **Recall@Budget** = fraction of true positives found within budget B
- **Precision@K** = fraction of top-K detector-ranked windows that are true positives
- **Attack yield per budget** = true_positives_found / total_PGD_steps_spent

### 6.3 Baselines

| Baseline | Description |
|---|---|
| Random window | Attack random windows in episode |
| Fixed window | Attack fixed [ws, we] interval |
| Phase-only | Trigger on qpos-based OPEN/CLOSE heuristic |
| Task-only | Always attack specific tasks |

---

## 7. Detector Trigger Policy

### 7.1 Deployment modes

**Mode A — Offline candidate ranking**:
- Run detector on full clean rollout
- Rank windows by susceptibility score
- Attack top-K windows with available budget

**Mode B — Online streaming**:
- Run detector at each timestep
- Maintain running window score
- Trigger attack when score exceeds threshold
- Cooldown after attack to avoid repeated triggers

### 7.2 Trigger constraints

- Max 3 attacks per episode (budget limit)
- Min 20 steps between attacks (cooldown)
- Do not attack if gripper already OPEN (qpos > 0.03) — ceiling guard
- Do not attack if arm is moving fast (safety)

---

## 8. Real-Robot Deployment Boundary

### 8.1 What changes from sim to real

| Aspect | Sim | Real |
|---|---|---|
| Visual domain | LIBERO render | Real camera (domain gap) |
| PGD perturbation | RGB pixel space | Same (camera-agnostic) |
| Action execution | MuJoCo physics | Real robot controller |
| Latency | Negligible | ~50-200ms per step |
| Safety constraints | None | Collision avoidance, joint limits |

### 8.2 Deployment gating

Before real-robot deployment:
1. Validate detector on held-out sim tasks (leave-task-out AUROC >= 0.65)
2. Run budget-constrained micro-attack on top-K detected windows (sim)
3. Confirm physical bridge yield >= 50% of command_susceptible yield
4. Domain adaptation audit (visual feature drift sim→real)
5. Safety review (max perturbation magnitude, action clamping, emergency stop)

---

## 9. Implementation Phases

| Phase | Deliverable | Gate |
|---|---|---|
| 1. Contract | This document | Review + approve |
| 2. Feature extraction | `online_window_features_v0.csv` | Coverage audit |
| 3. Dataset | 80-window plan + labels | >= 60 usable rows |
| 4. Detector v0 | LR/RF/TCN baselines | AUROC >= 0.65 on leave-task-out |
| 5. Budget attack | Micro-attack on top-K | Yield confirmation |

---

## 10. Version History

| Version | Date | Changes |
|---|---|---|
| v0 | 2026-06-07 | Initial contract after ProprioNoStep + Active Probe gate fails |
