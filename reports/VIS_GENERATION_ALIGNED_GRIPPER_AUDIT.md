# VIS Generation-Aligned Gripper Audit

**Date**: 2026-06-02
**Commit**: `338d1f4`
**Frame**: cream_cheese_step030
**Budget**: eps_raw=8, steps=40, pgd_restarts=3

## Core Finding

**Teacher-forced open_prob_mass is completely unreliable for OpenVLA autoregressive action generation.**

The model's `_audit_logits` runs a single teacher-forced forward pass with clean action prefix tokens. But actual generation is autoregressive — the PGD perturbation can change earlier arm tokens, which changes the context for gripper token generation. The teacher-forced probability mass on corrected OPEN tokens does not predict whether the generated gripper action will be OPEN.

## Results Per Objective

### gripper_open_region_ce

| Restart | Generated Gripper | is_open | arm_changed | armL2 | tf_open_mass |
|---------|-------------------|---------|-------------|-------|-------------|
| 0 | 31872 / **0.0000** | **True** | 6/6 | 0.518 | 1.0 |
| 1 | 31744 / 0.9961 | False | 5/6 | 0.864 | 1.0 |
| 2 | 31744 / 0.9961 | False | 6/6 | 0.684 | 1.0 |

- 1/3 restarts produce True OPEN
- Changes ALL arm tokens when successful
- tf_open_mass=1.0 on ALL restarts — completely non-discriminative

### prefix_locked_gripper_open_region_ce

| Restart | Generated Gripper | is_open | arm_changed | armL2 | tf_open_mass |
|---------|-------------------|---------|-------------|-------|-------------|
| 0 | 31872 / **0.0000** | **True** | 0/6 | 0.000 | 0.0 |
| 1 | 31744 / 0.9961 | False | 0/6 | 0.000 | 0.0 |
| 2 | 31744 / 0.9961 | False | 0/6 | 0.000 | 0.0 |

- 1/3 restarts produce True OPEN
- **Zero arm drift** when successful
- tf_open_mass=0.0 — anti-correlated with generation success

### prefix_locked_gripper_open_margin (BEST)

| Restart | Generated Gripper | is_open | arm_changed | armL2 | tf_open_mass |
|---------|-------------------|---------|-------------|-------|-------------|
| 0 | 31872 / **0.0000** | **True** | 0/6 | 0.000 | 0.0 |
| 1 | 31872 / **0.0000** | **True** | 0/6 | 0.000 | 0.0 |
| 2 | 31744 / 0.9961 | False | 0/6 | 0.000 | 0.0 |

- **2/3 restarts produce True OPEN**
- **Zero arm drift** on all restarts
- Best gripper-specific objective

### gripper_open_expected_action

| Restart | Generated Gripper | is_open | arm_changed | armL2 | tf_open_mass |
|---------|-------------------|---------|-------------|-------|-------------|
| 0 | 31744 / 0.9961 | False | 0/6 | 0.000 | 0.0 |
| 1 | 31744 / 0.9961 | False | 0/6 | 0.000 | 0.0 |
| 2 | 31872 / **0.0000** | **True** | 0/6 | 0.000 | 0.0 |

- 1/3 restarts produce True OPEN
- Note: this objective optimizes OPEN probability mass, not expected decoded action

## Root Cause: Teacher-Forced vs Autoregressive Disconnect

The teacher-forced audit passes clean action prefix tokens as input to the model. But during autoregressive generation, the model generates each token conditioned on previously generated tokens. When PGD perturbs the image, it changes the generated arm tokens, which changes the context for gripper token generation.

Evidence:
- `gripper_open_region_ce` restart 0: arm_changed=6/6 → different prefix context → gripper OPEN
- `gripper_open_region_ce` restart 1: arm_changed=5/6 → different prefix → gripper CLOSE despite tf_open_mass=1.0
- `prefix_locked` objectives: arm_changed=0/6 → same prefix context → generation matches clean prefix
- When prefix_locked succeeds (restart 0 for region/margin): the PGD found a perturbation that changes gripper logits without changing arm logits → zero arm drift

## Metric Validity

| Metric | Valid? | Reason |
|--------|--------|--------|
| true_generated_open | **YES** | Actual autoregressive re-decode |
| adv_gripper_action < 0.5 | **YES** | Derived from generation |
| adv_gripper_token in corrected OPEN | **YES** | Derived from generation |
| NAD_DoF7 | **YES** | From generated actions |
| armL2 | **YES** | From generated actions |
| arm_tokens_changed | **YES** | From generated tokens |
| teacher_forced_open_prob_mass | **NO** | Anti-correlated with generation for prefix objectives |
| open_after alone | **NO** | Measures wrong space |
| token_flip alone | **NO** | 31744→31745 flips but same action |

## Claim Boundary

**Allowed**:
- Teacher-forced open probability is not reliable for OpenVLA autoregressive action generation
- Generation-aligned restart selection is necessary for gripper-specific VIS
- `prefix_locked_gripper_open_margin` at eps=8 produces true generated gripper OPEN on cream_cheese_step030 with zero arm drift (2/3 restarts)
- This revives the gripper-specific VIS attack line at single-frame no-rollout level

**Not allowed**:
- Task failure, qpos opening, rollout success, multi-task claims
