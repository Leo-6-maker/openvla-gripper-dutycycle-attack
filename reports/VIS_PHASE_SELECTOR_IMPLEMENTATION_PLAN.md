# VIS Phase-Selector Implementation Plan

**Date**: 2026-06-03
**Branch**: `exp/vis-prefix-margin-repair-20260603`

## Why This Is Needed

### Evidence

1. **Fixed early window [10,27]**: VIS generates 18/18 OPEN, qpos opens (Δ≈0.038), task fails, armL2=0. Random clean. → **VIS-vulnerable**.

2. **ProprioNoStep late windows [73-110]**: VIS generates 18/18 OPEN, but qpos Δ≈0, task survives. → **Action-positive, physical-negative**.

3. **Phase audit**: ProprioNoStep T≈93 fires 14 steps AFTER natural clean OPEN at step 79. It's a **late-phase selector**, misaligned with the early-grasp VIS vulnerability window.

### Conclusion

**Generated OPEN is necessary but not sufficient for task failure. Physical vulnerability is phase-dependent.** We need a clean-only phase selector that identifies grasp_formation windows.

## Scripts Implemented

| Script | Purpose |
|--------|---------|
| `scripts/diagnostics/build_clean_phase_dataset.py` | Heuristic phase labeling from clean rollout traces |
| `scripts/diagnostics/audit_proprionostep_phase_alignment.py` | ProprioNoStep trigger vs phase event alignment |
| `scripts/train_phase_selector.py` | Clean-only causal TCN phase selector training |
| `scripts/diagnostics/evaluate_phase_selector_windows.py` | Phase prediction → attack window proposals |
| `scripts/vis_phase_conditioned_attack.py` | VIS prefix_margin on phase-selected windows |
| `scripts/diagnostics/audit_phase_conditioned_vis.py` | Phase-conditioned VIS result audit with claim gates |

## Phase Taxonomy

- **6-class**: approach, pregrasp, grasp_formation, stable_grasp_or_lift, carry_or_place, release_or_done
- **3-class**: pre_grasp, grasp_formation, post_grasp (default for initial selector training)

## Feature Schema

Default input features (13-D, clean-only proprioceptive):
gripper_command, gripper_qpos, gripper_width, eef_x/y/z, eef_vx/vy/vz, action_dx/dy/dz, action_gripper

Privileged features (object positions, distances, contact flags) are explicitly EXCLUDED from model input. They may only be used offline for label generation.

## Claim Gates

A window is `claim_usable` only if ALL of:
1. `action_bridge_positive`: VIS OPEN >= window_len - 2
2. `physical_bridge_positive`: VIS qpos_delta_post >= 0.03
3. `task_failure_positive`: VIS done=False
4. `denominator_clean`: random done=True, random OPEN=0
5. `natural_release_confounded=False`: clean natural OPEN ratio <= 0.5

## Forbidden Claims

- ProprioNoStep-guided VIS established
- Online detector-triggered VIS solved
- Broad LIBERO-wide generalization
- Universal phase vulnerability
- Detector-training-ready without train/val/test split
- Calling late-window results "VIS failure" → use "action-positive physical-negative"
