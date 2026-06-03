# VIS Phase Calibration Plan

**Date**: 2026-06-03
**Branch**: `exp/vis-prefix-margin-repair-20260603`

## Motivation

Fixed-window ketchup 10-27 eps6 achieves strong VIS prefix_margin results:
- 4/4 prefix: 18/18 OPEN, qposΔ≈0.038, armL2=0, done=False
- 6/6 random: 0/18 OPEN, done=True

But ProprioNoStep bridge smoke shows late windows (73-110):
- 18/18 generated OPEN at ALL windows
- qposΔ≈0 (no physical opening)
- done=True (task survives)

**Generated OPEN is necessary but not sufficient.** Physical vulnerability is phase-dependent. We need to identify which clean-only phase corresponds to VIS-sensitive physical vulnerability.

## Approach

1. Clean rollout collection (4 tasks × 3+ seeds)
2. Heuristic phase event extraction (T_gripper_close_onset, T_grasp_formation, T_grasp_lock, T_lift_start, T_release_start)
3. Phase labeling per step (6-class or 3-class)
4. Frozen ProprioNoStep alignment audit (T_prop vs phase events)
5. Phase selector options (calibrated offset, 3-class MLP/TCN, VIS-sensitive selector)
6. Phase-conditioned VIS attack evaluation

## Phase Taxonomy

### 6-class (audit/diagnostic)

| Label | Name | Description |
|-------|------|-------------|
| 0 | approach | EEF moving toward object, gripper open |
| 1 | pregrasp | EEF near object, preparing to close |
| 2 | grasp_formation | Gripper closing, stable grasp not yet established |
| 3 | stable_grasp_or_lift | Object held, EEF lifts |
| 4 | carry_or_place | Object transported to target |
| 5 | release_or_done | Natural release or terminal |

### 3-class (for initial selector training)

| Label | Name |
|-------|------|
| A | pre_grasp (approach + pregrasp) |
| B | grasp_formation |
| C | post_grasp (stable_grasp_or_lift + carry_or_place + release_or_done) |

## Deliverables

| File | Description |
|------|-------------|
| tables/phase_alignment_clean_rollouts.csv | Per-step clean features + phase labels |
| tables/proprionostep_phase_alignment.csv | ProprioNoStep trigger vs phase events |
| reports/VIS_PHASE_ALIGNMENT_AUDIT.md | Alignment audit of frozen ProprioNoStep |
| tables/phase_conditioned_vis_provenance.csv | Phase-conditioned VIS results |
| reports/VIS_PHASE_CONDITIONED_ATTACK_RESULT.md | Final attack result |

## Claim Boundaries

**Allowed**: phase alignment audit, frozen ProprioNoStep as late-phase selector, early-grasp vulnerability window, phase-conditioned VIS evaluation.

**Forbidden**: ProprioNoStep-guided VIS established, online detector-triggered VIS solved, detector-training-ready, LIBERO-wide generalization, universal pre-release vulnerability.
