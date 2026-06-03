# VIS Phase Alignment Audit

**Date**: 2026-06-03
**Task**: ketchup, seed 0

## Summary

Frozen ProprioNoStep triggers at T≈93, which is **14 steps after the start of natural clean OPEN (step 79)**. It is a late-phase / release-phase selector, NOT an early-grasp selector.

## Clean Phase Events (ketchup seed 0, 158 steps)

| Event | Step | Phase |
|-------|------|-------|
| Episode start | 0 | grasp_formation (held over from s0) |
| T_release_start | 79 | Natural OPEN begins (env_gripper=+1 for 3+ consecutive steps) |
| T_prop (ProprioNoStep) | ~93 | Detector trigger |
| T_done | 157 | Task complete |

## Phase Distribution

| Phase | Steps | Description |
|-------|-------|-------------|
| grasp_formation | 0-78 | Gripper closed/holding, object grasped |
| release_or_done | 79-157 | Natural OPEN, object placed |

## ProprioNoStep vs Phase Events

| Metric | Value |
|--------|-------|
| T_prop | ~93 |
| T_release_start | 79 |
| T_prop - T_release_start | +14 steps (after natural OPEN) |
| Phase at T_prop | release_or_done |
| Phase at T_prop - 20 | grasp_formation (step 73) |
| Phase at T_prop - 40 | grasp_formation (step 53) |
| Phase at T_prop - 80 | grasp_formation (step 13) |

## Fixed Window vs ProprioNoStep Windows

| Window | Range | Phase | VIS OPEN | qposΔ | done | Interpretation |
|--------|-------|-------|----------|-------|------|---------------|
| Fixed | 10-27 | grasp_formation | 18/18 | 0.038 | False | **VIS-vulnerable** |
| W-20 | 73-90 | grasp_formation→release | 18/18 | 0.0001 | True | action-positive, physical-negative |
| W-10 | 83-100 | release_or_done | 18/18 | 0.0001 | True | action-positive, physical-negative |
| W0 | 93-110 | release_or_done | 18/18 | 0.0000 | True | action-positive, physical-negative |

## Conclusions

1. **Frozen ProprioNoStep is a late-phase selector**: T≈93 fires during natural release (14 steps AFTER natural OPEN begins at step 79).

2. **VIS sensitivity is phase-dependent**: Early grasp_formation window [10,27] achieves physical qpos opening and task failure. Late release windows [73-110] achieve action OPEN but zero physical response — object is already placed, contact physics prevent gripper opening.

3. **Offset from T_prop to VIS-sensitive phase**: T_prop − 80 ≈ 13, which is within the early vulnerable window [10,27]. A fixed offset of -80 could recover the early window, but this is NOT online-feasible (requires looking back 80 steps from a future trigger).

4. **Generated OPEN is necessary but not sufficient**: All 4 tested windows achieve 18/18 generated OPEN. Only the early window achieves physical opening.

## Recommendation

Proceed to train a 3-class early-grasp phase selector (clean-only, no attacked data) to directly identify pre_grasp→grasp_formation windows, rather than relying on ProprioNoStep's late-phase trigger with a large negative offset.
