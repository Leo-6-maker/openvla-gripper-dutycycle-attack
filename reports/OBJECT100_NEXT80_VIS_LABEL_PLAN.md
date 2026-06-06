# Object100 Phase-2 VIS Labeling Plan

**Date**: 2026-06-07
**Prerequisite**: Stage-A Opportunity Localizer — Gate PASS (AUROC 0.997)

## Objective

Generate command_susceptible and physical_bridge labels for 80 windows
to train the Online Clean-Forward Window Detector.

## Label Generation Protocol

Per `COMMAND_SUSCEPTIBLE_LABEL_PROTOCOL_V2.md`:

1. Run VIS PGD20 attack on window [ws, we] (env.step mode, eps=6/255)
2. Run matched random Linf attack on same window
3. Compute per-frame decoded env_gripper (+1=OPEN)
4. Apply criteria: targeted open_count >= 6, streak >= 3, random contrast >= 3
5. Assign label tier: Gold (3R) / Silver (1R+random) / Bronze (1R only)

## Strata for VIS Labeling

### Stratum A: High-Score Opportunity Windows (20)
Top-20 teacher windows by Stage-A Localizer score (most confident positives).
These should be the best attack opportunities.

### Stratum B: Medium/Low-Score Windows (20)
Random sample of teacher windows with mid-range localizer scores.
Tests whether the localizer score correlates with VIS outcome.

### Stratum C: Early/Pregrasp Controls (20)
Windows from early_pregrasp_control stratum.
Verify that non-opportunity windows are NOT command_susceptible.

### Stratum D: Late/Post-Release Controls (20)
Mix of late_noncritical and post_release controls.
Hard negatives that should show no VIS effect (ceiling guard test).

## Budget Estimate

| Item | Count | GPU-min per window | Total GPU-min |
|---|---|---|---|
| VIS PGD20 attack | 80 | ~4 min | 320 |
| Matched random Linf | 80 | ~2 min | 160 |
| **Total** | | | **~480 min (8 GPU-hours)** |

With 3 GPU pairs parallel: ~2.7 hours.

## Task Balance Target

9 LIBERO Object tasks, ~9 windows per task.
Adjust for task difficulty (more windows for harder tasks).

## Not Yet

- Do NOT start VIS labeling without explicit approval
- Do NOT use this for detector training yet
- This plan is only for review

## Next Steps After Approval

1. Write VIS labeling driver script (reuses `vis_rollout_proprionostep_triggered.py`)
2. Launch 3 GPU pairs × 27 windows each
3. Extract command_susceptible + physical_bridge labels
4. Train Online Clean-Forward Window Detector v0
