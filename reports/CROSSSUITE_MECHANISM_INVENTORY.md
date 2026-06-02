# Cross-Suite Mechanism Inventory

**Date**: 2026-05-31

## Suite Summary

| Suite | Tasks | Pick-Place | Articulated | Multi-Object | Planar/Other |
|-------|-------|-----------|-------------|--------------|-------------|
| LIBERO-Spatial | 10 | 10 | 0 | 0 | 0 |
| LIBERO-Object | 10 | 10 | 0 | 0 | 0 |
| LIBERO-Goal | 10 | 6 | 3 | 0 | 1 |
| LIBERO-10 | 10 | 2 | 2 | 6 | 0 |

## Mechanism Classification

### LIBERO-Spatial (10 tasks — ALL pick_place_transfer)
All tasks: "pick up the black bowl [location] and place it on the plate"
- Mechanism: pick_place_transfer
- Gripper-duty applicable: YES (all)
- Eligible for cross-suite validation: YES (all 10)

### LIBERO-Goal (10 tasks — mixed mechanisms)
| tid | Task | Mechanism | Eligible |
|-----|------|-----------|----------|
| 0 | open_the_middle_drawer_of_the_cabinet | articulated_only | NO |
| 1 | put_the_bowl_on_the_stove | pick_place_transfer | YES |
| 2 | put_the_wine_bottle_on_top_of_the_cabinet | pick_place_transfer | YES |
| 3 | open_the_top_drawer_and_put_the_bowl_inside | articulated+pick_place | BOUNDARY |
| 4 | put_the_bowl_on_top_of_the_cabinet | pick_place_transfer | YES |
| 5 | push_the_plate_to_the_front_of_the_stove | planar_rearrangement | NO |
| 6 | put_the_cream_cheese_in_the_bowl | pick_place_transfer | YES |
| 7 | turn_on_the_stove | articulated_only | NO |
| 8 | put_the_bowl_on_the_plate | pick_place_transfer | YES |
| 9 | put_the_wine_bottle_on_the_rack | pick_place_transfer | YES |

Eligible for sus30 validation: 6 tasks (t1,2,4,6,8,9)

### LIBERO-10 (10 tasks — long-horizon, mixed)
Mostly multi-object transfer and articulated tasks.
- Eligible for clean shadow: 2 pick-place-segment tasks
- Eligible for sus30: DEFERRED — too complex for overnight pilot

## Clean Shadow Rollout Plan (Tonight)

| Suite | Tasks | States | Rollouts | GPU |
|-------|-------|--------|----------|-----|
| Spatial | 10 | 2 | 20 | GPU1,3 |
| Goal (eligible) | 6 | 2 | 12 | GPU2,6 |
| **Total** | **16** | — | **32** | — |
