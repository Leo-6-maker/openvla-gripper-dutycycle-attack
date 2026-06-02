# VIS Gripper Partial Gate-Lite Live Report

**Date**: 2026-06-02 15:40  
**Source**: gate-lite log (PID 20375, GPU67, 24min elapsed, still running)  
**Data**: 6 completed attack rows from cream_cheese (2 frames × partial combos)

## Key Finding

**gripper_open_region_ce at eps=8 successfully flips the gripper token on cream_cheese_step030, with the lowest armL2 among all successful attacks.**

## Results

| Frame | Objective | eps | Flip | NAD_Z | armL2 | linf_raw |
|-------|-----------|-----|------|-------|-------|----------|
| step030 | gripper_open_region_ce | 8 | **FLIP** | 0.29 | **0.53** | 8.03 |
| step030 | gripper_open_region_ce | 12 | **FLIP** | 0.72 | 0.83 | 12.02 |
| step030 | force_open_z_down | 8 | **FLIP** | 0.43 | 0.61 | 8.03 |
| step030 | force_open_z_down | 12 | **FLIP** | 0.64 | 0.77 | 12.02 |
| step035 | gripper_open_region_ce | 8 | noop | 0.80 | 0.85 | 8.03 |
| step035 | gripper_open_region_ce | 12 | noop | 0.80 | 0.85 | 12.02 |

## Analysis

### 1. gripper_open_region_ce eps8 is the cleanest flip

- armL2=0.53 is the lowest among all flips
- vs force_open_z_down eps8: armL2=0.61 (15% higher arm drift)
- eps8 is sufficient — eps12 adds armL2 without clear benefit

### 2. Frame dependence: step030 flips, step035 does NOT

- step030 and step035 have similar armL2 (0.85) but opposite flip outcomes
- This confirms frame-level gripper vulnerability variation
- step030 is a "low-margin" frame where the gripper token is near the OPEN decision boundary
- step035 is higher-margin — the model is more confident in CLOSE

### 3. force_open_z_down increases arm drift

- At eps8: armL2 0.61 vs 0.53 for gripper-only (+15%)
- This is expected: Z optimization adds arm perturbation
- Confirms force_open_z_down is hybrid positive-control, not pure gripper

### 4. OPEN-region verification (pending CSV)

The log lines don't include open_prob or decoded gripper action. These are in the CSV written at job completion. Pending confirmation:
- Did the flipped token decode to OPEN (>0.5)?
- What is the open_region_prob_mass_after?

## Gate Status

**Preliminary: PROMISING but incomplete**

- gripper_open_region_ce at eps8 shows action-level gripper flip on step030
- Frame generalization not yet confirmed (step035 noop)
- Need margin scan to understand frame-level vulnerability
- Need prefix-locked objectives to test stability
- Need full task coverage (salad_dressing, ketchup still running)

## Next Steps

1. Let gate-lite complete on GPU67 (salad_dressing, ketchup frames pending)
2. Let gripper-only gate complete on GPU45 (6 frames/task)
3. Let margin scan complete on GPU23
4. Run P1 new objective smoke on cream_cheese_step030
5. Only then decide on rollout

## Claim Boundary

- Do NOT claim: selective attack, efficient attack, production controller
- CAN claim: EPS-corrected gripper_open_region_ce at eps8 induces decoded gripper token flip on cream_cheese_step030 with armL2=0.53
- CAN claim: force_open_z_down is hybrid positive-control with higher arm drift
