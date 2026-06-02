# VIS Ketchup prefix_margin — Final Result

**Date**: 2026-06-03 (overnight)
**Objective**: `prefix_locked_gripper_open_margin`
**Task**: ketchup (LIBERO-object #4)

## Core Result

**prefix_margin causes reproducible early_grasp_disruption on ketchup across 3 budgets and 2 windows.**

## Prefix Results

| Window | eps | Seeds | OPEN | armL2 | qpos_delta | Done | Mechanism |
|--------|-----|-------|------|-------|-----------|------|-----------|
| 10-27 | 8 | 0,1,2 | 17-18/18 | 0.000 | 0.0375-0.0379 | All False | early_grasp_disruption |
| 10-27 | 6 | 0,3 | 16-18/18 | 0.000 | 0.0375-0.0378 | All False | early_grasp_disruption |
| 10-27 | 4 | 0,2 | 18/18 | 0.000 | 0.0375-0.0378 | All False | early_grasp_disruption |
| 20-37 | 8 | 0 | 18/18 | 0.000 | 0.0384 | False | early_grasp_disruption |
| 20-37 | 6 | 0 | 18/18 | 0.000 | 0.0384 | False | early_grasp_disruption |

**7/7 seeds across eps4/6/8, 2/2 windows, ALL fail. armL2=0.000 on all.**

## Random Controls

| Window | eps | Seeds | OPEN | Done | Specificity |
|--------|-----|-------|------|------|-------------|
| 10-27 | 4 | 3 | 0/18 | 3/3 True | Clean |
| 10-27 | 6 | 6 | 0/18 | 6/6 True | Clean |
| 10-27 | 8 | 6 | 0/18 | 5/6 True | Near-clean |
| 20-37 | 6 | 1 | 0/18 | True | Clean |
| 20-37 | 8 | 1 | 0/18 | True | Clean |

**15/16 random success, 0 OPEN on all.**

## Mechanism

All audited ketchup prefix failures are **early_grasp_disruption**:
- VIS induces OPEN at the first step of the attack window (step 10 or 20)
- qpos transitions from closed (~0.039) to fully open (~0.001) within the window
- Clean natural OPEN occurs at step 63 — VIS preempts by 40-53 steps
- Object is never stably grasped; episode runs to max 299 steps
- armL2=0.000 throughout — gripper-only, no arm drift

## Claim Level

| Level | Status |
|-------|--------|
| Action bridge (generated OPEN) | **Established** |
| Physical bridge (qpos response) | **Established** |
| Same-budget task-level (eps6) | **Established** |
| Budget-compressed (eps4) | **Candidate** (GPU-warning on source) |
| Window-generalized (20-37) | **Established** |
| ProprioNoStep-guided | **Not established** |
| Large-scale LIBERO | **Not established** |
