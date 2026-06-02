# VIS prefix_margin Minimal Rollout V3 — Final Result

**Date**: 2026-06-02
**Commit**: `338d1f4` (post-reboot: `d16ab76`)
**Objective**: `prefix_locked_gripper_open_margin`
**Budget**: eps_raw=8, steps=40, pgd_restarts=3, arm_preserve_weight=0.1, gripper_margin=5.0

## Action Bridge — STRONG PASS

All 5 prefix rollout runs across 3 tasks produce true generated gripper OPEN with zero arm drift.

| Task | Window | Cond | Seed | OPEN | armL2 | qpos_delta | Done |
|------|--------|------|------|------|-------|-----------|------|
| cream | 12-20 | prefix | 0 | 6/9 (67%) | 0.000 | 0.0045 | True |
| cream | 26-36 | prefix | 0 | 6/11 (55%) | 0.000 | 0.0173 | True |
| salad | 10-25 | prefix | 0 | 15/16 (94%) | 0.000 | 0.0372 | True |
| ketchup | 10-27 | prefix | 0 | 17/18 (94%) | 0.000 | 0.0377 | False |
| ketchup | 10-27 | prefix | 1 | 18/18 (100%) | 0.000 | 0.0379 | False |
| **Total** | | | | **62/72 (86%)** | **0.000** | | |

## Physical Bridge — PASS

| Task | cond qpos_delta | clean qpos_delta | Ratio |
|------|----------------|-----------------|-------|
| cream | 0.0045 | 0.0003 | **15×** |
| cream B | 0.0173 | — | — |
| salad | 0.0372 | 0.0007 | **53×** |
| ketchup s0 | 0.0377 | 0.0007 | **54×** |
| ketchup s1 | 0.0379 | 0.0007 | **54×** |

Ketchup qpos drops from 0.039 to 0.001 across the window — gripper fully opens.

## Task-Level — Mixed

| Task | prefix fail? | random fail? | Denominator |
|------|-------------|-------------|-------------|
| cream | No | No | **Clean** |
| salad | No | **Yes** | Polluted |
| ketchup | **Yes (reproducible)** | **Yes** | Polluted |

Ketchup prefix fails reproducibly (2/2 seeds) with 17-18/18 OPEN and qpos fully open. But random eps8 also fails at the same window, blocking VIS-specific task-level claim.

## Controls

| Task | Window | Cond | Seed | OPEN | qpos_delta | Done |
|------|--------|------|------|------|-----------|------|
| cream | 12-20 | clean | 0 | 0/9 | 0.0003 | True |
| cream | 12-20 | random | 0 | 0/9 | 0.0003 | True |
| salad | 10-25 | clean | 0 | 0/16 | 0.0007 | True |
| salad | 10-25 | random | 0 | 0/16 | 0.0007 | False |
| ketchup | 10-27 | clean | 0 | 0/18 | 0.0007 | True |
| ketchup | 10-27 | random | 0 | 0/18 | 0.0007 | False |

## Claim Boundary

**Allowed**:
- prefix_locked_gripper_open_margin achieves generation-aligned gripper OPEN in rollout across cream, salad, and ketchup with zero arm drift
- prefix_margin induces strong qpos response (15-54× clean baseline)
- ketchup prefix fails reproducibly with gripper fully open

**Not allowed**:
- random-controlled task-level VIS-specific failure is established
- ProprioNoStep-triggered attack
- large-scale LIBERO success rate
- stealth / low-budget attack
