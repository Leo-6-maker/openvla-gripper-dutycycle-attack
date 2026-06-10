# Stage-B RC1a S8 Extended VIS/RAND Milk Smoke Result

**Date**: 2026-06-10
**Smoke commit**: 501c5dc (runner v2), 74c0d87 (final pre-launch)
**Phase 1 ORACLE ref**: b640f9b
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

**Verdict**: Command gate PASS, Physical bridge gate FAIL. Full 24-job expansion NOT approved.

Milk_s0_w70_80 is the cleanest physical window (clean baseline pos_area=0, ORACLE L=10 pos_area=+0.261). Under current VIS attack (eps6, pgd20, prefix_locked_gripper_open_margin), the model produces many OPEN commands but does not generate positive qpos opening. The bottleneck is sustained physical-effective gripper duty cycle, not command-level vulnerability.

## Infrastructure

| Gate | Result |
|------|--------|
| 8/8 summary JSON | PASS |
| 8/8 trace CSV | PASS |
| 4/4 logical pairs | PASS |
| 0 FAILs / CUDA / EGL | PASS |
| GPU released | PASS |
| Trace OK 4/4 | PASS |

## Command-Level Results

| Length | Condition | Seed | Open Count | Max Streak |
|--------|-----------|------|------------|------------|
| short (10 steps) | VIS | 9 | **9** | 7 |
| short | VIS | 10 | **8** | 8 |
| short | RAND | 9 | 0 | 0 |
| short | RAND | 10 | 1 | 1 |
| extended20 (30 steps) | VIS | 9 | **15** | 4 |
| extended20 | VIS | 10 | **19** | 6 |
| extended20 | RAND | 9 | 5 | 3 |
| extended20 | RAND | 10 | 11 | 9 |

**Command gate PASS**: VIS open_count >> RAND open_count. VIS-specific command OPEN confirmed.

## Physical Results

| Length | Condition | pos_area | neg_area | abs_area |
|--------|-----------|----------|----------|----------|
| short | VIS atk=9 | **0** | 0.023 | 0.023 |
| short | VIS atk=10 | **0** | 0.029 | 0.029 |
| short | RAND atk=9 | **0** | 0.185 | 0.185 |
| short | RAND atk=10 | **0** | 0.169 | 0.169 |
| ext20 | VIS atk=9 | **0** | 0.069 | 0.069 |
| ext20 | VIS atk=10 | **0** | 0.022 | 0.022 |
| ext20 | RAND atk=9 | **0** | 0.161 | 0.161 |
| ext20 | RAND atk=10 | **0** | 0.018 | 0.018 |

**Physical bridge gate FAIL**: qpos_pos_area = 0 for all 8 jobs. All qpos movement is negative.

**ORACLE-normalized gate FAIL**: VIS pos_area = 0 / ORACLE_L10 pos_area (0.261) = 0.

## Comparison to Phase 1 ORACLE

| Condition | pos_area | neg_area | Interpretation |
|-----------|----------|----------|----------------|
| CLEAN (Phase 1) | 0.000 | 0.028 | Natural negative drift |
| ORACLE L=10 (Phase 1) | **+0.261** | 0.005 | Sustained OPEN overcomes drift |
| VIS (this smoke) | **0.000** | 0.02-0.07 | Intermittent OPEN does not overcome drift |
| RAND (this smoke) | **0.000** | 0.02-0.18 | Random perturbation follows natural drift |

## Interpretation

ORACLE_OPEN_ONLY (Phase 1) establishes that milk_s0_w70_80 is physically reachable: continuous forced OPEN at L=10 produces immediate positive qpos opening in the correct direction.

However, VIS-induced OPEN commands under the current attack settings are not physically effective: although VIS generates many OPEN actions (8-9/10 for short, 15-19/30 for extended20), the resulting qpos trajectory follows the natural negative drift rather than the ORACLE positive opening direction.

The bottleneck is **sustained physical-effective gripper duty cycle**, not command-level OPEN generation. VIS produces intermittent OPEN actions interspersed with CLOSE actions; ORACLE produces continuous OPEN. The intermittent pattern fails to overcome milk's natural negative qpos drift at this window.

## Gates Summary

| Gate | Result |
|------|--------|
| Infrastructure (8/8, 0 FAIL) | PASS |
| Command (VIS open > RAND open) | PASS |
| Physical bridge (VIS pos_area > 0) | FAIL |
| ORACLE-normalized (VIS >= 0.3 oracle) | FAIL |
| Full 24-job expansion | NOT APPROVED |

## Claim Boundary

**Allowed**:
- Current VIS attack produces command-level OPEN on milk but not positive qpos opening
- Physical bridge failed under eps6/pgd20/prefix_locked_gripper_open_margin/short+extended20
- Bottleneck is sustained physical-effective gripper duty cycle
- S6 CleanRand abstain and S8 ORACLE upper-bound remain valid

**Forbidden**:
- VIS attack cannot ever cause physical opening
- Layer-3 physical bridge is impossible
- Layer-1 command-level result is invalidated
- Expand to full 24-job queue without addressing sustained-open duty cycle

## Next Step

Conduct trace-level duty-cycle diagnosis on milk before attempting any attack modification:
1. Measure max_consecutive_open vs ORACLE's 10 consecutive steps
2. Count CLOSE interruptions during the attack window
3. Analyze qpos response during attack window (not just post-window)
4. Design sustained-open objective candidate based on findings
