# VIS Multi-Phase Paused Status

**Date**: 2026-06-02 15:25  
**Action**: Stopped all multi-phase rollout jobs per Leon's priority update.

## GPU State After Stop

| GPU | Bus ID | Status | User |
|-----|--------|--------|------|
| 0 | 04:00 | FREE (0%, 3 MiB) | — |
| 1 | 06:00 | FREE (0%, 4 MiB) | — |
| 2 | 07:00 | FREE (0%, 4 MiB) | — |
| 3 | 08:00 | FREE (0%, 3 MiB) | (Xid31 history) |
| 4 | 0C:00 | FREE (0%, 4 MiB) | — |
| 5 | 0D:00 | FREE (0%, 3 MiB) | — |
| 6 | 0E:00 | ACTIVE (48%) | gate-lite no-rollout audit |
| 7 | 0F:00 | ACTIVE (52%) | gate-lite no-rollout audit |

## Stopped Jobs

| PID | GPU Pair | Task | Schedule | Elapsed | Partial Trace |
|-----|----------|------|----------|---------|---------------|
| 16008 | 1,0 | ketchup | two_phase_strong_carry_preplace_d20_d20 | 27m30s | No |
| 5279 | 4,5 | ketchup | ultra_three_phase_d20_d20_d20 | 2m59s | Partial (random matched ok) |
| 2576 | — | queue scheduler | hardcase recovery | — | — |
| 41408 | — | queue scheduler | multiphase | — | — |

## Previously Completed (Preserved)

| PID | GPU Pair | Task | Schedule |
|-----|----------|------|----------|
| 48675 | 1,0 | salad_dressing | single_best_phase_d20 |
| 14128 | 2,3 | salad_dressing | two_phase_strong_carry_preplace_d20_d20 |
| 1147 | 4,5 | salad_dressing | three_phase_strong_d16_d16_d16 |

Traces in `/data/liuyu/outputs/vis_failure_first_multiphase_20260602/runs/`.

## Running (Preserved)

| PID | GPU Pair | Job |
|-----|----------|-----|
| 20375 | 6,7 | gate-lite: vis_l1_l2_no_rollout_audit (cream_cheese, salad_dressing, ketchup) |

## Xid Status

- GPU0 (04:00): clean
- GPU3 (08:00): Xid31 at 15:04 (PID 18519, salad_dressing job) — historical
- GPU7 (0F:00): Xid31 at 13:31 (PID 41770, old python job) — historical
- No new Xid during gate-lite run

## Reason

Per Leon: multi-window attacks address phase coverage, but current bottleneck is earlier — VIS does not reliably induce decoded gripper OPEN. Priority shifted to gripper-specific VIS attack objectives.

## New Mainline

- P0-P1: Gripper-specific objectives implemented (commit 0e23f0f)
- P2: Gripper margin scan (pending)
- P3: No-rollout gate with new objectives (pending)
- P4: Rollout only after gate passes
