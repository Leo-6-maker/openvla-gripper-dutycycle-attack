# VIS Failure Mechanism Audit

**Date**: 2026-06-02

## Claim Boundary

This audit treats the current VIS adaptive-controller results as **exploratory negative calibration**, not as a solved or validated adaptive controller. It does not establish Object-wide selective VIS, does not validate detector-triggered VIS, and does not use simulator success alone as VIS-specific failure evidence.

ProprioNoStep handles phase/timing better than the current VIS transfer path. VIS still has a physical-transfer gap: decoded OPEN commands do not reliably become physical gripper opening. The main mediator remains qpos_delta / physical gripper response.

`min_hold_qpos_cap` is promising but unvalidated. The completed traces require post-step qpos and denominator repair before any controller assessment; this report therefore uses them only as source-provenance and calibration evidence.

## Core Finding

VIS task failure is not explained by decoded OPEN count alone. The key mediator is physical gripper response (qpos/width). Decoded OPEN commands must translate to actual gripper opening to cause task failure, and that translation is task-dependent.

## Recomputed Adaptive Trace Status

Source tables:

- `tables/codex_recomputed_adaptive_result_summary.csv`
- `tables/codex_recomputed_trace_validity_audit.csv`

All 16 adaptive trace CSVs were present and recomputed from `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/`. However, all 16 have the old schema and are marked `schema_incomplete` because they lack `attack_attempted`, `pgd_applied`, `controller_active`, `controller_stopped`, `effective_attack_step_idx`, `qpos_pre_step`, and `qpos_post_step`.

Three adaptive logs were also found with CUDA illegal-memory crash traces and no completed episode; they are represented as `crashed` rows in the validity audit.

## Task Vulnerability Ranking

These rankings are mechanism hypotheses from the frozen VIS sweep and must be read with the provenance caveat above.

| Task | Fail Rate | Avg Flips | Avg ArmL2 | qpos_delta | Streak | Mechanism |
|------|-----------|-----------|-----------|------------|--------|-----------|
| salad | 71% (5/7) | 0.43 | 0.80 | 0.020-0.024 | 5-7 | High physical transfer; not a robust VIS control |
| cream | 57% (4/7) | 0.58 | 0.87 | 0.004-0.006 | 6 | Cleanest controlled VIS positive |
| ketchup | 33% (3/9) | 0.43 | 0.79 | near 0 | 2-4 | Action-level affected, low physical transfer |
| tomato | 17% (1/6) | 0.42 | 0.35 | near 0 (d60: 0.038) | 4 (d60: 10) | Physically resilient; d60 is over-budget |

## Mechanism by Task

### Salad Dressing

- Even d16 can produce high qpos_delta and long OPEN streaks.
- VIS OPEN commands more readily translate to physical gripper opening.
- Salad is not a stable robust control for VIS.
- Low-flip adaptive failures still require CQ/manual audit before being treated as VIS-specific.

### Cream Cheese

- Requires sustained OPEN density for failure.
- Moderate physical transfer: enough flips at enough density can open the gripper and drop the object.
- Cream remains the cleanest controlled VIS positive in this evidence set.

### Ketchup

- Decoded OPEN flips can be present while qpos_delta remains near zero.
- Ketchup tolerance is better explained by low physical transfer than by no action-level effect.
- Ketchup is not proof that VIS is selective or solved.

### Tomato Sauce

- qpos_delta remains near zero through d40 despite decoded flips.
- d60 can induce failure but is three times the selective budget and should not be claimed as selective VIS.

## Source Trace Provenance

| Task | Controller | K | Q | Timestamp | Success | Attacks | Flips Attacked | qpos_delta_pre | qpos_delta_post | Validity | Trace |
|------|------------|---|---|-----------|---------|---------|----------------|----------------|-----------------|----------|-------|
| cream_cheese | open_streak_stop | 5 | 0 | 042831 | True | 11 | 7 | 0.000936 | unavailable | schema_incomplete | `vis_cream_cheese_s0_vis_pgd_full_d11_w65_75_seed0_open_streak_stop_K5_Q0_md16_042831_trace.csv` |
| cream_cheese | open_streak_stop | 2 | 0 | 110852 | True | 2 | 2 | 0.000049 | unavailable | schema_incomplete | `vis_cream_cheese_s0_vis_pgd_full_d16_w65_80_seed0_open_streak_stop_K2_Q0_md16_110852_trace.csv` |
| cream_cheese | open_streak_stop | 3 | 0 | 112346 | True | 13 | 4 | 0.000515 | unavailable | schema_incomplete | `vis_cream_cheese_s0_vis_pgd_full_d16_w65_80_seed0_open_streak_stop_K3_Q0_md16_112346_trace.csv` |
| cream_cheese | open_streak_stop | 3 | 0 | 120533 | True | 16 | 7 | 0.001094 | unavailable | schema_incomplete | `vis_cream_cheese_s0_vis_pgd_full_d16_w65_80_seed0_open_streak_stop_K3_Q0_md16_120533_trace.csv` |
| cream_cheese | min_hold_qpos_cap | 0 | 0.008 | 114917 | False | 20 | 11 | 0.003464 | unavailable | schema_incomplete | `vis_cream_cheese_s0_vis_pgd_full_d20_w65_84_seed0_min_hold_qpos_cap_K0_Q0.008_md20_114917_trace.csv` |
| ketchup | open_streak_stop | 5 | 0 | 021314 | False | 5 | 5 | 0.003963 | unavailable | schema_incomplete | `vis_ketchup_s0_vis_pgd_full_d11_w93_103_seed0_open_streak_stop_K5_Q0_md16_021314_trace.csv` |
| ketchup | open_streak_stop | 5 | 0 | 110921 | True | 11 | 1 | 0.000063 | unavailable | schema_incomplete | `vis_ketchup_s0_vis_pgd_full_d11_w93_103_seed0_open_streak_stop_K5_Q0_md16_110921_trace.csv` |
| ketchup | open_streak_stop | 3 | 0 | 112613 | True | 16 | 3 | 0.000100 | unavailable | schema_incomplete | `vis_ketchup_s0_vis_pgd_full_d16_w93_108_seed0_open_streak_stop_K3_Q0_md16_112613_trace.csv` |
| ketchup | open_streak_stop | 3 | 0 | 120646 | False | 14 | 8 | 0.012237 | unavailable | schema_incomplete | `vis_ketchup_s0_vis_pgd_full_d16_w93_108_seed0_open_streak_stop_K3_Q0_md16_120646_trace.csv` |
| ketchup | min_hold_qpos_cap | 0 | 0.012 | 114638 | True | 20 | 8 | 0.000187 | unavailable | schema_incomplete | `vis_ketchup_s0_vis_pgd_full_d20_w93_112_seed0_min_hold_qpos_cap_K0_Q0.012_md20_114638_trace.csv` |
| salad_dressing | min_hold_qpos_cap | 0 | 0.008 | 114506 | False | 14 | 1 | 0.010007 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed0_min_hold_qpos_cap_K0_Q0.008_md20_114506_trace.csv` |
| salad_dressing | open_streak_stop | 2 | 0 | 112013 | False | 16 | 4 | 0.003670 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed0_open_streak_stop_K2_Q0_md16_112013_trace.csv` |
| salad_dressing | open_streak_stop | 3 | 0 | 120729 | False | 15 | 6 | 0.000177 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed0_open_streak_stop_K3_Q0_md16_120729_trace.csv` |
| salad_dressing | open_streak_stop | 5 | 0 | 021650 | True | 16 | 5 | 0.000531 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed0_open_streak_stop_K5_Q0_md16_021650_trace.csv` |
| salad_dressing | open_streak_stop | 5 | 0 | 043129 | True | 16 | 7 | 0.000204 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed0_open_streak_stop_K5_Q0_md16_043129_trace.csv` |
| salad_dressing | open_streak_stop | 2 | 0 | 120029 | True | 8 | 4 | 0.000030 | unavailable | schema_incomplete | `vis_salad_dressing_s0_vis_pgd_full_d21_w88_108_seed1_open_streak_stop_K2_Q0_md16_120029_trace.csv` |

## Implications

1. VIS payload is a gripper-channel instantiator, not a universal attack.
2. Physical transfer is task-dependent: object geometry, grasp contact, friction, and placement tolerance determine whether OPEN commands become qpos opening and then task failure.
3. Current adaptive-controller traces are exploratory negative calibration. They require post-step qpos and denominator repair before controller claims.
4. VisualTransferHead is the next mainline direction, but training should not start without explicit approval.
