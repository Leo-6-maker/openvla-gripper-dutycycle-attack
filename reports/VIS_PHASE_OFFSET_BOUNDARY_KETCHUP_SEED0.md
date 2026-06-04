# VIS Phase-Offset Boundary — Ketchup Seed0

**Date**: 2026-06-04
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Task**: ketchup (pick_up_the_ketchup_and_place_it_in_the_basket)
**Seed**: 0
**Objective**: `prefix_locked_gripper_open_margin`
**eps_raw_pixels**: 6
**PGD**: 40 steps, 3 restarts
**Window length**: 18 steps

---

## Abstract

VIS prefix_margin attack generates 18/18 OPEN across all tested windows.
Physical qpos opening begins at T+5 and persists through T+40.
**But task failure only occurs within a critical early-grasp vulnerability band: T+5 to T+25.**
Beyond T+30, the same generated OPEN and strong qpos opening do NOT cause task failure.

Generated OPEN is necessary but not sufficient.
Physical qpos opening is also necessary but not sufficient.
The attack must land inside the task-critical early-grasp coupling band.

---

## Offset-Margin Ablation Table

| Policy | Window | VIS OPEN | qpos_delta | armL2 | done | Steps | Clean OPEN | Random OPEN | Physical | Task | Claim |
|--------|--------|----------|-----------|-------|------|-------|------------|-------------|----------|------|-------|
| T+0 | [0,17] | 18/18 | 0.01925 | 0.000 | False | — | 0/18 | 0/18 | weak | fail | No |
| T+5 | [5,22] | 18/18 | 0.03458 | 0.000 | False | — | 0/18 | 0/18 | **strong** | **fail** | **Yes** |
| T+10 | [10,27] | 18/18 | 0.03756 | 0.000 | False | — | 0/18 | 0/18 | **strong** | **fail** | **Yes** |
| T+15 | [15,32] | 18/18 | 0.03804 | 0.000 | False | — | 0/18 | 0/18 | **strong** | **fail** | **Yes** |
| T+20 | [20,37] | 18/18 | 0.03813 | 0.000 | False | — | 0/18 | 0/18 | **strong** | **fail** | **Yes** |
| T+25 | [25,42] | 18/18 | 0.03811 | 0.000 | **False** | 299 | 0/18 | 0/18 | **strong** | **fail** | **Yes** |
| T+30 | [30,47] | 18/18 | 0.03813 | 0.000 | **True** | 158 | 0/18 | 0/18 | strong | success | No |
| T+35 | [35,52] | 18/18 | 0.03816 | 0.000 | **True** | 177 | 0/18 | 0/18 | strong | success | No |
| T+40 | [40,57] | 18/18 | 0.03824 | 0.006 | **True** | 203 | 0/18 | 0/18 | strong | success | No |
| Late W20 | [73,90] | 18/18 | 0.00014 | 0.000 | True | — | 0/18 | 0/18 | none | success | No |
| Late W10 | [83,100] | 18/18 | ~0.000 | 0.000 | True | — | 0/18 | 0/18 | none | success | No |
| Late W0 | [93,110] | 18/18 | ~0.000 | 0.000 | True | — | 0/18 | 0/18 | none | success | No |

Physical thresholds:
- strong_physical: qpos_opening_delta >= 0.03
- weak_physical: 0.01 <= qpos_opening_delta < 0.03
- physical_negative: qpos_opening_delta < 0.01

All claim_usable=True rows have:
- action bridge positive (18/18 OPEN)
- physical bridge strong (qpos_delta >= 0.03)
- task failure positive (done=False / timeout)
- denominator clean (clean OPEN=0, random OPEN=0, random done=True)
- no infrastructure contamination

---

## Key Findings

### A. Action Bridge

VIS prefix_margin generates 18/18 OPEN across ALL tested windows from T+0 to T+40,
and also on late ProprioNoStep windows. The action bridge is robust.

### B. Physical Qpos Bridge

Strong physical qpos opening (>= 0.03) begins at T+5 and persists through T+40.
T+0 shows weak physical opening (0.019). Late ProprioNoStep windows show no qpos opening.

**Qpos opening alone is NOT sufficient for task failure.**

### C. Task-Critical Vulnerability Band

Task failure occurs from T+5 through T+25.
At T+30, the task succeeds despite 18/18 OPEN and strong qpos opening (0.038).
The physical mechanism is still active — the gripper opens physically —
but the task has entered a robust phase where this no longer causes failure.

**Right boundary of task failure: between T+25 [25,42] and T+30 [30,47].**

### D. Late ProprioNoStep Windows

At [73,90] / [83,100] / [93,110], VIS generates 18/18 OPEN but qpos does not open.
The action bridge exists but the physical bridge is absent.
These are late-release-phase windows, not early-grasp coupling windows.

---

## Refined Mechanism Statement

1. **Generated OPEN is necessary but not sufficient** — action bridge is always present.
2. **Strong qpos opening is also necessary but not sufficient** — physical bridge persists through T+40, but task succeeds from T+30 onward.
3. **Attack must land in task-critical early-grasp coupling band: T+5 to T+25 for ketchup seed0.**
4. **Beyond T+30, grasp is stable; OPEN cannot disrupt it.**
5. **Late ProprioNoStep windows have no physical bridge.**

Failure mechanism: VIS induces OPEN at attack window → qpos opens → if window falls in early-grasp coupling band, stable grasp never forms → task timeout at 299 steps.

---

## Infrastructure Notes

T+25 required 4 attempts due to GPU hardware faults:
- GPU0 Xid13 (SM Warp Exception) × 2
- GPU3 Xid31 (MMU Fault) × 1
- Final successful run: GPU4,5

GPU health status:
- GPU0: Xid13 history — do not use for VIS/PGD
- GPU1: healthy
- GPU2: healthy
- GPU3: Xid31 history — monitor before long PGD
- GPU4,5,6,7: healthy

All failed attempts are excluded from results. Only the successful GPU4,5 run is reported.

---

## Claim Boundaries

### Allowed Claims

- VIS can force 18/18 generated OPEN across a broad offset range
- Strong physical qpos opening band: T+5 through T+40
- Task-critical early-grasp vulnerability band: T+5 through T+25
- Right boundary of task failure: between T+25 and T+30
- Generated OPEN and qpos opening are necessary but not sufficient
- Late ProprioNoStep windows are action-positive but physical-negative
- Phase/offset determines whether generated OPEN translates to task failure

### Forbidden Claims

- Window independence — the effect is strongly offset-dependent
- Any strong qpos opening causes failure — T+30/35/40 disprove this
- Physical bridge ends at T+25 — it persists through T+40
- LIBERO-wide generalization — single task, single seed
- Online detector solved
- Detector-driven VIS ready

---

## Next Experiments (Priority Order)

1. **Object oracle delay=-50 VIS smoke**: Validate pre-grasp CLOSED candidate band with 4–6 episodes
2. **Seed robustness**: ketchup seed1/2 T+10 CHAIN for reproducibility
3. **Anticipatory detector**: Train detector that predicts T_gform with 40–50 step lead time
4. **Detector-driven VIS**: Only after anticipatory detector proposal audit passes

---

## Data Provenance

- **Model**: OpenVLA-7B fine-tuned on LIBERO
- **Simulator**: LIBERO with MuJoCo, official preprocessing
- **Trace format**: v3 runner with `attack_invalid` metadata
- **Semantics**: `raw_gripper < 0.5 = OPEN` (canonical)
- **Prefix loss**: `prefix_locked_gripper_open_margin` with arm CE preserved
- **Output dir**: `/data/liuyu/outputs/milestone_7_vis_controlled_rollout_micro_20260601/runs/`
- **Right-boundary logs**: `/data/liuyu/outputs/right_boundary_vis_20260604/`
