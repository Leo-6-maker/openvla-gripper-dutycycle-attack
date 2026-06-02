# Night Plan — Updated After Cross-Suite Transfer Results

**Date**: 2026-05-31 | **Branch**: `exp/crosssuite-proprio-vis-night-20260531`

## Track A — Cross-Suite ProprioNoStep Validation

### A2: Clean Shadow Results (29+ / 32 episodes)

| Suite | Episodes | Success | Triggered | Trigger Rate | First Trigger (avg) | Max HS (range) |
|-------|----------|---------|-----------|-------------|---------------------|-----------------|
| Object (ref) | 50 | — | 47/50 | 94% | 102-259 | 0.92-0.98 |
| **Spatial** | 17+ | 12/17 | **11/17** | **65%** | **step 89** | 0.05-0.90 |
| **Goal** | 12 | 11/12 | **1/12** | **8%** | step 85 | 0.00-0.69 |

### A2.5: Schema Audit — Root Cause

**Root cause**: Distribution shift in EEF position (eef_z: Object 0.19 → Spatial/Goal 1.08, 5.7x). Not an engineering bug — schema is correct (no NaN, no missing keys, no zero flags).

**Spatial bridges the shift partially**: All Spatial tasks are "pick up black bowl and place on plate" — same `pick_place_transfer` mechanism as Object. Task similarity partially overcomes the distribution shift.

**Goal can't bridge**: Only 1/12 tasks triggered (`put_bowl_on_plate`, the closest pick_place task to Object). Articulated and planar tasks are completely silent.

### A3: Cross-Suite v2 Decision

- Object-ProprioNoStep: **frozen production baseline** — keep as-is
- CrossSuite-ProprioNoStep-v2: **required** if cross-suite attack is desired
- v2 needs: multi-suite training data, suite-aware normalization
- Do NOT train v2 before approval

### A4: Cross-suite sus30 — BLOCKED

Cannot launch cross-suite sus30 because:
- Goal: only 1 task triggers → sus30 would be equivalent to clean
- Spatial: 65% trigger rate → unreliable for selective attack
- ProprioNoStep is Object-production only

## Track B — VIS

### B1: Code Audit

- `TokenPrefixPGDAttacker` (white-box PGD): IMPLEMENTED in `src/gripper_attack/attack_adapter.py`
- `OpenVLAVisualAttacker` (factory): IMPLEMENTED
- Integration: EXISTS in `v4_run_eval_openvla.py`, NOT in production runner
- Threat model: White-box PGD on action-token loss, gripper-targeted, linf-bounded

### B2: Gradient Smoke

**VIS PGD mechanism confirmed working:**
- Model loads on GPU4,5 pair (distributed, ~9GB each)
- PGD produces linf-bounded perturbations (2/255, 4/255)
- Gripper action changes measurably (-0.996 delta from 0.996 → 0.0)
- Arm L2 drift: 0.42-1.31 (depends on frame)
- Runtime: <0.1s per full attack (5-10 steps)
- Direction needs calibration (targeting +1.0, getting 0.0)

### B3/B4: VIS rollout — BLOCKED

Not launching until:
- VIS target direction calibrated
- Runner integration completed (`attack_condition: vis_gripper_targeted`)
- B2 smoke shows correct gripper direction

## Production Line (Unchanged)

- **Detector**: ProprioNoStep (Object only)
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10
- **Status**: production_ready_for_group_meeting

## Key Claims

### Valid
- ProprioNoStep is production for LIBERO-Object.
- ProprioNoStep partially transfers to Spatial (65% trigger, contact-phase timing).
- ProprioNoStep does NOT transfer to Goal (8%, distribution shift dominant).
- VIS PGD mechanism confirmed working — gripper action changes, linf-bounded, fast.
- Cross-suite sus30 and VIS rollout are blocked pending further work.

### Forbidden
- ProprioNoStep universal across LIBERO
- VIS attack successful (mechanism confirmed, selectivity not tested)
- Command-layer sus30 equals VIS
- Universal attack

## GPU Status

All GPUs idle after remaining Spatial rollout completes.
GPU0: quarantined.

## Next Steps (Recommended)

1. Group meeting: Object selectivity is the headline
2. Cross-suite v2: suite-aware normalizer + multi-suite training
3. VIS: calibrate target direction + integrate into runner
4. Both require dedicated branches with Leon approval
