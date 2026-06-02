# 8h Autonomous Night Plan — Final Handoff

**Date**: 2026-05-31 | **Branch**: `exp/8h-crosssuite-vis-night-20260531`

## Executive Summary

Cross-suite validation complete: ProprioNoStep is Object-production only. VIS direction calibration blocked by engineering requirements. Production line unchanged.

## Phase A: Cross-Suite Shadow — COMPLETE (32/32)

| Suite | Episodes | Success | Triggered | Trigger Rate | First Trigger (avg) |
|-------|----------|---------|-----------|-------------|---------------------|
| Object (ref) | 50 | — | 47/50 | 94% | step 102-259 |
| **Spatial** | **20** | 13/20 | **13/20** | **65%** | step 93 |
| **Goal** | **12** | 11/12 | **1/12** | **8%** | step 85 |

### Key Findings

1. **ProprioNoStep transfers partially to Spatial** (65% trigger rate). All Spatial tasks are "pick up black bowl and place on plate" — same mechanism as Object. Task similarity partially bridges the eef_z distribution shift (5.7x).

2. **ProprioNoStep does NOT transfer to Goal** (8% trigger rate). Only `put_bowl_on_plate` (the closest pick_place task) triggers. Articulated/planar tasks are silent.

3. **All cross-suite triggers are contact-phase** (step 85-93, not pre-contact). When the detector fires, it fires at the right time — just not often enough.

4. **Root cause**: Distribution shift in EEF position (eef_z: Object 0.19 → Spatial/Goal 1.08, 5.7x). Schema correct (no NaN, no missing features). Not an engineering bug.

### Cross-Suite Decision

- Cross-suite sus30 (A4): **BLOCKED** — Goal trigger rate too low
- CrossSuite-ProprioNoStep-v2: **Needed** if cross-suite attack is desired
  - Replace raw eef_z with relative_eef_z
  - Multi-suite training with clean teacher labels
  - Suite-aware normalization
- Do NOT train v2 without approval

## Phase B: VIS Direction Calibration — BLOCKED

TokenPrefixPGDAttacker confirmed working in principle (Phase B2 smoke: gripper action changes -0.996, linf-bounded, <0.1s), but standalone calibration blocked by:
1. Config structure (`method` must be inside `attack_optimizer`)
2. Dtype mismatch (Half vs BFloat16 in observation vs model)
3. TokenPrefixPGDAttacker requires the full runner pipeline for dtype/preprocessing management

**Path forward**: VIS calibration must be done within the runner (`v4_run_eval_openvla.py`), not standalone.

## Blocked Items

| Phase | Reason |
|-------|--------|
| B1 (VIS direction) | Dtype mismatch in standalone mode |
| C (arm-preserving) | Dependent on B1 |
| E (forced-window VIS micro) | Gate B1 failed |
| F (detector-triggered VIS) | Gate E not reached |
| A4 (cross-suite sus30) | Goal trigger rate too low (8%) |

## Production Line (Unchanged)

- **Detector**: ProprioNoStep (Object only)
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10
- **Status**: production_ready_for_group_meeting

## GPU Status

All idle. GPU0 quarantined. No fresh Xid.

## Valid Claims

- ProprioNoStep is production for LIBERO-Object.
- Cross-suite transfer is partial (Spatial 65%) to minimal (Goal 8%).
- Distribution shift (eef_z 5.7x) is the root cause, not schema bugs.
- VIS PGD mechanism confirmed at single-step level; runner integration needed.
- Cross-suite sus30 and VIS rollout are blocked.

## Forbidden Claims

- ProprioNoStep universal across LIBERO
- VIS attack successful
- Cross-suite attack ready
- Command-layer sus30 equals VIS

## Next Actions

1. Group meeting: Object selectivity headline
2. CrossSuite-v2: suite-aware normalization plan
3. VIS: runner integration for calibration
