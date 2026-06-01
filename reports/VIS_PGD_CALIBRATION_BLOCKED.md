# VIS PGD Calibration — Blocked

**Date**: 2026-05-31 | **Status**: BLOCKED (Gate B1 fail ×2)

## Attempts

1. v1: `ExistingDenseAttackAdapter` always used (method in wrong config level, fallback triggered)
2. v2: `TokenPrefixPGDAttacker` reached but dtype mismatch (Half vs BFloat16)

## Root Cause

TokenPrefixPGDAttacker works within the `v4_run_eval_openvla.py` runner pipeline where dtype/image preprocessing is managed. Standalone calibration requires:
- Correct config structure (`method` inside `attack_optimizer`)
- `src/` on PYTHONPATH
- dtype consistency (observation Half, model BFloat16)
- KV-cache compat mode

## Path Forward

VIS calibration should be done WITHIN the runner pipeline:
1. Add `attack_condition: "vis_gripper_targeted"` to runner
2. Single-episode VIS smoke with ProprioNoStep trigger windows
3. Measure gripper delta, arm drift, perturbation in full pipeline context

## Blocked Dependencies

- Phase C (arm-preserving sweep): BLOCKED
- Phase E (forced-window VIS micro): BLOCKED
- Phase F (detector-triggered VIS mini): BLOCKED

## Status

VIS single-step gradient smoke (Phase B2) confirmed mechanism works.
VIS runner integration needed before further VIS work.
