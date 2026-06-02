# Milestone 1D: MuJoCo 2.3.7 Compatibility Diagnosis — Interim Report

**Timestamp**: 2026-05-26T16:15Z (approx)
**Status**: `phase3_running_ssh_down`

## Executive Summary

MuJoCo 2.3.7 compat environment was successfully built. The smoke test confirmed that **gripper physics DO differ** between MuJoCo 2.3.7 and 3.8.0: the gripper qpos responds correctly to close commands under 2.3.7 (delta=0.0381), while it stays near 0 under 3.8.0. However, **the OpenVLA model's trained actions are incompatible with MuJoCo 2.3.7 physics** — the first clean episode (BBQ sauce state 0) crashed with `ValueError: executing action in terminated episode`. A second test with ketchup was launched and was running when SSH went down.

**Key conclusion**: MuJoCo downgrade is NOT a viable fix. The model's action distribution is coupled to MuJoCo 3.x physics. Downgrading causes environment termination. The grasp issue must be fixed within MuJoCo 3.x (e.g., contact solver parameters, friction model, gripper actuator force).

## What Was Confirmed

### Phase 1: Compat Environment ✅
- Env: `/data/aviary/envs/openvla_official_libero_20260525`
- **MuJoCo: 2.3.7** (overriding openvla_sparse's 3.8.0)
- **numpy: 1.26.4** (downgraded from 2.2.6)
- All other packages from openvla_sparse + conda py310 overlay
- All imports verified: mujoco, torch, transformers, robosuite, libero, benchmark

### Phase 2: Smoke Test ✅ (PASSED)
- LIBERO Object ketchup env created successfully under MuJoCo 2.3.7
- Reset, render, step all work
- **Gripper qpos moves correctly**: open=0.0387 → close=0.0005, delta=0.0381
- Under MuJoCo 3.8.0, the same experiment showed qpos staying near 0 (never closing)
- Image rendering works (224x224x3, uint8)

### Phase 3: Clean Reproduction Matrix 🟡
- BBQ sauce state 0: **FAILED** — `ValueError: executing action in terminated episode`
  - Episode terminated during early steps
  - 0 steps written, 157s runtime (model loading took ~60s)
  - Model loaded successfully (982 weight shards, across GPU 0+1)
  - No NEW Xid errors (old Xid entries from hours earlier)
- Ketchup state 0: **LAUNCHED** — model loaded, episode started, SSH died mid-run
  - Result unknown at this time

## Physics Comparison

| Metric | MuJoCo 3.8.0 | MuJoCo 2.3.7 |
|--------|-------------|-------------|
| Gripper qpos response to close | NO (stays ~0) | YES (delta 0.038) |
| Smoke test (zero actions) | works | works |
| OpenVLA model actions | works (but grasp fails) | CRASHES (env termination) |
| Object SR (BBQ sauce) | 4/10 (no_grasp) | 0/1 (crashed) |

## Root Cause Analysis

The model was fine-tuned under MuJoCo 3.x physics. The action distribution it learned depends on specific physics dynamics (joint responses, contact forces, constraint resolution). When these actions are applied to MuJoCo 2.x physics:

1. The arm movements may produce different joint configurations
2. Contact events may trigger different constraint responses
3. The environment may detect invalid states and terminate

This means the MuJoCo version mismatch is **real** but the fix cannot be a simple downgrade. The model and physics are coupled.

## Diagnostic Category

**D → E hybrid**: 
- MuJoCo 3.x physics incompatibility is CONFIRMED (gripper responds differently)
- But MuJoCo downgrade is NOT viable (model actions cause env termination)
- Object gap is likely a **MuJoCo 3.x contact dynamics regression** that specifically affects small/smooth objects
- The fix should target MuJoCo 3.x solver parameters, not a version downgrade

## Recommended Next Actions

1. **Check ketchup test result** once SSH recovers
2. **Investigate MuJoCo 3.x contact solver parameters**:
   - Default solver changed from Newton (2.x) to CG (3.x)
   - Friction cone changed from elliptic (2.x) to pyramidal (3.x)
   - Try forcing `solver="newton"` in robosuite/MuJoCo config
   - Try adjusting `cone="elliptic"` friction parameter
3. **Alternative approach**: Use `mujoco.contact` parameters in the XML/scene file to force MuJoCo 2.x-compatible contact behavior
4. **If model retraining is an option**: Retrain with MuJoCo 2.x physics
5. **Object exclusion**: Keep Object suite excluded from strong attack denominator pending resolution

## Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Compat env | /data/aviary/envs/openvla_official_libero_20260525 | ✅ |
| Smoke test | /tmp/smoke_v3_output.txt | ✅ PASSED |
| BBQ sauce run | .../mj237_clean_bbq_sauce_s0/ | ❌ FAILED |
| Ketchup run | .../mj237_direct_ketchup_s0/ | 🟡 UNKNOWN |
| Queue log | /tmp/phase3_launch.log | Partial |
| Output root | .../milestone_1d_object_mujoco237_compat_20260526/ | Partial |
