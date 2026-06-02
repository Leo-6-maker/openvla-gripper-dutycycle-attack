# Milestone 2F — Object Detector Pilot Readiness

**Date**: 2026-05-29
**Status**: READY (blocked only by GPU availability)

## Summary

Object-100 detector pipeline is complete through replay validation, shadow validation, and attack injection plumbing. The patched artifact-rich runner supports detector-triggered attack injection for clean, oracle_open, random_control, and VIS_targeted conditions. GPU0,7 engineering smoke confirmed runner functionality but also confirmed GPU0,7 hardware unreliability (33% CUDA error rate). Formal 40-rollout attack pilot is blocked pending a clean dual-GPU pair (GPU2,6 after Goal-100 completes).

## Completed

### Data Pipeline
- [x] Object-100: 100 episodes, SR 81/100
- [x] Privileged state coverage: 100%
- [x] Teacher detector: 81/81 clean-success windows, 0/19 failed false positives
- [x] Labeled export: 18,875 rows, 0 leakage
- [x] Visual features: 18,875 images, 2176-dim (DINOv2+SigLIP fused)

### Parser v2
- [x] 40/40 LIBERO tasks correctly classified
- [x] Object-100 v1=v2 regression: 0 lost, 0 gained
- [x] Tests: 25/25 passed
- [x] Mechanism types: pick_place_transfer, multi_object_transfer, articulated_object, planar_rearrangement

### Student Detector
- [x] ProprioNoStep TCN: 38,602 params, 13 causal inputs
- [x] Coverage: 99.1% (offline replay)
- [x] AUROC: 0.969
- [x] Miss: 0
- [x] Failed-episode FP: 0
- [x] No normalized_step, no privileged state, no future timesteps

### Shadow Validation
- [x] Coverage: 97.8%
- [x] All gates passed
- [x] Cream_cheese + milk, 10 episodes

### Attack Injection Infrastructure
- [x] TCN trigger in `triggers.py` + `make_trigger`
- [x] Patched artifact-rich runner with detector + attack injection
- [x] Detector fields logged to step_records
- [x] 4 conditions: clean, oracle_open, random_control, VIS_targeted
- [x] Same detector trigger for all conditions

### GPU Provenance
- [x] GPU0: quarantined (Xid13, CUDA illegal memory access)
- [x] GPU7: quarantined for attack (OOM + Xid31 history)
- [x] GPU3: flagged (Xid31 May 28 historical)
- [x] GPU0,7 rehearsal: 33% CUDA error rate — excludes from official pilot

## Blocked

| Item | Blocker | Resolution |
|------|---------|------------|
| Formal 40-rollout attack pilot | GPU pair availability | Wait for GPU2,6 (Goal-100: 84/100, ~30min ETA) |
| VIS_targeted on GPU0,7 | GPU memory (2-3 GB free per GPU) | Use GPU2,6 for VIS |
| Universal cross-suite detector | Goal/L10 labels not refreshed | Postprocess after workers complete |

## Tests

| Test Suite | Tests | Result |
|------------|-------|--------|
| Parser v2 mechanism classification | 25 | All passed |
| Teacher privileged no student leakage | 10 | All passed |
| normalized_step deployment audit | 7 | All passed |

## Next Actions

1. Goal-100 completes → GPU2,6 free
2. GPU2,6 preflight (Xid, memory, process check)
3. Launch formal 40-rollout attack pilot
4. Aggregate + CQ evaluate
5. Manual audit subset
6. Oracle/VIS/Random gate analysis

## Boundary

No attacks included in this milestone. No VIS/random/oracle outcomes used for detector training. Privileged state is teacher-label-only. Student features are deployment-safe and causal.
