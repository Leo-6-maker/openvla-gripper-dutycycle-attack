# Spatial Post-L3 Backlog

## Status: DEFERRED_UNTIL_AFTER_L3_VIS

## Current Spatial State
- 100/100 LIBERO-Spatial clean episodes sealed
- 10 tasks × 10 states
- Clean success: 76/100
- D5 emit: 99/100
- Qpos/eef validity: 100/100
- 16/16 features aligned across Object and Spatial
- Largest ordinary feature offset: ~0.14σ
- Candidate_index offset: ~1.53σ (expected — episode-length dependent)

## Privileged Label Adapter Requirements

Spatial uses different proprio layout than Object:
- EEF site: `gripper0_grip_site` (Object uses `gripper0_center`)
- Gripper qpos: same `physical_gripper_state()` function
- Different task objects and grasping geometries

Adapter needed:
1. `SpatialD5TeacherPLabelAdapter` — produces ws/anchor/we labels from Spatial episodes
2. Must use the same D5FrozenFeatureAdapter + D5FrozenOnlineDetectorV1
3. Per-step qpos reading via `physical_gripper_state()` (qpos bug already fixed)
4. Label protocol: same First-CLOSE → anchor identification
5. Output: `d5_teacher_p_labels_spatial_v1.csv`

## Teacher-P Label Protocol
- Use D5 emit as First-CLOSE proxy
- Window: [emit, emit+10) for dense analysis
- Anchor: first CLOSE peak within window
- WS: window start, WE: window end
- Same repeatability check as Object (2nd rollout)

## Mixed-Detector Evaluation Design
- D5-v2 proposes to train on Object+Spatial mixed features
- Key question: does candidate_index distribution shift across suites?
- Evaluation: train on Object only, evaluate on Spatial (zero-shot)
- Then: train on Object+Spatial, evaluate held-out tasks
- Success criterion: Spatial emit alignment comparable to Object

## Real-Robot Sensor Adapter Plan
- Proprio reading: use robot SDK, not MuJoCo qpos
- EEF position: forward kinematics from joint states
- Gripper width: sensor reading or motor current-based proxy
- Feature extraction: same 16 features, different physical source
- Camera: same RGB frame → processor pipeline

## Priority Order (After L3 VIS Completes)
1. Spatial Teacher-P label generation (1 day)
2. Spatial D5-v2 mixed training evaluation (2 days)
3. Cross-suite generalization report (1 day)
4. Real-robot sensor adapter spec (draft only)

## Do Not Start Until
- Layer 3 VIS completes H2 (>=2/3 parents)
- Codex releases GPU(1,5) or GPU(2,6) becomes available
- Issue #28 authorizes Spatial phase
