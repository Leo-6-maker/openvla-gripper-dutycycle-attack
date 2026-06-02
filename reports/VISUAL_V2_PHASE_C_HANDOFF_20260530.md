# Visual V2 Phase C — Final Handoff

**Date**: 2026-05-30 | **Branch**: `exp/visual-v2-reranker-training-20260530` | **Status**: visual_v2_not_ready

## Executive Summary

Visual v2 was trained on Object-100 teacher labels (AUC 0.95) but **still fires at pre-contact (step 4) on all 50 Full10 episodes**. The gap between teacher label prediction and online contact-phase selectivity is fundamental: visual features encode scene appearance, not gripper-object contact dynamics.

## Phase C Decision Gate

| Rule | Condition | Result | Status |
|------|-----------|--------|--------|
| C2.1 | VisualNoStep_v2 still pre-contact? | YES — 100% at step 4 | **visual_v2_not_ready** |
| C2.2 | VisualProprio_v2 contact-phase + coverage ≥ 0.9? | NO — only 2/50 triggers | NOT reranker_candidate |
| C2.3 | VisualNoStep_v2 AUC high, threshold extreme, pre-contact? | YES — AUC 0.95, th=0.01, step 4 | **visual_only_scene_prior** |
| C2.4 | task_only ≈ visual? | NO — task_only AUC=0.34, visual AUC=0.95 | Visual signal is real |
| C2.5 | label_shuffle still elevated? | NO — final AUC=0.24, std=0.20 | Artifact explained |

## What We Learned

### 1. Visual signal encodes scene difficulty, not contact timing
This is now confirmed across three independent pieces of evidence:
- VisualNoStep V6 (frozen checkpoint) → pre-contact trigger (step 14-63)
- VisualNoStep_v2 (freshly trained) → pre-contact trigger (step 4)
- VisualProprioNoStep_v2 (freshly trained) → near-silent at calibrated threshold

### 2. Teacher AUC does not guarantee online selectivity
Teacher labels mark "this step is in a gripper-vulnerable window." The visual model learns "this scene contains a difficult object" from the first frame. From frame 1, the scene looks the same — the object is visible, the basket is visible, the robot is approaching. The visual signal has no natural mechanism to detect when gripper-object contact begins.

### 3. Proprioceptive signal is naturally contact-timed
ProprioNoStep fires across varied phases (grasp 34%, release 49%) and steps (102-259) because proprio features change when the gripper interacts with the object. The signal domain itself encodes timing.

### 4. Visual is not useless — it's a different modality
Visual AUC 0.95 vs task_only 0.34 proves the signal is real. Visual knows which objects are hard. But it doesn't know when they're being touched. Future use: visual as scene-level difficulty prior, re-ranker on proprio-timed windows.

## Claim Boundaries (Updated)

### Valid
- ProprioNoStep is the production online detector — fires at contact/transport/placement phase.
- sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive tasks (0/10) while preserving robust controls (10/10).
- Visual v2 predicts teacher_hazard (AUC 0.95) but fires pre-contact (step 4) on all tasks — it encodes scene difficulty, not contact timing.
- Visual signal is NOT useless — it contains task/object difficulty information. But it is not suitable as a standalone online detector because it lacks contact-phase timing.
- Proprioceptive signal naturally encodes gripper-object contact dynamics — this is why ProprioNoStep works.

### Forbidden
- VIS attack successful / failed
- Visual information is useless
- Visual v2 is production-ready
- Visual v2 replaces ProprioNoStep
- Universal attack
- Detector is oracle-optimal

## Online Pilot Recommendation

**Do NOT launch online pilot for Visual v2.**

Reasons:
1. VisualNoStep_v2 fires pre-contact on 100% of episodes — same failure mode as V6
2. VisualProprioNoStep_v2 is effectively silent at calibrated threshold
3. No model demonstrates contact-phase selectivity exceeding ProprioNoStep
4. Online pilot would waste GPU hours confirming already-known pre-contact behavior

## Future Directions

If visual is to be pursued, three paths are viable:
1. **Visual as re-ranker**: ProprioNoStep provides candidate windows (contact-timed); visual judges vulnerability within those windows. This bypasses the timing problem.
2. **Visual as difficulty prior**: Use visual scores as a per-task difficulty weight, not a per-step trigger.
3. **Temporal visual training**: Train visual on frame differences / optical flow to learn motion cues, not static appearance.

All require further research design before any GPU work.

## Production Line (Unchanged)

- **Detector**: ProprioNoStep
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10
- **Status**: production_ready_for_group_meeting

## Artifacts

| Type | Path |
|------|------|
| Models | `milestone_2k_visual_detector_v2_training_20260530/models/` |
| Replay results | `milestone_2k_visual_detector_v2_training_20260530/tables/full10_replay_results.csv` |
| Training logs | `milestone_2k_visual_detector_v2_training_20260530/logs/` |
