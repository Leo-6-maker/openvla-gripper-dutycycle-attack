# Contact-Aware Visual Detector — Longrun Handoff

**Date**: 2026-05-31 | **Branch**: `exp/contact-aware-visual-reranker-20260530` | **Status**: visual_contact_phase_insufficient

## Executive Summary

**Four independent attempts to make visual features contact-phase selective have all failed.** DINOv2+SigLIP visual features — whether static, delta, or combined with proprio — cannot distinguish contact-phase vulnerability from pre-contact scene appearance. All visual-based models fire at step 4 (pre-contact) on all 50 Full10 episodes.

## Evidence Summary

| # | Attempt | Training | Input | Result |
|---|---------|----------|-------|--------|
| 1 | VisualNoStep V6 | Frozen checkpoint | Static visual | Pre-contact (step 14-63) |
| 2 | VisualNoStep_v2 | teacher_hazard labels | Static visual | Step 4, 100% trigger |
| 3 | Proprio+Visual re-ranker | Post-hoc analysis | Static visual at Proprio windows | Scores decay to zero |
| 4 | Contact-aware delta | Proprio trigger window labels | Visual delta (frame diff) | Step 4, 100% trigger |

## Phase 1: Data Diagnosis

- Visual delta (frame-to-frame cosine change): mean=0.022, std=0.034
- 5.6% of frame pairs have delta > 0.05
- Contact-onset delta peaks: 1.2-7.1σ above mean
- **Signal exists** but is sparse and noisy

## Phase 5: Training Results

| Model | Input | Val AUC | Notes |
|-------|-------|---------|-------|
| VisualDelta | static+delta (4352-dim) | 0.611 | Barely above chance |
| VisualProprioDelta | static+delta+proprio (4365-dim) | 0.921 | Proprio dominates |
| task_only | 10-dim task one-hot | 0.403 | Below chance |
| label_shuffle | Same as visual_delta | 0.600 | Near-chance sanity check |

## Phase 6: Full10 Replay

**All models trigger at step 4 on 50/50 episodes — 100% pre-contact.**

| Model | Pre-contact % | Contact-phase % | Selectivity |
|-------|-------------|-----------------|-------------|
| ProprioNoStep | 8.5% | 91.5% | High 10/10, Robust 10/10 |
| VisualDelta | **100%** | 0% | None |
| VisualProprioDelta | **100%** | 0% | None |
| task_only | **100%** | 0% | None |

## Root Cause

DINOv2+SigLIP features encode **static scene appearance** (what objects are visible, their poses, the background). The "novelty" signal is highest at episode start when the scene is first observed, and decays monotonically. This is true regardless of:

- Training labels (teacher_hazard, Proprio trigger windows, contact-phase)
- Input features (static, frame-delta, combined with proprio)
- Architecture (standalone detector, re-ranker on Proprio windows)

The visual signal simply does not contain a contact-onset signature that generalizes across tasks and episodes.

## What Would Be Needed

To make visual useful for contact-phase detection, fundamentally different features would be required:
1. **Optical flow** — actual motion vectors between frames
2. **Object-centric crops** — gripper+object region rather than full scene
3. **Depth/disparity** — 3D information about gripper-object distance
4. **Tactile/force** — direct contact sensing

None of these are available in the current DINOv2+SigLIP frozen feature pipeline.

## Production Line (Unchanged)

- **Detector**: ProprioNoStep
- **Attack**: sustained_command_open_proxy_30
- **Selectivity**: High 0/10, Robust 10/10
- **Status**: production_ready_for_group_meeting

## Valid Claims

- ProprioNoStep is the production online detector.
- sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive tasks (0/10) while preserving robust controls (10/10).
- DINOv2+SigLIP visual features — whether static, delta, or combined — cannot achieve contact-phase selectivity across four independent attempts.
- Proprioceptive signal naturally encodes gripper-object contact dynamics — this is why ProprioNoStep works.
- Visual information encodes scene/object appearance and task difficulty, but lacks the temporal precision needed for contact-phase triggering.

## Forbidden Claims

- VIS attack successful/failed
- Visual information useless (it encodes task difficulty, just not contact timing)
- Visual production-ready
- Universal attack
- Detector oracle-optimal

## GPU Status

All idle. GPU0 quarantined. No fresh Xid.

## Reports

- `reports/CONTACT_AWARE_VISUAL_LONGRUN_HANDOFF_20260530.md`
- Server: `/data/liuyu/outputs/milestone_2m_contact_aware_visual_20260530/`
- Server: `/data/liuyu/outputs/reports/` (previous handoffs)
