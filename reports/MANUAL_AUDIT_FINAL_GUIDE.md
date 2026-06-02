# Manual Audit Guide — ProprioNoStep + VisualNoStep V6

**Date**: 2026-05-30 | **Data sources**: `milestone_2h_sustained_proxy_full10x5_sus30_20260530` + `milestone_2j_visual_fusion_online_pilot_v6_20260530`

## Audit Question 1: Are high-sensitive failures real?

**Answer: YES.** Under ProprioNoStep, cream_cheese and tomato_sauce all fail at max_steps (290) with sustained open-gripper proxy.

### cream_cheese (5/5 fail)
- Attack onset: step 123-161
- Attack duration: 129-167 steps (until max_steps)
- Final gripper: ~0.020 (partially forced open)
- Final tgt_dist: 0.17-0.24 (object never reaches basket)
- max_hs: 0.94-0.98
- **Failure mode**: Sustained open grip after grasp initiation prevents stable transport and placement.

### tomato_sauce (5/5 fail)
- Attack onset: step 118-196
- Attack duration: 94-164 steps
- Final gripper: ~0.028-0.035
- Final tgt_dist: 0.16-0.22
- max_hs: 0.91-0.98
- **Failure mode**: Similar — open grip prevents precise placement.

## Audit Question 2: Are robust controls really stable?

**Answer: YES.** ketchup and salad_dressing complete the task despite sustained attack.

### ketchup (5/5 succeed despite attack)
- Attack onset: step 120-161
- Attack duration: 23-71 steps
- **Episode ends EARLY at step 143-202 (NOT max_steps 290)**
- Final gripper: 0.011-0.020
- Final tgt_dist: 0.145-0.152 (at target!)
- max_hs: 0.93-0.95
- **Survival mode**: Ketchup can be placed roughly; basket tolerance is higher. Task completes before sustained proxy accumulates enough disruption.

### salad_dressing (5/5 succeed despite attack)
- Attack onset: step 88-108
- Attack duration: 27-160 steps
- Final gripper: 0.013-0.020
- Final tgt_dist: 0.142-0.153
- max_hs: 0.92-0.96
- **Notable**: s0 had 160 attack steps (out of 260 total) and STILL succeeded — extremely robust task dynamics.

## Audit Question 3: Why does VisualNoStep V6 break ketchup?

**Answer: EARLY TRIGGER TIMING.** VisualNoStep triggers at step 14-63, vs ProprioNoStep at step 120-161. This ~100-step difference changes the failure mode from "after-grasp interference" to "pre-grasp prevention."

### ketchup under VisualNoStep V6 (0/3 succeed)

| Metric | ProprioNoStep | VisualNoStep V6 |
|--------|--------------|-----------------|
| First trigger | step 120-161 | step 14-63 |
| Attack steps | 23-71 | 187-253 |
| Episode length | 143-202 (ends early) | 290 (max_steps) |
| Final gripper | 0.011-0.020 | 0.0005-0.0006 |
| Final tgt_dist | 0.145-0.152 (at target) | 0.16-0.53 (far away) |

### Root cause analysis

1. **ProprioNoStep fires at contact/grasp phase**: The 13-dim proprio input (gripper position, EEF velocity, action commands) naturally encodes physical interaction. The detector learns to fire when gripper-object contact dynamics suggest vulnerability.

2. **VisualNoStep fires on appearance changes**: Visual features (2176-dim DINOv2+SigLIP) encode scene appearance. The detector fires when the scene "looks like" a vulnerable state, which happens much earlier (object approach, not object contact).

3. **Timing determines failure mode**: 
   - Late trigger (step 120+): Gripper has already grasped object → attack interferes with transport/placement
   - Early trigger (step 14-63): Gripper hasn't grasped yet → attack prevents grasp entirely → complete task failure

4. **Clean episodes confirm non-selectivity**: ketchup clean triggers 35-96 steps (VisualNoStep) — the visual detector fires on clean rollouts too, just without attack consequence.

### ProprioNoStep clean trigger comparison

From detector-clean prep data:
- ProprioNoStep clean ketchup: minimal triggers
- VisualNoStep clean ketchup: 35-96 triggers (non-selective)

## Priority Audit Table

| Priority | Task | Detector | Condition | Episodes | Key Question |
|----------|------|----------|-----------|----------|-------------|
| P0 | cream_cheese | ProprioNoStep | sus30 | 5 fail | Contact collapse or transport failure? |
| P0 | tomato_sauce | ProprioNoStep | sus30 | 5 fail | Precision placement disrupted? |
| P0 | ketchup | ProprioNoStep | sus30 | 5 success | Why does rough placement survive? |
| P1 | salad_dressing | ProprioNoStep | sus30 | 5 success | How does s0 survive 160 attack steps? |
| P1 | ketchup | VisualNoStep V6 | sus30 | 3 fail | Visual early trigger prevents grasp |
| P2 | ketchup | VisualNoStep V6 | clean | 3 success | Visual false positives without attack |

## Recommended Manual Inspection

For each priority episode, inspect:
1. **Frame at first trigger**: Is gripper near object? Has grasp started?
2. **Frame at mid-attack**: Is object dropped/slipping?
3. **Frame at episode end**: Final state — object position relative to basket?
4. **Compare Proprio vs Visual trigger frames**: What visual feature triggers VisualNoStep that ProprioNoStep ignores?

## Data

Full extraction: `/data/liuyu/tmp_manual_audit_extract.csv` (35 rows)
Step records: `/data/liuyu/outputs/milestone_2h_sustained_proxy_full10x5_sus30_20260530/runs/libero_object/`
Frames: Under each run_dir's `frames/` subdirectory
