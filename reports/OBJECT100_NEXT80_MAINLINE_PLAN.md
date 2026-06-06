# Object100 Next80 Mainline Candidate Plan

**Date**: 2026-06-07
**Based on**: Teacher window sanity audit (73/74 usable, 0 too-late)

## Data Inventory

| Source | Count | Status |
|---|---|---|
| Object100 teacher windows (contact/transfer) | 74 | 73 usable |
| Object100 episodes with step_records | 90 | all available |
| Existing31 VIS windows (pre-grasp) | 31 | legacy diagnostic only |
| Reanchored windows | 1 | 1 extracted |

## Selection Strata

### Stratum A: High-Confidence Teacher Windows (20)

Windows directly from Object100 teacher labels with mechanism_eligible=True and clean_success=True.
These are the highest-quality windows from the privileged teacher detector.

Selection: top 20 by mechanism confidence, balanced across 9 tasks.

### Stratum B: Random Pre-Release / Carry Windows (20)

Windows in the late-carry phase, just before the gripper naturally opens for release.
These test whether the detector can identify windows where inducing premature OPEN
would be most impactful.

Selection: For each episode, pick 20-step window ending 5 steps before final_release_step.
If final_release_step is unknown, use step range [tws-25, tws-5] relative to teacher window.

### Stratum C: Early Carry / Pre-Grasp Controls (20)

Windows in the early approach or pre-grasp phase where gripper is CLOSED.
These are negative controls — the model is not in a transfer state.

Selection: Windows where gripper_qpos < 0.03 (CLOSED), before first_approach_open.
Window length: 15-20 steps.

### Stratum D: Post-Release / Natural-Open Hard Negatives (20)

Windows after the final release where gripper is already OPEN.
These should NOT be attacked — ceiling guard test.

Selection: Windows where gripper_qpos > 0.035, after final_release_step.
Window length: 10-15 steps.

## Task Balance Target

| Task | Stratum A | Stratum B | Stratum C | Stratum D | Total |
|---|---|---|---|---|---|
| alphabet_soup | 3 | 3 | 3 | 2 | 11 |
| bbq_sauce | 3 | 3 | 2 | 2 | 10 |
| butter | 3 | 2 | 2 | 2 | 9 |
| cream_cheese | 2 | 3 | 2 | 2 | 9 |
| ketchup | 3 | 3 | 3 | 3 | 12 |
| milk | 2 | 2 | 3 | 3 | 10 |
| salad_dressing | 2 | 2 | 2 | 2 | 8 |
| orange_juice | 1 | 1 | 1 | 2 | 5 |
| tomato_sauce | 1 | 1 | 2 | 2 | 6 |
| **Total** | **20** | **20** | **20** | **20** | **80** |

## Feature Availability

All 80 windows have Object100 clean step_records with:
- gripper_qpos, gripper_width, gripper_command
- eef_x/y/z position
- raw_action, action_gripper
- step_idx, done

Online-legal window-level features (35-dim) already extracted for all windows.

## Current Labels

| Label type | Available | Needed |
|---|---|---|
| VIS targeted PGD20+ | 16 (existing31 only) | 80 |
| Matched random Linf | 0 | 80 |
| command_susceptible | 16 | 80 |
| physical_bridge | 16 | 80 |

## Implementation Order

1. **Phase 1**: Train detector v0 on 73 teacher windows using teacher mechanism labels (NOT VIS labels)
   - Use mechanism_eligible as binary label
   - Train LR/RF baselines with 35 online-legal features
   - Leave-task-out eval

2. **Phase 2**: If detector v0 shows signal (AUROC >= 0.65 on teacher labels):
   - Select 20 windows (10 high-score, 10 low-score) for VIS PGD20 labeling
   - Run matched random controls
   - Compute command_susceptible labels per protocol v2

3. **Phase 3**: If VIS labels confirm detector signal:
   - Expand to full 80-window VIS labeling
   - Train detector v1 on command_susceptible labels

## Gate Criteria

- Phase 1 → 2: detector v0 AUROC >= 0.65 on teacher mechanism labels (leave-task-out)
- Phase 2 → 3: precision@10 >= 0.5 on VIS command_susceptible labels
- Phase 3 completion: >= 60 windows with command_susceptible labels

## Exclusions

- Do NOT include existing31 pre-grasp VIS windows in mainline training
- Do NOT use VIS attack outcome as detector feature
- Do NOT use teacher mechanism label as final label (only as training proxy)
