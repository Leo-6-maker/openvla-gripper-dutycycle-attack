# Online Window Detector — Next 80 Candidate Dataset Plan

**Date**: 2026-06-07
**Status**: Draft

## Goal

Build a balanced 80-window dataset for online window detector training,
with clean-forward features only (no VIS outcome leakage).

## Current Inventory

| Source | Windows | Task Coverage | Clean Traces |
|---|---|---|---|
| `object_phase_response_labels_v2.csv` | 31 | 9 tasks (all LIBERO Object) | 10/31 |
| ProprioNoStep shadow runs | 12 episodes | 6 tasks × 2 seeds | 12 episodes |

**Existing usable**: 10 windows with clean traces + VIS labels.
**Gap**: need 70+ more windows.

## Selection Strata (target 80 total)

### Stratum A: High-Risk Pre-Grasp (20 windows)

Windows where the model is about to approach/grasp an object.
Gripper is CLOSED (qpos < 0.03), EEF is moving toward object.
High prior probability of VIS vulnerability.

Selection:
- 10 from existing label positives (phys_bridge=1)
- 10 new candidates from unlabeled states:
  - For each of 8 tasks, select 1-2 additional seeds not in existing labels
  - Window: pre-grasp phase (~20 steps before typical grasp timing)
  - Use qpos proximity heuristic: qpos < 0.03 AND eef_z decreasing (descending)

### Stratum B: Random Pre-Grasp (20 windows)

Randomly selected pre-grasp windows from LIBERO Object episodes.
Not biased by known positive labels. Serves as representative sample.

Selection:
- For each of 8 tasks, 2-3 random seeds
- Random start within [5, episode_length - 25]
- Window length: 15-20 steps
- Constraint: window_start must have qpos < 0.03 (gripper closed)

### Stratum C: Middle/Control (20 windows)

Windows during mid-episode where gripper is neither fully closed nor fully open.
These test the detector's ability to reject non-vulnerable phases.

Selection:
- 10 windows where gripper is partially open (0.01 < qpos < 0.035)
- 5 windows during arm transit (high EEF velocity, no object interaction)
- 5 windows during post-grasp hold (object in gripper, stable)

### Stratum D: Hard Negative / Post-Open (20 windows)

Windows where gripper is already OPEN or close to OPEN.
These should NOT be attacked — ceiling guard.

Selection:
- 10 windows where qpos > 0.035 (gripper already open)
- 5 windows after task completion (done=True region)
- 5 windows during early episode idle phase (arm at rest, gripper closed)

## Task Balance

| Task | Stratum A | Stratum B | Stratum C | Stratum D | Total |
|---|---|---|---|---|---|
| ketchup | 3 | 3 | 3 | 3 | 12 |
| butter | 3 | 3 | 2 | 2 | 10 |
| cream_cheese | 3 | 3 | 2 | 2 | 10 |
| salad_dressing | 3 | 3 | 2 | 2 | 10 |
| bbq_sauce | 2 | 2 | 3 | 3 | 10 |
| milk | 2 | 2 | 3 | 3 | 10 |
| alphabet_soup | 2 | 2 | 3 | 3 | 10 |
| tomato_sauce | 1 | 1 | 1 | 1 | 4 |
| orange_juice | 1 | 1 | 1 | 1 | 4 |
| **Total** | **20** | **20** | **20** | **20** | **80** |

## Implementation Phases

### Phase 1: Consolidate existing (10 windows)

Already have clean traces + VIS labels for 10 windows from ProprioNoStep shadow runs.
Use these immediately for detector v0 prototype.

### Phase 2: Clean trace generation (70 windows)

For each new window:
1. Run clean LIBERO Object rollout for the task+seed (max 300 steps)
2. Save per-frame proprio/action/EEF features
3. Save OpenVLA clean forward logits (if GPU budget allows)

Estimated cost: 70 episodes × ~2 min = ~140 GPU-minutes.
With 3 GPU pairs parallel: ~25-30 minutes.

### Phase 3: VIS attack labeling (subset)

Not all 80 windows need VIS labels initially.
For detector v0, use the 10 existing labeled windows.
As budget allows, add VIS attacks on top-K detector predictions.

## Gate Criteria

Before proceeding to full 80-window dataset:
1. Detector v0 trained on 10 existing windows must show above-random signal
2. At least 4 positive and 4 negative windows available for training
3. Feature importance analysis shows which feature groups contribute

## Feature Coverage (35 features from v0 extractor)

| Group | Count | Description |
|---|---|---|
| Gripper qpos | 8 | mean, std, min, max, at_start, range, is_closed, is_open |
| Gripper action | 7 | open/close count, rate, mean, std, switches, streak |
| Raw gripper | 2 | mean, std before normalize |
| End-effector | 5 | displacement, velocity, z-mean/std/trend |
| ProprioNoStep | 6 | hazard mean/max, thresholds, release, phase |
| Temporal (pre→window) | 2 | qpos delta, action delta from pre-window |
| Window position | 5 | start/center/length fractions, step, remaining |
| **Total** | **35** | |
