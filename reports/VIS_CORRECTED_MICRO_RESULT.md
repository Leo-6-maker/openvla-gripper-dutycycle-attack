# VIS Corrected Micro Rollout Result

**Date**: 2026-06-01 | **Branch**: `exp/vis-payload-upgrade-validation-20260601` | **HEAD**: `8ff150d8`

## Bugs Fixed

| Bug | Before | After | Impact |
|-----|--------|-------|--------|
| `get_libero_image` | Single flip `[::-1]` | Double flip `[::-1, ::-1]` | All tasks failed due to mirrored image |
| `normalize_gripper_action` | Simple `>=0` threshold | Production: `2*val-1 → sign` | 0.0 maps to CLOSE, 0.996 maps to OPEN |
| `invert_gripper_action` | `1.0 - val` | `-1.0 * val` | MuJoCo compatibility |

## Configuration

- Objective: `gripper_open_region_ce`, eps=4/255, steps=20, step_size=1/255
- Window: 11-13 steps (default), around known positive no-rollout frames
- Decode: `prompt()` wrapper + action prefix token 29871 (corrected)
- GPU pairs: (1,2) and (4,5)

## Results (9 rollouts, corrected image processing)

| Task | Condition | Success | Steps | Window OPEN | Token Flips | Avg ArmL2 |
|------|-----------|---------|-------|-------------|-------------|-----------|
| cream_cheese | clean | **True** | 127 | 0/11 (0%) | 0 | 0 |
| cream_cheese | vis_pgd | **True** | 135 | **7/11 (64%)** | 7 | 0.91 |
| cream_cheese | random | **True** | 125 | 0/11 (0%) | 0 | 0.06 |
| tomato_sauce | clean | **True** | 156 | 0/13 (0%) | 0 | 0 |
| tomato_sauce | vis_pgd | **True** | 144 | **8/13 (62%)** | 6 | 0.47 |
| tomato_sauce | random | False* | 294 | 0/13 (0%) | 0 | 0.01 |
| ketchup | clean | **True** | 147 | 0/11 (0%) | 0 | 0 |
| ketchup | vis_pgd | **True** | 160 | 0/11 (0%) | 1 | 0.84 |
| ketchup | random | **True** | 154 | 0/11 (0%) | 0 | 0.03 |

*tomato random_linf failure: 0 token flips, 0 grip delta, armL2=0.01 → classified as environment fluctuation, not VIS effect.

## Key Findings

1. **VIS PGD changes online gripper behavior**: Cream cheese OPEN commands increase from 0% (clean) to 64% (VIS). Tomato from 0% to 62%.
2. **Random baseline negative**: All random_linf conditions show 0/11-13 token flips, 0 grip delta.
3. **High-sensitive > Robust**: Ketchup (robust control) shows minimal VIS effect (0% OPEN, 1 token flip).
4. **BUT task survives**: Both high-sensitive tasks succeed under VIS PGD. The perturbation window (11-13 steps) is too short to cause task failure.

## qpos Response

Gripper qpos stays near 0 during the perturbation window regardless of condition. The OPEN commands do not produce measurable qpos/width change during the short window. This is expected: the perturbation is applied during contact/transport phase where the gripper is already closed around the object.

## Decision

**Case R2 — Action-level PASS, task-level FAIL**

VIS PGD produces rollout-level decoded gripper-action changes (increased OPEN commands) that are specific (not random-reproducible) and stronger on high-sensitive tasks. But the current 11-13 step window is insufficient to cause task failure. Duration calibration is the next step.

## Valid Claims

- Corrected VIS PGD increases decoded OPEN commands during perturbation window
- Effect is specific to VIS perturbation (random does not reproduce)
- High-sensitive tasks show stronger effect than robust control
- Task-level failure not achieved under current window duration

## Forbidden Claims

- "VIS causes task failure" — not yet demonstrated
- "VIS is ready for detector-triggered evaluation" — rollout remains blocked
- "VIS effect transfers to physical gripper" — qpos response not yet confirmed
