# Object-100 Online Shadow Validation Plan

**Status**: READY TO LAUNCH (pending Stage 2 calibration results + user approval)
**Date**: 2026-05-29

## Purpose
Run normal clean Object rollouts while the detector runs online and records trigger outputs. No actions are modified. This validates that the offline replay behavior translates to live streaming inference.

## Shadow Mode Definition
- Policy action remains clean (OpenVLA `generate()` output)
- Detector receives only causal history (timestep < t)
- Detector outputs trigger fields each step
- Trigger is **logged but not executed**
- No visual perturbation, no oracle open, no random control

## Recommended Tasks (small shadow set)
Two stable Object tasks, 5 states each = 10 episodes:

| Task | States | Rationale |
|------|--------|-----------|
| cream_cheese (or ketchup) | s0-s4 (5) | Stable pick-place, well-defined contact window |
| milk (or salad_dressing) | s0-s4 (5) | Stable pick-place, different object geometry |

## Detector Configuration
- **Model**: ProprioNoStep TCN (38,602 params)
- **Inputs**: 13 proprioceptive (gripper, EEF, action history)
- **History**: 16 timesteps
- **Hazard threshold**: 0.1 (from replay audit — high sensitivity for 99% coverage)
- **Trigger duration**: 5 consecutive steps
- **Cooldown**: 10 steps after trigger

## Logged Fields Per Step
| Field | Description |
|-------|-------------|
| episode_key | Unique episode identifier |
| task_name | LIBERO task name |
| state_id | LIBERO state index |
| step_idx | Timestep within episode |
| detector_phase | Phase classification (0-7) |
| hazard_score | Raw hazard logit |
| release_safe_score | Raw release-safe logit |
| confidence | Phase confidence |
| trigger_now | Boolean: trigger active this step |
| trigger_duration | Consecutive steps triggered |
| trigger_reason | e.g. "consecutive_5" |
| clean_action | OpenVLA action [dx, dy, dz, dgripper] |
| gripper_qpos | Current gripper position |
| gripper_width | Current gripper width |
| eef_pos | End-effector position [x, y, z] |
| image_path | RGB frame path |
| NO attack fields | No perturbation, oracle, or random |

## Post-Shadow Evaluation
After 10 episodes complete:
1. Compare shadow trigger with teacher window (offline privileged labels)
2. Compute:
   - Window coverage (should match offline 0.991)
   - False early strict
   - False early tolerant (±1, ±2, ±3)
   - Miss rate (should be 0)
   - Trigger latency (should match offline 0.2 mean)
   - Failed episode FP (should be 0)
   - Mechanism-ineligible FP (should be 0)
3. If shadow metrics match offline metrics within 5%: shadow validation PASSES

## Gate Criteria
- [ ] Shadow coverage > 0.90 ✓ (from offline: 0.991)
- [ ] Shadow miss = 0 ✓ (from offline: 0)
- [ ] Shadow FP on failed = 0 ✓ (from offline: 0)
- [ ] Shadow latency within ±2 steps of offline (0.2 mean)
- [ ] No online-specific errors (NaN, crash, OOM)

## Output
- `/data/liuyu/outputs/milestone_2f_object100_online_shadow_validation_20260527/tables/shadow_validation_results.csv`
- `/data/liuyu/outputs/milestone_2f_object100_online_shadow_validation_20260527/reports/SHADOW_VALIDATION_STATUS.md`

## Blocking
- Do not launch until user approves
- Do not launch until Stage 2 fusion calibration confirms best trigger policy
- Script: `scripts/run_detector_shadow_rollout.py` (prepared)
- Config: `configs/object100_shadow_validation.yaml` (prepared)
