# Pilot V2 Proxy Weakness Diagnosis

**Generated**: 2026-05-30 02:25 CST

## Diagnosis Summary

**final_diagnosis = proxy_burst_too_short_and_magnitude_insufficient**

The detector triggers at the same steps for both oracle and proxy (identical first_trig). The difference is in attack execution:

| Metric | Oracle | Proxy | Ratio |
|--------|--------|-------|-------|
| First trigger step | 111-197 | 111-197 | identical |
| Attack burst length | 28-144 steps | 9-19 steps | 3-8x shorter |
| Attacked action gripper | +1.00 (full open) | -0.50 to -0.99 (inverted) | different direction |
| qpos change post-trigger | stays closed | stays closed | both minimal in qpos log |
| Official SR | 2/6 | 6/6 | oracle 4/6 fails |

## Mechanism Analysis

### 1. Burst Duration (primary)
Oracle sustains attack for 28-144 steps (until episode end for failed episodes). Proxy applies attack for only 9-19 steps, then attack stops even though detector continues triggering. The short burst allows OpenVLA closed-loop recovery before contact is broken.

### 2. Action Magnitude / Direction
Oracle sets `env_action[-1] = +1.0` (full open command). Proxy inverts the original gripper action sign. If original action was +0.3 (slightly closing), proxy outputs -0.3 (slightly opening). This is much weaker than oracle's maximum-strength +1.0 open.

### 3. MuJoCo Dynamics Absorption
Even oracle's +1.0 doesn't show immediate qpos change in logging (qpos stays at +0.040 closed). This suggests qpos lags behind action by at least one physics step, or the MuJoCo gripper actuator has limited torque/inertia. A brief, moderate-strength proxy attack may be fully absorbed by the actuator dynamics before gripper physically opens.

### 4. Closed-Loop Recovery
OpenVLA observes the gripper state. When proxy inverts gripper sign briefly, the policy may immediately counteract by applying a stronger close command in the next step. The slow MuJoCo dynamics mean the gripper never physically opens before the policy corrects.

## Supporting Evidence

### Oracle failures (tomato 3/3):
- Attack starts at step 147-197
- Attack continues for 94-144 steps (until episode end)
- Episode hits max_steps=290 (timeout)
- Sustained attack prevents task completion

### Proxy successes (6/6):
- Attack starts at same steps (111-197)
- Attack lasts only 9-19 steps
- Episode completes in 129-205 steps
- Short attack, policy recovers

## Decision for Phase 3 and 4

**Phase 3 (Oracle expansion)**: ALLOWED. Gate passes.
- Oracle evidence is strong and worth expanding to salad_dressing + ketchup.
- This proves detector sensitivity is not tomato-specific.

**Phase 4 (Proxy sweep)**: CONDITIONALLY ALLOWED.
- With trigger_duration=5, proxy burst is 9-19 steps.
- Reducing trigger_duration to 3 may shorten burst further (worse).
- Increasing trigger_duration above 5 or changing cooldown may help.
- BUT: the current runner's trigger_duration may control both detection window AND attack burst.
- Recommend: run ONE test on tomato_sauce with longer trigger_duration if code supports it.
- If code change needed: STOP and report.

## Next Action

Proceed to Phase 3: Oracle expansion on salad_dressing + ketchup (12 rollouts).
Phase 4 proxy sweep deferred until burst duration issue can be addressed without code changes.
