# Phase E Canary Root Cause — 2026-06-05

**Final classification**: `INVALID_ACTION_SPACE_CONFOUNDED` (not BUDGET_TOO_BROAD, not MEASUREMENT_BUG)

## Evidence

### Observation
- cream_cheese_s4: VIS_OPEN=10/10, done=False, qpos_delta=-6.9e-05
- bbq_sauce_s5: VIS_OPEN=10/10, done=False, qpos_delta=-7.8e-05

### Investigation
1. qpos measurement: Canary script uses `env.sim.data.qpos` (Mujoco primary) — measurement chain IS correct
2. vis_open=10/10: PGD succeeds at flipping decoded gripper token → confirms policy-level attackability
3. PHYSICAL qpos_delta ≈ 0: Mujoco gripper qpos does NOT open

### Root cause
Canary script feeds **raw decoded action** to `env.step()`:
```python
obs, reward, done, info = env.step(adv_act)  # WRONG
```

Correct LIBERO pipeline (`vis_rollout_adaptive_v3.py` L538-540):
```python
env_action = normalize_gripper_action(raw_action, binarize=True)
env_action = invert_gripper_action(env_action)
obs, reward, done, info = env.step(env_action)  # CORRECT
```

Normalize: raw[0,1] → [-1,1]. Invert: flip sign.
- raw 0.0 (OPEN) → normalize: -1.0 → invert: +1.0 (env OPEN)
- raw 0.0 fed directly → env sees 0.0 → NEUTRAL, not OPEN

### Consequence
- vis_open=10/10 means PGD flipped decoded token (policy-level success)
- qpos_delta ≈ 0 means physical gripper did NOT open (action not in env space)
- done=False likely from arm action drift, not gripper opening
- Results DO NOT characterize low-budget VIS as too broad

## What the canary DOES tell us

1. eps=4, steps=10 PGD CAN flip the decoded gripper token on both samples
2. Policy-level attack is achievable at low budget
3. Physical transfer requires CORRECT env action injection

## Fix required

Phase E script must apply normalize_gripper_action + invert_gripper_action before env.step().
After fix, re-run 2-sample canary on GPU 4,5 when free.

Formal script status: `scripts/diagnostics/run_phase_e_canary.py` now enforces the official action transform and records `previous_phase_e_v0_status=INVALID_ACTION_SPACE_CONFOUNDED`.

## Do NOT

- Expand to 5-sample with current script
- Conclude "low-budget VIS too broad"
- Mark as silver label
