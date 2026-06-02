# Proxy Burst Duration Code Audit

**Generated**: 2026-05-30 02:30 CST

## Root Cause

Attack burst duration is controlled by `attack_remaining = det_out["trigger_duration"]`, which is always **5** (from `--detector_trigger_duration 5`) for both oracle and proxy. There is NO independent attack_burst_steps parameter.

```python
# Line 450-457 of run_official_eval_artifact_rich.py
if det_out["trigger_now"] and attack_remaining == 0 and attack_condition != "clean":
    attack_remaining = det_out["trigger_duration"]  # always 5
if attack_remaining > 0 and attack_condition != "clean":
    env_action = attack_action(env_action, attack_condition, attack_rng)
    attack_applied = True
    attack_remaining -= 1
```

Cooldown is 0, so `attack_remaining == 0` gate is always open when not actively attacking.

## Why Oracle Burst Appears Longer (28-143 steps)

Oracle creates a **positive feedback loop**:
1. Oracle sets gripper action to +1.0 (full open)
2. Gripper opens physically → detector re-triggers (sustained hazard)
3. Task fails to progress → episode extends to max_steps=290
4. attack_remaining keeps getting reset to 5 by new triggers
5. This appears as a continuous attack burst in the data

## Why Proxy Burst is Short (9-19 steps)

Proxy lacks the feedback loop:
1. Proxy inverts gripper action sign
2. Gripper does NOT physically respond (MuJoCo dynamics absorb the signal)
3. Detector stops triggering after the window passes
4. attack_remaining runs from 5→0, attack stops
5. Episode completes normally (129-205 steps)

## Conclusions

1. **Proxy burst duration limits**: attack_remaining=5, cooldown=0. No sustained attack possible without re-triggering.
2. **Proxy direction**: Inverts gripper sign. May or may not point toward open depending on original action.
3. **Proxy strength**: Action-level inversion only. No amplification. MuJoCo dynamics absorb it.
4. **Oracle vs proxy burst asymmetry**: Caused by feedback loop (oracle) vs no loop (proxy), not by different code paths.
5. **attack_burst_steps**: Does not exist. Would need new parameter and code change.
6. **Sustained-open proxy**: Would require code change to keep attack_active for longer or independent burst_steps.

## Final Classification

**final_diagnosis = proxy_strength_insufficient_due_to_no_feedback_loop_and_short_burst**

- attack_remaining=5 is identical for both conditions
- Oracle gets sustained attack via feedback (gripper opens → re-triggers)
- Proxy gets no feedback (gripper doesn't respond → detector stops triggering)
- To fix without code changes: NOT POSSIBLE
- Requires: independent attack_burst_steps OR sustained-attack mode OR stronger per-step perturbation

## Recommendation

**Do NOT sweep threshold/duration tonight.** Current parameters cannot change the fundamental mechanism. The 5-step attack_remaining is hardcoded and identical for all conditions. Changing trigger_duration to 10 would increase burst to 10 steps, but:
- This also changes the DETECTOR's trigger window length
- Confounds detection sensitivity with attack strength
- Still unlikely to break contact at current inversion strength

Next: Design sustained-attack proxy or true VIS PGD after discussing with Leon.
