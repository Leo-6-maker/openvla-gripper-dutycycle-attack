# Sustained Proxy Attack Design Spec

**Generated**: 2026-05-30 02:40 CST
**Status**: Design spec only — no code changes applied

## 1. Current gripper_inversion_proxy Failure Analysis

| Failure Mode | Mechanism | Evidence |
|-------------|-----------|----------|
| Burst too short | attack_remaining=5, no re-trigger feedback | 9-19 steps vs oracle 28-143 |
| No qpos response | MuJoCo dynamics absorb brief inversion | pre_q ~= post_q across all 6 proxy episodes |
| No feedback loop | gripper doesn't physically open → detector stops triggering | proxy triggers stop, oracle triggers sustain |
| Action magnitude weak | inversion of original sign ≈ -0.5 to -1.0 vs oracle +1.0 fixed | avg_att_g: proxy ≈ -0.7, oracle = +1.0 |

## 2. Oracle Feedback Loop Mechanism

```
detector triggers → oracle sets gripper=+1.0 → gripper opens physically
→ detector re-triggers (sustained hazard) → attack_remaining resets to 5
→ attack continues → episode fails → max_steps=290 timeout
```

This feedback loop is why oracle effective burst = 28-143 steps despite attack_remaining=5.

## 3. Proposed Sustained Proxy Design

### Core changes needed:

1. **Decouple trigger gate from attack burst**:
   - Add `--attack_burst_steps` (default 30) independent of `--detector_trigger_duration`
   - `attack_remaining = attack_burst_steps` on first trigger, NOT `trigger_duration`

2. **Sustained open hold**:
   - Once triggered, hold `attacked_env_action[-1] = +1.0` (open) for `attack_burst_steps`
   - Distinguish from oracle: oracle permanently forces open; sustained proxy holds open for fixed duration then releases

3. **Optional gripper-feedback extension**:
   - If `gripper_qpos` drops below threshold (actually opens), extend burst by N steps
   - Creates a bounded feedback loop

4. **Run-level provenance**:
   - `attack_burst_steps` logged in manifest
   - `burst_start` / `burst_remaining` logged in step_records
   - `burst_completed` flag on episode finish

### Files to modify:

| File | Change |
|------|--------|
| `scripts/run_official_eval_artifact_rich.py` | Add `--attack_burst_steps`, modify `attack_remaining` logic, add sustained-open hold |
| `src/gripper_attack/triggers.py` | Possibly add new attack_action mode for sustained_open |
| `scripts/launch_*.sh` | Add `--attack_burst_steps` to launch commands |
| `tests/v4/test_*.py` | Add burst step independence tests |

### New condition (optional, requires code):

`sustained_open_proxy_N` where N = burst steps (30, 60)
- NOT oracle (which is permanent open)
- NOT current proxy (which is brief inversion)
- Sustained open for N steps, then release, allow recovery

## 4. Test Plan

| Test | What it verifies |
|------|-----------------|
| `test_burst_steps_independent` | attack_burst_steps != detector_trigger_duration |
| `test_sustained_proxy_holds_N_steps` | attack_applied count = attack_burst_steps |
| `test_clean_never_attacks` | clean condition ignores attack_burst_steps |
| `test_burst_logged_in_manifest` | manifest includes attack_burst_steps |
| `test_burst_logged_in_step_records` | step_records include burst_start/burst_remaining |

## 5. Minimal Pilot Experiment

After code review and approval:
- Tasks: tomato_sauce (most oracle-sensitive)
- States: s0, s1, s2
- Conditions:
  - clean
  - oracle_open (upper bound reference)
  - sustained_open_proxy_30
  - sustained_open_proxy_60
- Total: 12 rollouts
- GPU: 2,6

## 6. Risks and Caveats

1. **Sustained proxy may overlap with oracle**: If hold duration is too long, it becomes indistinguishable from oracle. Must pick duration short enough to allow recovery but long enough to overcome MuJoCo dynamics.
2. **Command-layer limitation**: Even sustained open is still action-level, not visual. True VIS PGD operates in pixel space and may be more effective.
3. **Safety/compliance**: Must not be called VIS or visual attack.
4. **MuJoCo actuator limits**: Gripper may have max force/torque that limits opening speed even under sustained command.

## 7. Implementation Priority

1. **Priority 1**: `attack_burst_steps` parameter + decoupling (2-3 lines of code)
2. **Priority 2**: Sustained open hold during burst (5-10 lines)
3. **Priority 3**: Gripper-feedback extension (optional, 10+ lines)
4. **Priority 4**: New condition name (cosmetic, 1 line)

## 8. Decision

**Do NOT implement tonight.** This spec is ready for code review tomorrow. The current evidence (oracle strong, proxy weak) is sufficient to justify the design direction.
