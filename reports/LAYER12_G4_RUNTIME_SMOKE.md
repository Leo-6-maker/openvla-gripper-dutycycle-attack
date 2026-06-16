# Layer 1/2 G4 — Runtime Smoke

**Date:** 2026-06-16 | **Commit:** `e644a87`

## G4 Runtime Smoke: PASS

`D5FrozenOnlineDetectorV1` deployed in live LIBERO episode loop on GPU2,6.

```
6 parents × 1 episode each
All 6: detector initialized without SHA errors
All 6: detector.update() called per step without crashes
All 6: audit records written (candidates, features, scores, abstain, emit)
```

### Latency

| Parent | Steps | Candidates | avg_det_us | avg_vla_us |
|--------|-------|------------|------------|------------|
| alphabet_soup_s2 | 280 | 51 | 637 | 1,115,238 |
| bbq_sauce_s27 | 280 | 31 | 351 | 1,165,821 |
| butter_s2 | 280 | 44 | 571 | 1,135,546 |
| orange_juice_s8 | 280 | 33 | 462 | 1,170,337 |
| tomato_sauce_s2 | 280 | 35 | 406 | 1,174,984 |
| alphabet_soup_s17 | 280 | 19 | 182 | 1,086,546 |

Detector latency: 182–637 μs. VLA inference: ~1.1 s. Detector overhead < 0.06%.

### Abstain gates

All 6 episodes: detector correctly identifies `too_early` abstain on early
candidates. No abstained candidate emitted. First non-abstained candidates
score below tau=0.050 → no emission. This is correct behavior.

## G4 Known Issues (NOT PASSED)

### 1. Timing order: post-env.step (WRONG)

The current runner calls `detector.update()` AFTER `env.step()`, reading
post-step qpos/EEF from `obs`/`info`. The D5 training schema uses PRE-step
proprioception. This causes a one-tick offset between capture-time features
and live features.

**Fix required:** Read qpos/EEF BEFORE `env.step()`, pass to detector,
then step.

### 2. No OFF/ON comparison

The runner only creates `*_g4_shadow` episodes. No reference (OFF) episodes
were run, and no comparison against existing reference episodes was performed.
The `action_hash_pre == action_hash_post` check only proves the Python variable
wasn't modified in-place — not that the rollout trajectory is identical.

**Fix required:** Run matched OFF (reference mode) and ON (shadow mode) episodes,
compare action hashes, env action hashes, trace lengths, and success.

### 3. 6/6 success=False

All 6 episodes reached max_steps=280 with success=False. The original sidecar
smoke parents (same tasks/states) completed with success=1 in 111–168 steps.
Root cause likely: runner script uses different env setup, preprocessing, or
success checking than the standard collector (`run_d4_clean_shadow.py`).

**Fix required:** Use standard collector with `--detector-mode` flag rather
than custom runner. Verify the original 5 parents still succeed.

### 4. gripper_qpos fallback to 0.0

```python
try:
    gripper_qpos = float(info.get("gripper_qpos", [0.0])[0])
except:
    gripper_qpos = 0.0
```

If `info` doesn't contain `gripper_qpos`, the detector receives 0.0 silently.
This masks missing proprioception data.

**Fix required:** Use the standard collector's qpos extraction path.

## G4 Status

| Sub-gate | Status |
|----------|--------|
| Runtime smoke (detector loads, runs, doesn't crash) | PASS |
| SHA gates operational | PASS |
| Latency within budget | PASS |
| Abstain gates functional | PASS |
| Correct pre-step timing | FAIL |
| OFF/ON action identity | NOT TESTED |
| Success preservation | FAIL (6/6 false) |
| Standard collector integration | NOT DONE |

**Overall G4: NOT PASSED — requires corrected canary with standard collector.**
