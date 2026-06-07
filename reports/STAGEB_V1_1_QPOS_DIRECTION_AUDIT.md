# Stage-B v1.1 Qpos Direction Audit

**Date**: 2026-06-07
**Source**: cream_cheese s2 VIS trace from Smoke3-B

## 1. Truth Table Confirmed

From 59 env-action to qpos-shift pairs in cream_cheese VIS trace:

| env_action_6 | samples | mean qpos_abs_sum delta | physical |
|-------------|---------|------------------------|----------|
| -1.0 (OPEN) | 51 | **+0.000793** | qpos INCREASES → OPEN |
| +1.0 (CLOSE) | 8 | **-0.001952** | qpos DECREASES → CLOSE |

**Direction confirmed**: env=-1 → qpos_abs_sum INCREASES. The current convention (`qpos_delta > 0 = toward open`) is correct.

## 2. Window-Level Qpos Detail (cream_cheese VIS [45,55])

Per-step in the attack window:

| step | env | open_bool | qpos | delta |
|------|-----|-----------|------|-------|
| 45 | -1.0 | 1 | 0.0746 | -0.0038 |
| 46 | +1.0 | 0 | 0.0708 | -0.0012 |
| 47 | -1.0 | 1 | 0.0695 | -0.0017 |
| 48 | -1.0 | 1 | 0.0678 | +0.0015 |
| 49 | -1.0 | 1 | 0.0693 | **+0.0035** |
| 50 | -1.0 | 1 | 0.0729 | **+0.0023** |
| 51 | +1.0 | 0 | 0.0751 | +0.0004 |
| 52 | +1.0 | 0 | 0.0755 | -0.0028 |
| 53 | -1.0 | 1 | 0.0727 | -0.0030 |
| 54 | -1.0 | 1 | 0.0697 | +0.0009 |
| 55 | -1.0 | 1 | 0.0706 | **+0.0030** |

**Observation**: 8 OPEN commands produce mixed per-step qpos deltas (4 positive, 4 negative). VIS-induced OPEN commands do not consistently translate to positive qpos shift at every step. This is expected — mechanical dynamics, contact physics, and gripper limits modulate the qpos response.

## 3. cream_cheese VIS qpos=-0.001 Explanation

The total shifted qpos_delta of -0.000978 results from:
- 8 OPEN steps with mean delta ≈ 0
- 3 CLOSE steps with mean delta ≈ -0.001
- Several large negative steps (45, 47, 52, 53) dominate the sum

The qpos is starting from ~0.075 (already near open), so further opening has limited mechanical range.

## 4. Recommendation

Keep `qpos_delta > 0.01` threshold for `physical_response_sensitive` but also track:
- `qpos_moves_toward_open_count` = fraction of OPEN steps where qpos INCREASES
- `qpos_moves_toward_open_mean` = mean delta only over OPEN steps

These capture the direction better than a single total sum.
