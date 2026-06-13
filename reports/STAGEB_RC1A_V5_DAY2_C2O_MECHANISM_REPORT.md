# V5 Day 2 C2O Mechanism Diagnostic Report

**Date:** 2026-06-13
**Server:** klfy-SYS-4028GR-TR2
**Branch:** exp/vis-prefix-margin-repair-20260603
**HEAD:** f1292c4 (modified: v5 runner fixes, TARGET_OBJECT_GUESS, invariants)
**Runner SHA256:** 4e6d97d4 (modified from f1292c4)

## Executive Summary

**Final Label: `TRAJECTORY_UNSTABLE_PARENT`** — `LOCAL_C2O_NOT_REPRODUCED`

Day 2 systematically tested whether cream_cheese_s35_w77_83_c80 seed=99 C2O could be reproduced under controlled conditions. Results show the C2O event was an artifact of mujoco trajectory non-determinism, not a stable local vulnerability. The clean policy at step 80 varies between OPEN and CLOSE across different walks, making the C2O event trajectory-dependent and non-reproducible on demand.

## Gates Summary

| Gate | Condition | Result | Action |
|------|-----------|--------|--------|
| G0 | 12/12 invariants | PASS | Proceed |
| G1-A | step80 CLOSE >=2/3 | **FAIL** (0/3) | Frozen diagnostic only |
| G1-B | RAND C2O <=1/3 | PASS (0/3) | No RAND confound |
| G2 | Snapshot fidelity | **SKIPPED** | G1-A failure |
| G3-G6 | - | **SKIPPED** | No parent qualified |

## Phase Results

### Phase 0 — Preflight
- 12/12 runner invariants PASS (after fixing invariant #8: attack_result None→RuntimeError)
- GPU 4,5 available and used; GPU 0,1,2,3 idle; GPU 6,7 running clean scan (277/400)
- Python 3.10.16, openvla_official_libero_20260525

### Phase 1 — Parent Provenance + RAND Gate
- **3 clean walks:** step80 OPEN 3/3 (0% CLOSE)
- **3 RAND walks (seeds 99,100,101):** step80 OPEN 3/3
- **G1-A FAILED:** Parent is trajectory-unstable — step80 not reliably CLOSE
- **G1-B PASSED:** RAND shows no independent C2O effect (clean already OPEN)
- **Label:** `TRAJECTORY_UNSTABLE_PARENT`

### Phase 2 — Frozen Observation Replay
- Walked with smoke-matched timing (no dummy wait, 80 policy steps)
- Clean at step80: OPEN (-1.0), not CLOSE
- VIS seed99×5: 0/5 C2O (3/5 CLOSE, 2/5 OPEN)
- VIS seed100×3: 0/3 C2O (2/3 CLOSE, 1/3 OPEN)  
- VIS seed101×3: 0/3 C2O (2/3 CLOSE, 1/3 OPEN)
- RAND: 3/3 OPEN
- **Label:** `CLEAN_NOT_CLOSE_NO_C2O_TESTABLE`

### Phases 3-6 — Skipped
No parent qualified for closed-loop or full rollout (G1-A failed).

## Root Cause Analysis

| Hypothesis | Evidence | Verdict |
|-----------|----------|---------|
| A. Pre-window trajectory divergence | 0/4 walks reproduced CLOSE at step80 | **PRIMARY** |
| B. PGD random-start instability | Not testable (clean never CLOSE) | Unconfirmed |
| C. logsumexp/region objective mismatch | Day 1: CE→0 but no action flip | Secondary |
| D. Random-sensitive window | RAND gate passed | Ruled out |
| E. Single-step C2O too sparse | Not testable | Unconfirmed |

**Primary finding:** The C2O event at cream_cheese_s35 step80 was observed in specific mujoco trajectories that happened to produce CLOSE at that step. These trajectories cannot be reliably reproduced due to floating-point non-determinism in the physics simulation. The "C2O" was not a stable attack effect but a trajectory-conditioned event.

## Artifacts

| File | Path |
|------|------|
| Parent provenance | tables/s20d_v5_day2_parent_provenance.csv |
| Frozen event CSV | tables/s20d_v5_day2_frozen_event_replay.csv |
| Frozen observation | frozen_events/cream_cheese_s35_step80_frozen.npz |
| This report | reports/STAGEB_RC1A_V5_DAY2_C2O_MECHANISM_REPORT.md |

## Recommended Day 3 Actions

1. **Address trajectory divergence:** Use exact replay (record clean action sequence from trace, replay open-loop) to achieve deterministic state reproduction
2. **Fix objective:** Replace logsumexp(open_region) with max(open_logits) to couple loss with argmax
3. **Broader event bank:** Build frozen event bank from all 35 clean_CLOSE smoke events for objective comparison
4. **Test exact state replay:** Save mujoco state at window start, restore for each branch, verify identical clean trajectories

## Allowed Labels Used

- `TRAJECTORY_UNSTABLE_PARENT`
- `LOCAL_C2O_NOT_REPRODUCED`
- `CLEAN_NOT_CLOSE_NO_C2O_TESTABLE`

## Forbidden Labels (Not Used)

- `CONFIRMED_CMD`, `CONFIRMED_PHYSICAL`, `Layer3 success`
- `PARENT_LEVEL_MULTIRESTART_CMD_CONFIRMED`
- `DETERMINISTIC_PARENT_CANDIDATE`
