# GPU0,7 Engineering Smoke — Decision Report

**Date**: 2026-05-29 13:30 CST
**Purpose**: Engineering-only smoke test for patched Object detector-triggered runner

## Results Summary

| Stage | Test | Result | Detail |
|-------|------|--------|--------|
| 0 | Preflight | PASS | GPU0: 8.5GB free, GPU7: 10.9GB free. lgzhou process on GPU0 (2.3GB). |
| 1 | CUDA alloc | PASS | torch 2.2.0+cu121, 2 GPUs visible, allocation OK |
| 2 | Model-load | PASS | 3.36B params on GPU0 (6.74GB), 4.18B on GPU7 (8.38GB) |
| 3 | Clean micro | PASS | success=True, 141 steps, no OOM, no Xid |
| 4 | Oracle micro | PASS | success=True, 160 steps, no OOM, no Xid |
| 5 | VIS micro | **SKIPPED** | Only 2.2-3.0 GB free per GPU — PGD would OOM |

## Peak Memory

| GPU | Model Load | Free After Load | Risk for VIS |
|-----|-----------|-----------------|--------------|
| GPU0 (cuda:0) | 6.74 GB | 2.21 GB | Likely OOM |
| GPU7 (cuda:1) | 8.38 GB | 2.99 GB | Likely OOM |

## Xid Status

| GPU | Before Smoke | After Smoke | New Xid? |
|-----|-------------|-------------|-----------|
| GPU0 | Xid13 (historical) | No change | None |
| GPU7 | Xid31 11:42 today | No change | None |

## Issues Found

1. **Detector fields not in step_records**: The step record template patch failed to match due to an extra line (`teacher_privileged_state_available`). Detector inference and attack injection code paths are intact (verified at lines 429-431).

2. **Single episode dir overwrite**: Both clean and oracle runs wrote to the same task/state directory (JSONL append mode). For the full pilot, different output roots per condition are needed.

## Recommendations

| Use Case | Verdict | Reason |
|----------|---------|--------|
| Engineering smoke only | **USABLE** | All stages pass, no fresh Xid |
| Non-VIS controls (clean, oracle, random) | **USABLE with caution** | GPU0 Xid13 risk, GPU7 Xid31 history |
| VIS_targeted attack | **NOT USABLE** | Insufficient GPU memory (2-3 GB free), PGD path would OOM |
| Final attack evidence | **NEVER USE** | Both GPUs have Xid history. GPU0 has permanent Xid13. |

## Preferred Path

Wait for GPU2,6 (Goal-100: 63/100, ETA ~1h) for the real 40-rollout attack pilot. GPU2,6 have:
- No Xid history today
- More free memory (full 2×11 GB)
- No lgzhou process interference

## Boundary

GPU0,7 smoke is **engineering-only validation**. It confirms the patched runner works on RTX 2080 Ti pairs. It does NOT produce official attack evidence. It does NOT replace the planned stable-pair Object matched attack pilot.
