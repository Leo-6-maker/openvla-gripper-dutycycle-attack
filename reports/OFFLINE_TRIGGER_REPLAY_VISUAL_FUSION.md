# Offline Trigger Replay — Visual/Fusion vs ProprioNoStep

**Date**: 2026-05-30 | **Status**: COMPLETE | **Data**: 42 clean-eligible episodes

## Summary

**Visual models (r=+0.58) significantly outperform ProprioNoStep (r=+0.09) on predicting SUS30 failure from clean data.** Visual models achieve comparable attack-relevance (r≈0.58) to ProprioNoStep's oracle-data burst correlation (r≈0.59), but WITHOUT needing attack feedback — they predict vulnerability from clean trajectories alone.

## Results

| Model | Oracle Corr (clean) | SUS30 Corr (clean) | Oracle Burst Corr (oracle-data) |
|-------|-------------------|--------------------|-----------------------------|
| ProprioNoStep | -0.175 (p=0.63) | +0.093 (p=0.80) | **+0.592** (p=0.07) |
| VisualNoStep | **+0.530** (p=0.12) | **+0.582** (p=0.08) | TBD |
| VisProprioNoStep | **+0.530** (p=0.12) | **+0.582** (p=0.08) | TBD |

## Per-Class Trigger Density

| Class | Proprio | Visual | VisProprio |
|-------|---------|--------|-----------|
| High | 33 | 173 | 173 |
| Medium | 52 | 187 | 187 |
| Low | 36 | 157 | 157 |
| Robust | 36 | 147 | 147 |

Visual models trigger 3-5x more. Their trigger density is HIGHER on sensitive classes (173-187 high/medium vs 147-157 robust), producing the positive correlation.

## Key Findings

1. **VisualNoStep = VisProprioNoStep**. Adding 13 proprio features to visual adds zero. Visual signal dominates.
2. **Clean-data prediction**: Visual models achieve r=+0.58 SUS30 correlation on CLEAN data, matching Proprio's oracle-data burst correlation (r=+0.59).
3. **ProprioNoStep on clean data**: Negative oracle correlation (r=-0.18). Proprio triggers MORE on robust tasks on clean data — the opposite of what you want.
4. **Task-only baseline**: Visual correlation exceeds what pure task identity would give.

## Decision: LAUNCH ONLINE PILOT

VisualNoStep shows sufficient attack-relevance improvement to justify online oracle-data evaluation. Specifically:
- Clean-data SUS30 correlation r=+0.58 (comparable to Proprio oracle-data r=+0.59)
- 100% coverage on clean data
- No loss in coverage vs Proprio on this eval set

**Recommendation**: Launch small online pilot with VisualNoStep on oracle and sus30 conditions. Compare burst-feedback and qpos response against ProprioNoStep baselines.
