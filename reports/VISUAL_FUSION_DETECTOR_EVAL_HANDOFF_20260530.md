# Visual/Fusion Detector Evaluation Handoff

**Date**: 2026-05-30 | **Status**: OFFLINE COMPLETE, ONLINE BLOCKED

## Results

### Clean-Data Attack-Relevance (42 episodes)

| Model | Oracle Corr | SUS30 Corr | Coverage |
|-------|------------|-----------|----------|
| ProprioNoStep | -0.175 | +0.093 | 100% |
| VisualNoStep | **+0.530** | **+0.582** | 100% |
| VisualProprioNoStep | **+0.530** | **+0.582** | 100% |

### Oracle-Data Attack-Relevance (oracle episodes)

| Model | Burst-Failure Corr |
|-------|-------------------|
| ProprioNoStep | **+0.592** |
| Visual | TBD (online blocked) |
| VisProprio | TBD (online blocked) |

## Key Finding

**VisualNoStep achieves r=+0.58 SUS30 correlation on CLEAN data, matching ProprioNoStep's oracle-data burst correlation (r=+0.59).** Visual models predict vulnerability without needing attack feedback.

## Online Pilot: BLOCKED

`OnlineDetector` is hardcoded to `N_PROPRIO` input dimension. Visual models (2176-dim) cannot be loaded without modifying the runner. Per sustained proxy freeze rules, no code changes allowed.

## Recommendation

**Keep ProprioNoStep** for current command-layer experiments. The visual offline result is promising but insufficient to justify code changes until:
1. Oracle-data visual evaluation is run (requires code change)
2. Visual coverage verified in online setting (currently 100% offline)

**Future**: When code freeze is lifted, add variable input-dim support to OnlineDetector and run small online pilot with VisualNoStep on oracle/sus30 conditions.

## Valid Claims

- VisualNoStep shows promising clean-data vulnerability prediction (r=+0.58)
- ProprioNoStep remains the only battle-tested online detector
- Offline evaluation confirms visual signal is not just task identity
- Visual/VisProprio behave identically (visual features dominate)

## Forbidden Claims

- VIS attack / visual attack
- Visual detector proven superior online
- Proprio is globally optimal
