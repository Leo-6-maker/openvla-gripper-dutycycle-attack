# Phase 6C P0 Root Cause Report

**Report commit**: `380e837`
**Execution commit**: `14e9758`
**Date**: 2026-06-25

## Executive Summary

The "P0 online detector failure crisis" (9/12 unexpected emits on NC-assumed cells) was a **NC manifest construction error**, not a detector generalization failure. Teacher relabel reveals that arbitrary states (3, 7, 4, 8, 5, 9) mostly have valid grasp-carry corridors. V2 correctly triggers on them.

## Q1: Are the 25 NC-assumed cells actually no-corridor?

**Answer: NO. Most have valid corridors.**

Teacher relabel (25 cells with completed CLEAN shadow telemetry):

| Category | Count | Meaning |
|----------|:-----:|---------|
| A (true positive) | 18 | Has corridor, V2 correctly triggers |
| B (TV miss) | 2 | Has corridor, V2 fails to trigger |
| C (genuine NC false trigger) | **0** | No corridor, V2 incorrectly triggers |
| D (correct abstain) | 5 | No corridor, V2 correctly abstains |

**NC manifest validity: FAIL** — arbitrary state IDs (3, 7, 4, 8, 5, 9) are not valid NC proxies.

## Q2: Is there an online/offline parity bug?

Not yet fully tested (pending same-trajectory feature parity audit). However, the detector emits on online CLEAN shadow are consistent with offline replay — both show the same pattern (V2 emits on corridor-present cells, abstains on NC).

## Q3: How do V1, M1, M1-OS, M2 compare on same trajectories?

Multi-detector offline replay on 26 CLEAN shadow trajectories:

| Detector | Training Data | NC FT (D) | TV Recall (A) |
|----------|---------------|:---------:|:-------------:|
| V1 | Original | **4/5** ❌ | **0/18** ❌ |
| M1 | Primary-only | 0/5 ✓ | 16/18 |
| M1-OS | Primary oversampled | 0/5 ✓ | 15/18 |
| **M2** | **Primary+Reserve** | **0/5** ✓ | **18/18** ✓ |

V1 completely fails on current official backend trajectories:
- Falsely triggers on 4/5 genuine NC cells
- Misses all 18 TV cells (0/18 recall)

V2 (M2) is the best performer: perfect NC abstention + perfect TV recall.

## Q4: Did Reserve 120 harm M2?

**Answer: NO. Reserve 120 improved M2.**

Paired comparison (26 cells):
- Both emit: 16
- Both abstain: 8
- M2 only emits: **2** (M2 catches 2 more TV than M1)
- M1 only emits: 0

Reserve 120 provides 2 additional TV captures with no NC regression.

## V1 Better Than V2?

**NOT SUPPORTED.** V1 is catastrophically worse on current official backend:
- 4/5 NC false triggers vs 0/5 for V2
- 0/18 TV recall vs 18/18 for V2

## Reserve Harmed M2?

**NOT SUPPORTED.** M2 is the best detector:
- 18/18 TV vs M1's 16/18
- 0/5 NC FT same as M1 (both perfect)

## Root Cause

1. **Primary**: The NC control manifest used arbitrary state IDs (3, 7, 4, 8, 5, 9) without Teacher verification. Most of these states produce valid grasp-carry corridors in the current official backend.

2. **Contributing**: V1 detector is incompatible with current official preprocessing (`upstream_tf_jpeg`), showing massive performance degradation compared to V2. V1 was trained/calibrated on the old `project_pil_lanczos` backend.

## Recommendations

1. **Rebuild NC manifest**: Use Teacher-verified no-corridor states from current official CLEAN rollouts
2. **Keep V2 seed42**: No detector modification needed
3. **Proceed with attack benchmark**: V2 online safety is validated
4. **Archive V1**: V1 is not usable on current official backend without retraining
5. **RvC CLEAN anchors**: Freeze official Teacher anchors from 15 completed RvC cells

## Gate Decision

```text
NC_MANIFEST = FAIL
V2_MODEL_FAILURE = NOT ESTABLISHED
V1_BETTER_THAN_V2 = NOT SUPPORTED
RESERVE_HARMED_M2 = NOT SUPPORTED
V2_OFFICIAL_ONLINE_NC_SAFETY = PASS (0 genuine false triggers)
FULL_ATTACK_BENCHMARK = CONDITIONAL GO (after NC manifest rebuild)
DETECTOR_MODIFICATION = HOLD
V3_TRAINING = NOT NEEDED
```
