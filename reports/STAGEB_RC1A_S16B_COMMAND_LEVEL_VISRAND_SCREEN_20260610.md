# Stage-B RC1a S16b — Broad Command-Level VIS>RAND Attack Screen

**Date**: 2026-06-10
**GitHub HEAD**: 8ecea25 (S15 freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603
**Type**: Single-seed command-level screen — NOT parent-level confirmation, NOT physical bridge

## Executive Summary

**S16b completed 24/24 jobs, 0 errors.** 12 parents (5 fresh + 7 calibration) screened at seed50 with VIS+RAND matched pairs. Two fresh command_attack_positives found: milk_s0_w240-250 (VIS 7/10, streak 4, RAND 2/10) and tomato_s2_w95-105 (VIS 10/10, streak 10, RAND 2/10). One fresh borderline (salad_s1_w50-60, VIS 5/10). Two fresh command_weak (salad_s0_w55-65 VIS 0/10, tomato_s0_w50-60 VIS 3/10). All 7 calibration parents reproduce known patterns.

**This is a single-seed screen. Do not claim parent-level command success or physical bridge from S16b alone.**

## Results

See [tables/s16b_command_level_visrand_screen.csv](../tables/s16b_command_level_visrand_screen.csv) for full data.

### Fresh Parents (5)

| Parent | VIS OPEN | Streak | RAND OPEN | Gap | Class |
|--------|----------|--------|-----------|-----|-------|
| milk_s0_w240-250 | **7** | **4** | 2 | +5 | **COMMAND_ATTACK_POSITIVE** |
| tomato_s2_w95-105 | **10** | **10** | 2 | +8 | **COMMAND_ATTACK_POSITIVE** |
| salad_s1_w50-60 | 5 | 2 | 2 | +3 | BORDERLINE |
| tomato_s0_w50-60 | 3 | 2 | 1 | +2 | COMMAND_WEAK |
| salad_s0_w55-65 | 0 | 0 | 1 | -1 | COMMAND_WEAK |

### Calibration Parents (7)

| Parent | VIS OPEN | Streak | RAND OPEN | Gap | Class | Confirms |
|--------|----------|--------|-----------|-----|-------|----------|
| milk_s0_w70-80 | 8 | 4 | 0 | +8 | POSITIVE | anchor positive |
| milk_s0_w235-245 | 5 | 3 | 2 | +3 | BORDERLINE | S15b pattern |
| milk_s0_w230-240 | 4 | 2 | 2 | +2 | BORDERLINE | S15b pattern |
| cream_s2_w50-60 | 2 | 1 | 0 | +2 | WEAK | S12b cmd_weak |
| butter_s0_w80-90 | 1 | 1 | 1 | 0 | WEAK | S12a pattern |
| tomato_s2_w155-165 | 4 | 3 | **8** | -4 | CONFOUNDED | S11b pattern |
| cream_s0_w85-95 | 4 | 1 | **3** | +1 | CONFOUNDED | S15b pattern |

## Key Findings

### 1. milk_s0_w240-250 — first non-anchor milk command-positive

VIS 7/10 OPEN, streak=4, RAND 2/10. This breaks the pattern from S15 where late milk (w230-240, w235-245) only achieved VIS 5/10 streak=3. The w240-250 window, despite being even later in the episode, shows stronger attack transfer. **Milk's vulnerability window extends beyond w70-80 but remains temporally localized.**

### 2. tomato_s2_w95-105 — 10/10 complete takeover

VIS 10/10 OPEN, streak=10, qpos_pos=0.659. This is the strongest single-seed command result across all S16b parents. However, tomato family has systematic RAND contamination history: w155-165 RAND 2-8/10, w150-160 RAND 2-5/10, w90-100 RAND 2-3/10, w55-65 RAND 0-4/10. **Single-seed RAND clean at seed50 is insufficient. Requires 3-seed RAND-veto before any claim.**

### 3. salad — Layer1 false positive

salad_s0_w55-65: Layer1 cmd_specific (pV=0.8, risk_rand=0.0) but VIS 0/10 OPEN. salad_s1_w50-60: VIS 5/10 borderline. **Layer1 command selectivity does not guarantee VIS attack transfer on salad.** This is valuable as a Layer1 failure mode example.

### 4. Calibration consistency

All 7 calibration results reproduce known patterns from S9-S15. No systematic drift in seed50 relative to prior seeds. The S16b screen methodology is internally consistent.

## Classification Criteria

```
COMMAND_ATTACK_POSITIVE:
  VIS open >= 6/10 AND streak >= 4
  AND RAND open <= 2/10
  AND VIS-RAND gap >= 4
  AND VIS streak > RAND streak

BORDERLINE:
  VIS open 4-5/10
  OR VIS-RAND gap 2-3
  OR other intermediate patterns

COMMAND_WEAK:
  VIS open <= 3/10

RANDOM_CONFOUNDED:
  RAND open >= 3/10
```

## Infrastructure

| Gate | Result |
|------|--------|
| 24/24 summary JSON | PASS |
| 24/24 trace CSV | PASS |
| 24/24 infra=ok | PASS |
| 0 FAILs / CUDA / EGL | PASS |
| 3 GPU pairs released | PASS |

## Claim Boundary

### Allowed

- S16b found 2 fresh single-seed command_attack_positives at seed50.
- Calibration parents reproduce previous patterns, confirming internal consistency.
- milk_s0_w240-250 is the first non-anchor milk command-positive window.
- tomato_s2_w95-105 is a very strong VIS command-positive candidate at seed50, but requires RAND-veto.

### Forbidden

- Single-seed result as parent-level confirmation.
- tomato positive before 3-seed RAND-veto.
- Command success as physical bridge.
- Milk task-wide bridge.
- Non-milk bridge established.
- Layer3 solved / Detector solved.

## Next Step

S16c: Parent-level command confirmation with seeds 51-53 for milk_s0_w240-250, 3-seed RAND-veto for tomato_s2_w95-105, and local ORACLE references.

## Artifacts

| Artifact | Path |
|----------|------|
| S16b report | `reports/STAGEB_RC1A_S16B_COMMAND_LEVEL_VISRAND_SCREEN_20260610.md` |
| Results table | `tables/s16b_command_level_visrand_screen.csv` |
| Funnel census | `tables/s16a_object_task_state_window_funnel.csv` |

Server: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s16b_command_level_visrand_screen/`
