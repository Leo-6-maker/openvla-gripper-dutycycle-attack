# Phase 7 Object — Complete Experimental Results (270 runs)

## 1. Main Matrix (sealed, 0 issues)

| Condition | Runs | FR (qualified) | Token Duty |
|-----------|:----:|:--------------:|:----------:|
| VIS (TRUE_T10) | 33/33 | 21/27 = 0.778 | 0.937 |
| RAND_T10 | 33/33 | 0/27 = 0.000 | 0.000 |

Seal SHA: 81415b8c63b07023

## 2. Control Conditions (27 runs each)

| Condition | FR | Token Duty |
|-----------|:--:|:----------:|
| SHUFFLED_T10 | 0.182 (6F/33, incl. no-emit) | 0.000 |
| Adapted Untargeted PGD | 0.000 (0F/27) | 0.000 |

## 3. Timing Conditions (27 runs each)

| Condition | FR | TASR_frame |
|-----------|:--:|:----------:|
| RANDOM-TIME | 0.074 (2F/27) | 0.993 |
| EARLY-SHIFT | 0.444 (12F/27) | 0.956 |
| **Student VIS** | **0.778 (21F/27)** | 0.937 |

> Random-Time achieves highest TASR (99.3%) but lowest FR (7.4%).
> Timing, not token-level control, determines task-level outcome.

## 4. 2x2 Method Matrix (27 runs each)

| Objective | No Lock | Hard Arm Lock | Lock Effect |
|-----------|:-------:|:------------:|:-----------:|
| TMA CE (vanilla_tma_gripper_open_ce) | 22/27 (81.5%) | 22/27 (81.5%) | 0 |
| Prefix log-ratio (Ours) | 21/27 (77.8%) | **27/27 (100.0%)** | +22.2% |

### Per-Cell Breakdown

| Cell | VIS | TMA-Vanilla | TMA-ArmLock | Ours-ArmLock |
|------|:---:|:-----------:|:-----------:|:------------:|
| salad_dressing_s0 | FFF | FFF | FFF | FFF |
| bbq_sauce_s0 | FFF | FFS | FFF | FFF |
| ketchup_s0 | FFF | FFF | FFF | FFF |
| milk_s4 | FFF | FFF | FFF | FFF |
| butter_s2 | FFF | FFF | FSS | FFF |
| alphabet_soup_s0 | FFF | FFF | FSS | FFF |
| orange_juice_s0 | FFS | FFF | FFF | FFF |
| butter_s0 | FSS | FSS | FFF | FFF |
| tomato_sauce_s0 | SSS | FSS | FFS | FFF |

## 5. Seed 789 VIS Expansion

| Cell | Result | Token Duty |
|------|:------:|:----------:|
| salad_dressing_s0 | FAIL | 1.0 |
| bbq_sauce_s0 | SUCC | 1.0 |
| ketchup_s0 | FAIL | 1.0 |
| milk_s4 | FAIL | 1.0 |
| butter_s2 | SUCC | 0.9 |
| alphabet_soup_s0 | FAIL | 0.9 |
| orange_juice_s0 | FAIL | 1.0 |
| butter_s0 | SUCC | 0.9 |
| tomato_sauce_s0 | FAIL | 0.7 |

**Seed 789: 6F/9 = 0.667** (consistent with 3-seed FR=0.778)

## 6. Four-Denominator Summary

| Denominator | Cells | Runs | VIS FR | Definition |
|-------------|:-----:|:----:|:------:|------------|
| Primary conditional | 8 | 24 | 18/24 = 75.0% | Excludes alphabet_soup + no-emit |
| Expanded conditional | 9 | 27 | 21/27 = 77.8% | All clean-qualified |
| Primary ITT | 10 | 30 | 18/30 = 60.0% | Includes no-emit (attack failure) |
| Expanded ITT | 11 | 33 | 21/33 = 63.6% | All cells including no-emit |

## 7. Detector Operating Point

| Detector | TV Recall | Formal NC FT | Specificity |
|----------|:---------:|:------------:|:-----------:|
| M1 | 116/134 (86.6%) | 1/44 (2.3%) | 97.7% |
| M1_OS | 114/134 (85.1%) | 1/44 (2.3%) | 97.7% |
| **M2/V2** | **126/134 (94.0%)** | **2/44 (4.5%)** | **95.5%** |

McNemar p = 0.0063 (M2 vs M1 on TV recall)

## 8. Key Scientific Findings

1. **Timing dominates objective**: Random 7.4% < Early 44.4% < Student 77.8-81.5%
2. **TMA CE matches Ours**: 81.5% vs 77.8% — 3.7pp difference
3. **Arm lock effect is objective-dependent**: 0 for TMA, +22.2pp for Ours
4. **Ours+ArmLock = 100% FR**: All 9 cells fail, including previously resistant tomato
5. **TASR ≠ FR**: Random achieves 99.3% TASR but 7.4% FR
6. **NC Safety**: M2 fails zero-FT gate (2/44) but improves TV recall (+7.4pp vs M1)

## 9. Grand Total: 270 runs across 9 conditions

- Main: VIS 33 + RAND 33 = 66
- Controls: SHUFFLED 33 + UNTARGETED 27 = 60
- Timing: RANDOM 27 + EARLY 27 = 54
- 2x2: TMA-Vanilla 27 + TMA-ArmLock 27 + Ours-ArmLock 27 = 81
- Seed expansion: VIS 789 = 9
- **Total: 270**
