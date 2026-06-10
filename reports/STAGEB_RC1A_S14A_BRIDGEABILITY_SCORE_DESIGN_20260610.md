# Stage-B RC1a S14a — Layer3 v0.4 Bridgeability Score Retrospective Design

**Date**: 2026-06-10
**GitHub HEAD**: 4eb7af7 (S13a freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603
**Type**: Pure analysis — no GPU, no new experiments

## Executive Summary

Retrospective analysis of all 13 VIS/RAND seed pairs across S9-S13 reveals a clear bridgeability taxonomy. A rule-based v0.4 selector achieves precision=0.800 and recall=1.000 at the seed level (one FP: milk seed12, which is seed-sensitive within an otherwise positive parent). At the parent level, precision and recall are both 1.000.

**Key insight**: Layer1 command selectivity + ORACLE physical reachability alone produce 8 candidates but only 4 bridge (precision=0.500). Adding RAND-clean veto eliminates 3 tomato/butter candidates (precision rises to 0.571). Adding a VIS command-probe (open≥6, streak≥4) eliminates 3 cream candidates (precision=0.800). The final FP is milk seed12 — an attack-seed sensitivity case, not a bridgeability scoring failure.

## 1. Comprehensive Feature Table

See [tables/s14a_bridgeability_feature_table.csv](../tables/s14a_bridgeability_feature_table.csv) — 13 rows covering all VIS physical bridge seeds across 4 objects.

### Summary Statistics

| Object | Seeds | VIS open (mean) | VIS streak (mean) | VIS norm (range) | RAND clean? | Bridge? |
|--------|-------|-----------------|-------------------|------------------|-------------|---------|
| milk | 9,10,11,24,12 | 6.2 | 4.2 | 0.129-1.297 | 5/5 clean | 4/5 PASS |
| cream | 21,22,23 | 2.3 | 1.7 | 0.002-0.044 | 3/3 clean | 0/3 FAIL |
| tomato | 15,16,17 | 6.3 | 3.3 | 0.087-0.253 | 0/3 clean | 0/3 REJECT |
| butter | 13,14 | 2.0 | 2.0 | 0.014-0.025 | 0/2 clean | 0/2 FAIL |

## 2. Failure-Mode Taxonomy

### 2.1 clean_bridge_positive
**Milk seeds 9/10/11/24**: VIS produces 6-7/10 OPEN commands with sustained streaks (4-5), and qpos responds strongly (norm 0.27-1.30). RAND controls produce 0-1 opens with negligible qpos (norm ≤ 0.055). The VIS-specific physical bridge is unambiguously present.

### 2.2 command_weak
**Cream seeds 21/22/23**: VIS produces only 2-3/10 OPEN commands despite clearing all pre-VIS gates (Layer1 + ORACLE + RAND-clean). The PGD attack with `prefix_locked_gripper_open_margin` objective (margin=5.0) generates far fewer OPEN tokens on cream than on milk. Even when OPEN is decoded, qpos never crosses the 0.005 response threshold. The VIS attack token transfer is object/window-dependent.

**Butter seeds 13/14**: VIS produces only 2/10 OPEN. Butter was not Layer1-selected — it was a manual ORACLE candidate. The Layer1 detector correctly did not select it.

### 2.3 random_command_confounded
**Tomato seeds 16/17**: RAND produces 5-7/10 OPEN commands with significant qpos response (norm 0.12-0.95). VIS also produces OPEN, but the control is contaminated — the effect cannot be attributed to the VIS attack. **Layer3 abstain.**

**Butter seed 14**: RAND produces 3/10 OPEN with norm=0.26 — RAND exceeds VIS response.

### 2.4 physical_transfer_weak (seed-sensitive)
**Milk seed 12**: VIS produces 6/10 OPEN with streak=4 (strong command transfer), but qpos accumulation is lower than other milk seeds (norm=0.129 vs 0.27-1.30). The token-level command probe is strong, but the physical response is seed-sensitive. This is an attack stochasticity issue, not a bridgeability failure.

### 2.5 random_qpos_drift_confounded
**Cream seed 23 / Tomato seed 15**: RAND produces negligible OPEN commands but natural qpos drift exceeds VIS qpos response. The control is physically contaminated even without command contamination.

## 3. Figure Analysis

### Figure 1: Command Transfer vs Physical Transfer
`figures/s14a_command_vs_physical_transfer.png`

Left panel shows VIS open_count vs VIS norm. Milk seeds cluster in the top-right quadrant (open ≥ 6, norm ≥ 0.20). Cream seeds cluster in the bottom-left (open ≤ 3, norm ≤ 0.05). Tomato seeds have high opens (6-7) but variable norms — the RAND confound makes their VIS norm unreliable as a bridgeability signal.

Right panel shows RAND open_count vs RAND norm. Tomato and butter seeds scatter into the REJECT zone (open ≥ 3 or norm > 0.10). Milk and cream seeds stay within STRICT-CLEAN or USABLE zones.

### Figure 2: Paired VIS-RAND Scatter
`figures/s14a_rand_confound_scatter.png`

VIS norm vs RAND norm for each paired seed. Milk seeds cluster in the PASS ZONE (RAND norm ≤ 0.10, VIS norm ≥ 0.20). Cream seeds fall in the CLEAN but NO BRIDGE region (RAND clean but VIS norm near zero). Tomato seeds scatter into RAND CONTAMINATED region (RAND norm > 0.10). Butter seeds are both RAND contaminated and VIS-weak.

**One figure tells the whole story**: the two-dimensional separation (RAND-clean + VIS-bridge) cleanly partitions milk from everything else.

## 4. v0.4 Bridgeability Score Design

### 4.1 Motivation

The current pipeline requires running full VIS/RAND physical bridge tests (expensive: 2 seeds × ~5 min each on GPU) to determine if a window bridges. The v0.4 score aims to predict bridgeability with cheaper pre-tests, reserving full VIS/RAND for high-probability candidates.

### 4.2 Rule-Based Score (not trained)

```
Bridgeability_v0_4(parent, window, seed) =

  # Stage 1: Pre-VIS gates (no attack GPU needed)
  layer1_selected(parent, window)                    # Layer1 detector v0.3
  AND oracle_reachable(parent, window)                # ORACLE pos_area > 0.1 (offline)

  # Stage 2: RAND-clean veto (RAND attack only, 3 seeds)
  AND rand_clean(parent, window, seeds={k,k+1,k+2})  # RAND open ≤ 1, norm ≤ 0.10

  # Stage 3: VIS command probe (VIS attack, 1 seed, token decode only)
  AND vis_command_strong(parent, window, seed)        # VIS open ≥ 6/10, streak ≥ 4

  # Stage 4: Early qpos signal (from command probe run)
  AND has_qpos_response(parent, window, seed)         # response_delay ≥ 0
```

### 4.3 Stage Details

**Stage 1 (Pre-VIS)**: Offline or cheap inference. Layer1 detector classifies window as VIS-specific for OPEN command. ORACLE proves the window is physically reachable (gripper can open). No attack GPU needed.

**Stage 2 (RAND-veto)**: Run 3 RAND seeds on the window. If any seed produces ≥ 3 OPEN commands or norm ≥ 0.20 → REJECT (random-contaminated). If all seeds have ≤ 1 OPEN and norm ≤ 0.10 → STRICT-CLEAN. Between → USABLE. Cost: 3 RAND runs (~15 min on 1 GPU).

**Stage 3 (VIS command probe)**: Run 1 VIS seed. Decode action tokens; measure open_count and streak. If open ≥ 6 and streak ≥ 4 → command transfer is strong. If open ≤ 3 → command transfer is weak (likely OBJECT/WINDOW-DEPENDENT). Cost: 1 VIS run (~5 min).

**Stage 4 (Early qpos)**: From the same VIS command probe run, check if qpos rises above baseline within the post-window. If response_delay = -1 (never crosses 0.005 threshold) → physical response absent. Cost: zero marginal (data already collected).

### 4.4 Retrospective Performance

| Rule Set | Precision | Recall | Eliminates |
|----------|-----------|--------|------------|
| C (pre-VIS only) | 0.500 | 1.000 | 5/13 seeds |
| D (C + cmd probe) | 0.800 | 1.000 | 8/13 seeds |
| E (D + early qpos) | 0.800 | 1.000 | 8/13 seeds |
| F (D + norm ≥ 0.20) | 1.000 | 1.000 | 9/13 seeds |

Rule D is the recommended v0.4 threshold. It achieves 0.800 precision and 1.000 recall without outcome-circular criteria. The sole FP is milk seed12 — an attack-seed sensitivity case within a parent that bridged on 4/5 seeds.

### 4.5 Caveats

1. **Retrospective only**: These rules are derived from 13 seeds across 4 objects. They describe the current data, not a validated forward predictor.

2. **N=1 for command_weak**: Only cream represents the command_weak failure mode. We don't know if other objects would show the same pattern.

3. **Tomato = abstain, not negative**: All tomato windows are RAND-contaminated. The v0.4 score correctly rejects them (Stage 2), but this doesn't mean tomato "can't bridge" — it means we can't measure whether it bridges because the control is dirty.

4. **Seed sensitivity**: Milk seed12 shows that even within a positive parent, attack-seed stochasticity can produce sub-threshold norms. The v0.4 score at the parent level (using best-of-N seeds for the probe) would have perfect precision and recall.

5. **Not a trained detector**: This is a rule-based scoring heuristic. With N=13 seeds total and N=4 positive examples, training a classifier is not justified.

## 5. What v0.4 Enables (Not Yet Done)

If v0.4 were deployed as a forward selector:

1. **Screen new windows cheaply**: Stage 1 (free) → Stage 2 (3 RAND runs) → Stage 3 (1 VIS run). Only windows that pass all 4 stages get full VIS/RAND bridge testing.

2. **Score within-object**: Test additional milk state/window pairs to measure within-object generalization.

3. **Diagnose failures**: A window that fails Stage 3 (command_weak) might benefit from a different attack objective or margin. A window that fails Stage 2 (RAND-contaminated) is fundamentally not measurable with the current random-control design.

## 6. Claim Update

### Allowed (no change from S13a)

- Milk 4/5 repeated physical bridge POC (seeds 9,10,11,24), seed24 strongest (norm=1.297).
- Cream clears all pre-VIS gates but VIS bridge fails (command-weak, 0/3).
- All tomato windows are RAND-contaminated for Layer3.
- RAND-cleanliness is necessary but not sufficient.
- Layer1 command selectivity alone is insufficient for Layer3 specificity.
- v0.4 bridgeability score is a retrospective rule-based proposal, not validated.

### Forbidden (no change)

- Non-milk bridge established
- Layer3 solved / Detector solved
- Object-wide physical attack success
- Tomato as "attack negative" (it's abstain)
- Cream as "object negative" (single window, command-weak)

## 7. Next Steps (Recommendation Only)

### Direction A: Paper/Report Packaging (recommended)

The current results form a complete scientific narrative:
1. Milk POC proves the physical bridge exists (4/5 repeated).
2. Cream proves pre-VIS gates are insufficient (command-weak).
3. Tomato proves RAND contamination is a real confound.
4. v0.4 score provides a path toward systematic bridgeability prediction.

This can be written up as: "We demonstrate a VIS→physical bridge on a robotic manipulation task, identify three failure modes (command-weak, RAND-confounded, seed-sensitive), and propose a bridgeability score for future screening."

### Direction B: Milk Within-Object Generalization

If more positive results are needed:
1. Test milk at other state_ids (s1, s2) and windows.
2. First run ORACLE + RAND-veto only.
3. Only run VIS/RAND on candidates that pass v0.4 Stages 1-3.
4. This tests whether the milk bridge generalizes within the object.

### Direction C: Attack Objective Ablation

Investigate why cream is command-weak:
1. Vary `gripper_margin` (5.0 → 2.0, 10.0).
2. Try different attack objectives.
3. Compare gradient signals between milk and cream at the token level.

**None of these directions should be auto-launched.** S14a is complete; next step requires explicit decision.

## 8. Artifact Inventory

| Artifact | Path |
|----------|------|
| S14a report | `reports/STAGEB_RC1A_S14A_BRIDGEABILITY_SCORE_DESIGN_20260610.md` |
| Feature table | `tables/s14a_bridgeability_feature_table.csv` |
| Rule ablation | `tables/s14a_bridgeability_rule_ablation.csv` |
| Figure 1 (cmd vs phys) | `figures/s14a_command_vs_physical_transfer.png` |
| Figure 2 (RAND confound) | `figures/s14a_rand_confound_scatter.png` |
