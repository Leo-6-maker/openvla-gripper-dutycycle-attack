# Stage-B RC1a S13a 3-Hour Auto-Progress — Layer3 Failure-Mode Audit + RAND-Veto + Milk Positive-Control

**Date**: 2026-06-10
**GitHub HEAD**: 264f056 (S12b freeze) → TBD (this freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

**S13a complete. Milk positive-control PASS (norm=1.297, strongest yet). Both remaining tomato RAND-veto candidates REJECT. All tomato windows across all tested parents are RAND-contaminated for Layer3 physical bridge. No new non-milk VIS candidate is justified.**

The failure-mode audit across 12 VIS/RAND pairs (milk, butter, tomato, cream) reveals a clear taxonomy: milk has both high command transfer (6-7/10 OPEN) and high physical transfer (qpos_open_area >> qpos_nonopen_area); cream is command-weak (2-3/10 OPEN) with zero physical transfer; tomato is random-command-confounded; butter is both command-weak and random-drift-confounded.

## Gate 0: Pre-Launch (PASS)

| Check | Result |
|-------|--------|
| git HEAD = 264f056 | PASS |
| Branch = exp/vis-prefix-margin-repair-20260603 | PASS |
| No uncommitted code changes to runner | PASS |
| GPU 0,1,2,4,5,6 all 0 MiB used | PASS |
| S12b artifacts preserved (6 summaries) | PASS |

## Gate 1: Milk Positive-Control Seed24 → PASS

| Metric | VIS | RAND | Gate |
|--------|-----|------|------|
| open_count | **6/10** | 0/10 | PASS |
| max_streak | **4** | 0 | PASS |
| pos_area | **0.382** | 0.000 | PASS |
| ORACLE norm | **1.297** | 0.000 | PASS |
| RAND open ≤ 1 | — | ✅ | PASS |
| RAND norm ≤ 0.10 | — | ✅ | PASS |
| VIS pos > RAND pos | ✅ | — | PASS |

**PASS with strongest norm to date (1.297).** VIS pos_area exceeds ORACLE reference (0.295), suggesting the current attack seed produces an even more effective gripper opening sequence than the forced-binary ORACLE baseline. Runner and attack confirmed operational.

## Gate 2: Remaining Tomato RAND-Veto → Both REJECT

### tomato_s0_w55-65 (ORACLE ref = 0.5039)

| Seed | open | streak | pos | norm | Verdict |
|------|------|--------|-----|------|---------|
| 24 | 1 | 1 | 0.009 | 0.018 | STRICT-CLEAN |
| 25 | **4** | 2 | 0.015 | 0.030 | **REJECT** |
| 26 | 0 | 0 | 0.008 | 0.016 | STRICT-CLEAN |

**REJECT — seed 25 open=4 ≥ 3.** Two of three seeds clean, but the threshold is per-parent.

### tomato_s2_w90-100 (ORACLE ref = 0.2686)

| Seed | open | streak | pos | norm | Verdict |
|------|------|--------|-----|------|---------|
| 24 | 2 | 1 | 0.024 | 0.089 | USABLE |
| 25 | 2 | 1 | 0.035 | 0.129 | MARGINAL |
| 26 | **3** | 1 | 0.010 | 0.037 | **REJECT** |

**REJECT — seed 26 open=3 ≥ 3.** Seed 25 also above USABLE norm threshold (0.129 > 0.10).

### Tomato Family Summary

All tested tomato parents are RAND-contaminated for Layer3:
- tomato_s2_w155-165: REJECT (S11b, RAND open 2-7/10)
- tomato_s2_w150-160: REJECT (S12a, RAND open 2-5/10)
- tomato_s0_w55-65: REJECT (S13a, RAND open 4/10)
- tomato_s2_w90-100: REJECT (S13a, RAND open 3/10)

**Tomato is fundamentally random-sensitive for the OPEN command in the Phase1-port runner.** The object identity (tomato_sauce) or its state dynamics make the model's gripper policy vulnerable to random pixel perturbations producing spurious OPEN tokens.

## Gate 3: Failure-Mode Audit

Full table: `tables/s13a_layer3_failure_mode_audit.csv`

### Failure Mode Taxonomy

| Mode | Definition | Examples |
|------|-----------|----------|
| `clean_bridge_positive` | VIS open ≥ 6, streak ≥ 4, norm ≥ 0.20, RAND clean | milk seeds 9,10,11,24 |
| `physical_transfer_weak` | VIS open ≥ 4 but norm < 0.20, RAND clean | milk seed 12 (norm 0.129) |
| `command_weak` | VIS open < 4/10, norm < 0.05 | cream seeds 21,22,23; butter 13,14 |
| `random_command_confounded` | RAND open ≥ 3/10 | tomato seeds 16,17; butter seed 14 |
| `random_qpos_drift_confounded` | RAND pos ≥ VIS pos or RAND norm ≥ 0.10 | cream seed 23 (RAND pos 0.024 > VIS 0.0005) |
| `mixed_seed_sensitive` | Same parent: some PASS, some FAIL | milk seed 12 (norm 0.129 vs others 0.27-1.30) |

### Why Milk Bridges — Quantitative

Milk positive seeds share:
1. **High command transfer**: 6-7/10 OPEN with streak ≥ 4 (sustained opens)
2. **Strong physical transfer**: qpos_open_area 0.031-0.086 dominates qpos_nonopen_area
3. **Fast response**: response_delay 4-8 steps after first OPEN
4. **RAND clean**: 0-1 RAND opens, norm ≤ 0.055

### Why Cream Fails

Cream clears all pre-VIS gates (Layer1 + ORACLE + RAND-clean) but:
1. **VIS command weak**: only 2-3/10 OPEN (vs milk 6-7)
2. **Physical transfer zero**: qpos_peak never exceeds 0.005 threshold (response_delay = -1)
3. **qpos_open_area negligible**: 0.000-0.004 (vs milk 0.031-0.086)
4. **RAND clean but irrelevant**: command cleanliness doesn't help if VIS can't transfer

The attack objective (`prefix_locked_gripper_open_margin`, margin=5.0) produces far fewer OPEN tokens on cream than on milk. The cream task description or visual features may weaken the token-level attack gradient.

### Why Tomato Is Not Positive

All tomato RAND-veto candidates fail the RAND-clean gate:
- RAND produces 2-7 OPEN commands with qpos response up to 0.590
- VIS command transfer exists (6-7/10) but cannot be distinguished from random perturbation effect
- **Layer3 abstain — not "tomato attack fails," but "tomato RAND confounded, cannot measure VIS-specific effect"**

### Why Butter Does Not Evaluate Detector v0.3

Butter was a manual ORACLE-referenced candidate, not Layer1-selected:
- Layer1 v0.3 did not select butter as VIS-specific
- VIS command weak (2/10) and RAND confounded (3/10 opens, pos up to 0.090)
- Not a detector failure — detector correctly did not select butter

## Gate 4: Recommendation (Case B)

```
milk seed24 PASS ✓
AND remaining tomato all REJECT ✓
→ 推荐停止 non-milk VIS 扩展
→ 转向 Layer3 v0.4 selector / bridgeability score
```

### Rationale

1. All S11a overlap Layer1+ORACLE candidates have been exhausted:
   - butter: REJECT (S12a RAND-veto)
   - cream: TESTED → VIS bridge FAIL (S12b, command-weak)
   - tomato_s2_w150-160: REJECT (S12a)
   - tomato_s2_w155-165: REJECT (S11b, RAND confounded)
   - tomato_s0_w55-65: REJECT (S13a)
   - tomato_s2_w90-100: REJECT (S13a)

2. No remaining candidate clears all three pre-VIS gates.

3. The failure-mode taxonomy provides the data needed for Layer3 v0.4: a "bridgeability score" that predicts whether a Layer1+ORACLE+RAND-clean window will actually bridge, factoring in command transfer strength and physical response.

### If Expansion Is Still Desired

The only justified path is:
- Return to milk and test additional state_ids or windows to establish within-object generalizability
- Or: design a v0.4 selector that scores bridgeability (not just command OPEN specificity)

### Forbidden

- New non-milk VIS without a bridgeability selector
- S14 full-queue expansion
- Claiming tomato as negative (it's RAND-confounded → abstain)
- Claiming cream as negative (it's command-weak → not a general claim about the object)
- Saying "Layer3 solved" or "detector solved"

## Artifact Inventory

| Artifact | Path |
|----------|------|
| S13a report | `reports/STAGEB_RC1A_S13A_THREE_HOUR_AUTOPROGRESS_20260610.md` |
| Milk positive-control | `tables/s13a_milk_positive_control_seed24.csv` |
| Tomato RAND-veto | `tables/s13a_remaining_tomato_rand_veto.csv` |
| Failure-mode audit | `tables/s13a_layer3_failure_mode_audit.csv` |
| Layer3 aggregate | `tables/layer3_physical_bridge_status_all.csv` |

Server output: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s13a_rand_veto_and_milk_positive/`
- 8/8 summaries ✅
- 8/8 traces ✅
- 0/0 FAIL/infra errors ✅

## Current Allowed Claim

**Allowed:**
- Milk remains the only clean repeated physical bridge POC, now 4/5 (seeds 9,10,11,24).
- Milk seed24 is the strongest positive yet (ORACLE norm=1.297).
- Cream clears all pre-VIS gates but VIS bridge fails 0/3 (command-weak).
- All tomato windows are RAND-confounded for Layer3 physical bridge.
- RAND-cleanliness is necessary but not sufficient for Layer3.
- Layer1 command selectivity alone is insufficient for Layer3 physical bridge specificity.

**Forbidden:**
- Non-milk bridge established
- Layer3 solved / Detector solved
- Object-wide physical attack success
- Tomato as "attack negative" (it's RAND-confounded → abstain)
- Cream as "object negative" (it's command-weak on this window only)

## No S14 Auto-Launch

S13a completed within 3-hour window. No S14 or new non-milk VIS was launched. Next step requires explicit user decision.
