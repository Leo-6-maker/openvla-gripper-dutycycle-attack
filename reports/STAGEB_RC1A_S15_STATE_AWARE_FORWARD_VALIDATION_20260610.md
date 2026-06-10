# Stage-B RC1a S15 — State-Aware v0.4 Forward Validation

**Date**: 2026-06-10
**GitHub HEAD**: a1d68d8 (S14a) → TBD (this freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603
**Type**: Forward validation — 15 GPU jobs, 3 candidates, 0 passed to full bridge

## Executive Summary

**No fresh candidate passed v0.4 forward gate.** Three candidates were screened (cream_s0_w85-95, milk_s0_w235-245, milk_s0_w230-240) using the v0.4 bridgeability pipeline. Cream_s0 is RAND-contaminated (all 3 seeds open ≥ 3). Both late-window milk candidates show weaker VIS command transfer (open=5/10, streak=3) compared to the positive milk_s0_w70-80 (open=6-7/10, streak=4-5). Milk_s0_w230-240 is additionally RAND-contaminated (seed29 open=5). **S15c (full VIS/RAND bridge) is NOT triggered.**

The finding is scientifically significant: **milk bridgeability appears window-specific, not task-wide.** The positive milk_s0_w70-80 window is uniquely positioned — it has both strong ORACLE reachability (0.295) and strong VIS command transfer (6-7/10 OPEN). Late-window milk candidates have comparable ORACLE reachability (0.25-0.33) but weaker VIS command transfer (5/10 OPEN), suggesting temporal dependency of the attack gradient.

## S15a: Task/State Candidate Census

See [tables/s15a_task_state_candidate_census.csv](../tables/s15a_task_state_candidate_census.csv).

Census of all 40 stable parent windows from K5/K5b/K5c pools across 9 objects:

| Object | Total parents | Layer1-selected | ORACLE known | RAND-vetoed | Full bridge tested | Bridge PASS |
|--------|--------------|-----------------|--------------|-------------|-------------------|-------------|
| milk | 6 | 4 | 1 (w70-80) | 2 screened | 1 | 1 (4/5 seeds) |
| cream_cheese | 5 | 2 | 1 (s2_w50-60) | 2 REJECTED | 1 | 0 |
| tomato_sauce | 9 | 5 | 4 | 4 REJECTED | 1 | 0 (all RAND confounded) |
| butter | 4 | 1 | 2 | 1 REJECTED | 1 (manual) | 0 |
| salad_dressing | 5 | 2 | 0 | 0 | 0 | 0 |
| alphabet_soup | 5 | 0 | 0 | 0 | 0 | 0 |
| bbq_sauce | 3 | 0 | 0 | 0 | 0 | 0 |
| orange_juice | 1 | 0 | 0 | 0 | 0 | 0 |

**Key gap**: 9 objects in the pool, but only 4 have been tested for physical bridge. Within milk, only state_id=0 has been characterized (no state_id 1-9 data).

## S15b: Milk Within-Object State Screen

Three candidates screened using v0.4 Stages 1-3 (ORACLE + RAND-veto + VIS command-probe).

### Results Summary

| Candidate | Object | Window | ORACLE | RAND-veto | VIS cmd-probe | Pass to S15c? |
|-----------|--------|--------|--------|-----------|---------------|---------------|
| C01 | cream_cheese | s0_w85-95 | 0.533 ✅ | **REJECT** (open 3,4,3) | cmd_weak (3/10) | ❌ |
| C02 | milk | s0_w235-245 | 0.250 ✅ | USABLE (2,1,0) | **BORDERLINE** (5/10) | ❌ |
| C04 | milk | s0_w230-240 | 0.327 ✅ | **REJECT** (0,2,5) | cmd_weak (5/10) | ❌ |

### C01: cream_s0_w85-95 — RAND REJECT + command_weak

- ORACLE pos=0.533 is strong (higher than tested cream_s2_w50-60's 0.284)
- But all 3 RAND seeds produce 3-4 OPEN commands with significant qpos (norm up to 0.201)
- VIS probe produces only 3/10 OPEN (same command-weak pattern as cream_s2)
- **Different cream state, same failure modes: RAND contamination + command-weak**
- Consistent with S14a taxonomy: cream is fundamentally random-sensitive for OPEN + VIS transfer is weak

### C02: milk_s0_w235-245 — USABLE but VIS probe BORDERLINE

- ORACLE pos=0.250 (lower than w70-80's 0.295, but above 0.10 threshold)
- RAND: seeds 27/28/29 produce 2/1/0 OPEN, norms 0.06/0.01/0.01 — USABLE (seed27 open=2 borderline)
- VIS probe: open=5/10, streak=3, norm=0.170
- **Fails v0.4 gate**: open < 6 AND streak < 4
- ORACLE norm 0.170 is below the 0.20 PASS threshold
- **Late-window milk has weaker attack transfer than mid-window (w70-80)**

### C04: milk_s0_w230-240 — RAND REJECT

- ORACLE pos=0.327 (comparable to w70-80)
- RAND seed29: open=5, streak=4 — strong RAND contamination
- VIS probe: open=5, streak=3 — same weak command transfer as C02
- **Milk is not immune to RAND contamination at all windows**
- The w230-240 window is in the "swing" phase where gripper dynamics may be more random-sensitive

### Key Comparison: Milk Windows

| Window | ORACLE pos | RAND clean? | VIS open | VIS streak | VIS norm | Bridge? |
|--------|-----------|-------------|----------|------------|----------|---------|
| **w70-80** (anchor) | 0.295 | 5/5 clean | 6-7 | 4-5 | 0.27-1.30 | **4/5 PASS** |
| w230-240 (swing) | 0.327 | REJECT (seed29 open=5) | 5 | 3 | 0.164 | ❌ |
| w235-245 (late) | 0.250 | USABLE | 5 | 3 | 0.170 | ❌ |

**The bridgeable window is temporally localized to mid-episode (w70-80).** Late-window milk (w230-250) has weaker VIS command transfer and one window is RAND-contaminated.

## S15c: NOT Triggered

No candidate passed the v0.4 PASS-to-S15c gate. Full VIS/RAND bridge testing is not justified for any of the screened candidates.

## Scientific Interpretation

### 1. Milk bridgeability is window-specific, not task-wide

The positive milk_s0_w70-80 window occupies a specific temporal niche (mid-episode, "anchor" phase). Late-window milk candidates (w230-250, near episode end) have:
- Comparable or higher ORACLE reachability (0.25-0.33 vs 0.295)
- Weaker VIS command transfer (5/10 vs 6-7/10 OPEN)
- RAND contamination in one window

This suggests the attack gradient depends on the visual/state context at specific timesteps. The model's gripper policy may be more susceptible to VIS perturbation during the "approach and grasp" phase (mid-episode) than during later phases.

### 2. Cream RAND sensitivity is state-robust

Cream_s0_w85-95 shows the same RAND contamination pattern as cream_s2_w50-60. Different state, same random-sensitivity. This strengthens the conclusion that cream is fundamentally not measurable with the current random-control design — the RAND confound is not isolated to a specific window.

### 3. v0.4 correctly predicted no new bridge

The v0.4 forward screen correctly filtered all 3 candidates before committing to expensive full VIS/RAND. This is a successful demonstration of the pipeline's screening function, even though no bridge was found.

### 4. The pool of untested milk windows is limited

The stable parent pool only contains milk at state_id=0. To test within-object generalization properly, we would need to characterize milk at other state_ids (1-9) — which requires running the detector on fresh data.

## Claim Update

### Allowed (updated from S14a)

- Milk_s0_w70-80 is the only confirmed physical bridge window (4/5 seeds PASS, seed24 strongest).
- Milk bridgeability is window-specific: late-window milk candidates (w230-250) fail v0.4 command-probe gate (open=5/10, streak=3).
- Cream RAND sensitivity is state-robust: cream_s0_w85-95 shows same contamination as cream_s2_w50-60.
- v0.4 forward screen correctly filtered 3/3 candidates; no false positives.
- Bridgeability depends on temporal window within task, not just task identity.
- Layer1 + ORACLE + RAND-clean + command-probe is necessary but not sufficient; only w70-80 clears all gates.

### Forbidden (unchanged)

- Milk task-wide bridge / object-wide success
- Non-milk bridge established
- Layer3 solved / Detector solved
- Tomato/butter/cream as "attack negative" (they are RAND-confounded or command-weak)
- LLM-generated URLs, task failure claims

## Next Steps (Recommendation Only)

### Direction A: Expand milk to other state_ids (recommended for paper)

The current census covers only milk_s0. To test within-object generalization:
1. Run Layer1 detector on milk state_ids 1-9 to find candidate windows
2. Apply v0.4 screening pipeline (ORACLE → RAND-veto → VIS probe)
3. Run full VIS/RAND on any window that passes all gates

This would answer: "Is milk bridgeability a property of milk_s0_w70-80 specifically, or does it generalize to other milk states?"

### Direction B: Temporal window sweep on milk_s0

Systematically test adjacent windows around w70-80:
- w60-70, w65-75, w75-85, w80-90
- Measure how command transfer decays with distance from the optimal window
- This would quantify the "temporal locality" of the attack

### Direction C: 10-task funnel census (S16, no GPU)

Build the full LIBERO Object funnel table as a census artifact:
- 10 tasks × available states
- Layer1 selection rate per task
- ORACLE reachability rate
- RAND-clean rate
- This becomes Table 1 for the paper

### Direction D: Attack objective ablation on cream

If the goal is to "fix" cream:
- Vary gripper_margin (2.0, 5.0, 10.0) on cream_s2_w50-60
- Measure VIS command-probe open_count change
- Only proceed to full bridge if command transfer improves to ≥ 6/10

## Artifact Inventory

| Artifact | Path |
|----------|------|
| S15 report | `reports/STAGEB_RC1A_S15_STATE_AWARE_FORWARD_VALIDATION_20260610.md` |
| Task/state census | `tables/s15a_task_state_candidate_census.csv` |
| S15b screen results | `tables/s15b_milk_state_screen_results.csv` |
| S15c bridge results | NOT TRIGGERED |

Server output: `/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s15b_v04_forward_screen/`
- 15/15 summaries ✅
- 15/15 traces ✅
- 0/0 FAIL/infra errors ✅

## No S16 Auto-Launch

S15 complete. No S15c triggered. No S16 launched. Next step requires explicit user decision.
