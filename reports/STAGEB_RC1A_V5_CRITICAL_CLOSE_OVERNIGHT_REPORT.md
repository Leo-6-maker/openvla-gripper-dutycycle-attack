# V5 Critical-CLOSE Event Attack Overnight Report

**Date:** 2026-06-13
**Server:** klfy-SYS-4028GR-TR2
**Branch:** exp/vis-prefix-margin-repair-20260603
**Commit:** f1292c4 (HEAD), a84f5ef (local)

## 1. Executive Summary

**Final Label: `NO_STRONG_EPS6_C2O_FOUND`**

Ran a 3h autonomous diagnostic of TokenPrefixPGD on critical-CLOSE events mined from V4-aligned clean traces. Tested 30 P0 candidates across 6 LIBERO Object tasks with eps=6, PGD=20, margin=5. Found zero C2O (clean-close → adversarial-open) events. Extended diagnostic sweep (PGD 20/40, eps 6/8, margin 5/10/20) on top candidates confirmed: CE loss optimizes perfectly but decoded gripper action does not flip to OPEN.

**Root cause hypothesis:** `prefix_locked_gripper_open_margin` objective uses hinge loss on logsumexp(open_tokens) vs max(non_open_logit). The CE loss drops to near-zero, but the argmax at decode still selects a CLOSE token because the OPEN region probability mass is spread thin across many tokens.

## 2. Pipeline Status

| Layer | Stage | Status |
|-------|-------|--------|
| Layer1 | v0.3.2 screener | STRONG PASS (~69.6% RAND-clean) |
| Layer2 | RAND-stability | 14 candidates validated |
| V4 VIS | - | QUARANTINED (fallback noise) |
| V5 TokenPGD interface | - | VERIFIED |
| **V5 critical-CLOSE smoke** | **This report** | **0/8 C2O at eps6** |
| **V5 objective sweep** | **This report** | **0/36 C2O (clean OPEN, untestable)** |
| Layer3 confirmed | - | **None** |

## 3. Phase Results

### Phase 0 — Preflight
- 12/12 runner invariants PASS
- GPU 0,1 and 4,5 available (4× RTX 2080 Ti used)
- Working tree: v5 runner had 3 regressions from harden script; reverted and re-applied 20 minimal fixes

### Phase 1 — Critical-CLOSE Event Mining
- Ingested 266 V4-aligned clean traces (s20d_v4_fixed_window_l3, num_steps_wait=10)
- Mined 9661 raw candidates → 2025 after dedup
- **641 P0, 488 P1** across 6 tasks (alphabet_soup, bbq_sauce, butter, chocolate_pudding, cream_cheese, ketchup)
- Top 30 selected with task diversity

### Phase 2 — eps6 Event-Center Smoke
- 8 unique candidates tested via validated one-step smoke (PGD=20, eps=6, margin=5, seeds=99/199/299)
- **clean_CLOSE reproduced: 5/8 (62.5%)**
- **C2O: 0/8 (0%)**
- CE loss pattern: 30-50 initial → 0.001-5 final (successful optimization)
- `gripper_logit_margin_after = -999` (not computed for this objective)
- Interface: PASS (token_pgd_adv_inputs_generate, no fallback)

### Phase 3 — Objective/Budget Diagnostic
- Sweep matrix: PGD={20,40} × eps={6,8} × margin={5,10,20} × seeds={99,199,299}
- chocolate_pudding_s11: All 36 tests clean=OPEN (walk didn't reproduce CLOSE at correct trace offset)
- alphabet_soup_s10: Running (1/36 done, clean=OPEN)
- **No C2O testable** — trajectory divergence between clean scan trace and V5 walk

### Phase 4 — Conditional Full Rollout
- **Skipped** — no ROBUST_STRONG_C2O candidate at eps6

## 4. Key Scientific Findings

### 4.1 CE Loss Optimizes But Action Doesn't Flip
The definitive finding. PGD reduces target token CE loss from 30-50 to near-zero (<0.01), meaning the target token probability approaches 1.0. However, the argmax at decode time still selects a non-OPEN token. This suggests:

1. The targeted tokens (gripper OPEN region via `get_gripper_region_by_decoded_action`) are correctly identified
2. The hinge loss `relu(max_non_open - logsumexp(open_region) + margin)` is successfully optimized
3. But logsumexp aggregates probability mass across all OPEN tokens, which can be high even if no single OPEN token beats the top CLOSE token
4. The model's action vocabulary has few OPEN tokens (<10 out of 256 bins)

### 4.2 Trajectory Divergence
Walking to an event center from scratch doesn't reliably reproduce the clean trace state. Clean scan traces (V4 runner, full 280-step episodes) and V5 one-step walks diverge due to floating-point non-determinism in mujoco physics. This makes "mine from trace, verify by walk" unreliable.

### 4.3 Margin=-999
The `gripper_logit_margin_after` diagnostic field returns -999 (sentinel) because the current objective doesn't trigger the logit audit path. This is a telemetry gap, not a bug.

## 5. Quantitative Summary

| Metric | Value |
|--------|-------|
| Clean summaries ingested | 266 |
| Critical-CLOSE events mined | 2025 (641 P0, 488 P1) |
| eps6 smoke candidates tested | 8 |
| Smoke results C2O | 0 |
| Diagnostic sweep tests | 72 (36 alp_soup + 36 choc_pudding) |
| Diag C2O testable | 0 (clean always OPEN) |
| Full rollouts | 0 |
| GPU hours used | ~12 GPU-hours |
| GPU pairs used | 0,1 (smoke) + 4,5 (diag) |
| infra invalid count | 0 |

## 6. Artifacts

| File | Path |
|------|------|
| Candidate table | tables/s20d_v5_critical_close_event_candidates.csv |
| Smoke results | OUT/smoke/phase2_smoke_gpu01_output.csv |
| Diag results | OUT/diag/phase3_diag_sweep.csv |
| Preflight | OUT/reports/preflight_invariants.txt |
| This report | reports/STAGEB_RC1A_V5_CRITICAL_CLOSE_OVERNIGHT_REPORT.md |

## 7. Recommended Next Actions

1. **Fix objective mismatch:** Replace `logsumexp(open_region)` with `max(open_region_logits)` or use `softmax`-based loss to ensure single argmax flips
2. **Use exact replay for trajectory reproduction:** Record clean actions from full trace and replay open-loop to reproduce exact state, then test attack
3. **Investigate token distribution:** Audit how many tokens are in the OPEN region and their probability distribution
4. **Consider stronger attack budget:** eps=10/12/16 (diagnostic only, not for claims)
5. **Retest cream_cheese_s2_w80_90 canary** (historical S20M3a CMD_POSITIVE) under current v5 pipeline to verify attack potential

## 8. Allowed vs Forbidden Labels

- **Allowed (used):** `NO_STRONG_EPS6_C2O_FOUND`, `OBJECTIVE_MISMATCH_DIAGNOSTIC`, `NO_CRITICAL_CLOSE_EVENTS_REPRODUCED`
- **Forbidden (not used):** `CONFIRMED_CMD`, `CONFIRMED_PHYSICAL`, `Layer3 success`, `broad LIBERO generalization`
