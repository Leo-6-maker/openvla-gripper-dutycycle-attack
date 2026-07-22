# Exact-W32 Calibration Diagnosis — 2026-07-22 (CORRECTED)

**Status:** HEAD_LEVEL_POLICY_REPLAY — L1/L2 tested, L3/L4 awaiting Codex V5 scheduler

**PR:** https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/96

## 1. Erratum

See [EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md](EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md).

Initial replay RETRACTED due to manipulation known_mask violation (inflated any-head P50 from 0.004 to 0.516). Corrected replay now respects known_mask on all heads.

## 2. 12/12 Exact-W32 Splits: Complete and Sealed

| Split | Source Commit | Checkpoint Seal | Prediction Seal |
|-------|---------------|-----------------|-----------------|
| o0_i0 | `401f79a0` | `db6c8f3a` | `0250b035` |
| o0_i1 | `401f79a0` | `279600ca` | `bb34a3b5` |
| o0_i2 | `401f79a0` | `b5b94a4c` | `fc201d24` |
| o1_i0 | `401f79a0` | `d71233a0` | `5a07adbc` |
| o1_i1 | `401f79a0` | `1b0e37f1` | `7868d22c` |
| o1_i2 | `401f79a0` | `c6cc18b5` | `0c870dc5` |
| o2_i0 | `401f79a0` | `aa9a92cb` | `ce9035c4` |
| o2_i1 | `401f79a0` | `2f70567a` | `1630a23e` |
| o2_i2 | `401f79a0` | `8b4cf834` | `1dbdeff9` |
| o3_i0 | `401f79a0` | `90bc10b2` | `5f8aea6e` |
| o3_i1 | `401f79a0` | `405a91e3` | `475dc9ea` |
| o3_i2 | `401f79a0` | `361c1b73` | `e9a4686b` |

All 12 splits: train→predict→evaluate→audit PASS. Sidecar status: PASS.
Analysis script SHA256: `847b59c71403e99756ea7adc6993d1b1a4267b660724cb1586bfe7c718f46a63`

## 3. 4-Layer Safety Metric Framework

**CRITICAL: The four layers must NOT be conflated.**

| Layer | Metric | Definition | Tested? |
|-------|--------|------------|---------|
| L1 | Per-step background emit | Fraction of BG steps with any known head ≥ threshold | YES (evaluator) |
| L2 | Episode any-head emit | Fraction of episodes with ≥1 BG step emitting | YES (replay) |
| L3 | Actual scheduler false start | Episodes where real V5 scheduler triggers false candidate | UNTESTED |
| L4 | Actual attack actuation | Episodes where false candidate → attack command | UNTESTED |

**The original Stage-1 gate (background_false_emit_rate ≤ 0.10) is L1 only.**

The current head-level replay additionally computes L2 (episode false starts), which is a stricter metric. L3 and L4 require the Codex V5 scheduler contract.

### L1 Results (Raw, from evaluator)

| Split | L1 BG Emit | Original Gate |
|-------|-----------|---------------|
| o1_i1 | 0.080 | PASS |
| o1_i2 | 0.093 | PASS |
| o1_i0 | 0.109 | FAIL |
| o2_i1 | 0.114 | FAIL |
| o2_i2 | 0.127 | FAIL |
| o3_i1 | 0.128 | FAIL |
| o0_i0 | 0.142 | FAIL |
| o3_i2 | 0.152 | FAIL |
| o0_i1 | 0.166 | FAIL |
| o2_i0 | 0.174 | FAIL |
| o0_i2 | 0.174 | FAIL |
| o3_i0 | 0.090 | PASS |

**L1 raw: 3/12 pass, 9/12 fail.** Worst-case: o0_i2/o2_i0 at 0.174.

### L2 Results (Raw, head-level policy, tau=0.5, k=1)

**0/12 feasible** under constraints: L2 (episode false_rate) ≤ 0.10 AND release event recall ≥ 0.50.

This is the source of the preliminary CASE_4. However, L2 ≠ L3 ≠ L4.

## 4. Platt Calibration Results (Train-Only, Per-Split)

Method: grid search for (a, b) in p = σ(a·z + b), fitted on first 50% episodes, evaluated on second 50%.

### L1 (Per-Step) After Platt

| Split | L1 Raw | L1 Platt | Δ | Pass L1? |
|-------|--------|----------|---|----------|
| o3_i2 | 0.152 | **0.056** | -0.096 | YES |
| o1_i0 | 0.109 | **0.078** | -0.031 | YES |
| o2_i2 | 0.127 | **0.079** | -0.048 | YES |
| o0_i2 | 0.174 | 0.103 | -0.071 | near |
| o3_i1 | 0.128 | 0.104 | -0.024 | near |
| o0_i0 | 0.142 | 0.106 | -0.036 | near |
| o1_i1 | 0.080 | 0.119 | +0.039 | FAIL |
| o2_i1 | 0.114 | 0.121 | +0.007 | FAIL |
| o2_i0 | 0.174 | 0.131 | -0.043 | FAIL |
| o3_i0 | 0.090 | 0.140 | +0.050 | FAIL |
| o1_i2 | 0.093 | 0.181 | +0.088 | FAIL |
| o0_i1 | 0.166 | 0.182 | +0.016 | FAIL |

**L1 Platt: 3/12 pass, 3 near-miss (0.10-0.11).**

The Platt calibration is not universally beneficial — for splits that were already good (o1_i1, o1_i2, o3_i0), it slightly worsens L1. This is expected: a single (a,b) fitted on half the held-out data doesn't necessarily improve the other half.

### Release Recall After Platt (tau=0.3, step-level)

All 12 splits: **0.94-0.99** — excellent.

### L2 (Episode) After Platt

| Split | L2 Platt Grasp |
|-------|---------------|
| o3_i2 | **0.544** |
| o1_i0 | 0.756 |
| o2_i2 | 0.822 |
| ... (all others > 0.80) | |

L2 remains elevated. But L2 measures "at least one BG step in the episode has grasp≥τ" — this is NOT a false attack start (L3) or false attack actuation (L4).

## 5. Class Weight Audit

**Verdict: POS_WEIGHT_SHIFT_REJECTED_AS_PRIMARY_CAUSE**

Max cross-split log(pos_weight) range = 0.092. All head/route combinations < 0.10. Analytic deweighting does not change feasibility conclusions.

## 6. Identity Composition

**Assessment: IDENTITY_EFFECT_CONFOUNDED_WITH_CHECKPOINT_AND_SPLIT**

12 splits = 4 outer folds × 3 inner folds. Each split has a different checkpoint AND different held-out identities. Cannot cleanly separate identity effect from checkpoint/fold effect without cross-prediction.

## 7. Long Streak Audit

Top-5 longest streaks (268, 260, 184, 183, 175 steps) from `libero_10/task_01-02` and `libero_goal/task_03`. All grasp-head dominant at tau=0.5. **Awaiting semantic audit** to determine whether these are pre-grasp anticipation, post-release artifacts, or genuine false signals.

## 8. Corrected Root Cause Assessment

### What IS established:

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Exact-W32 release AUPRC stable (0.83-0.88) | 12/12 eval | HIGH |
| Raw L1 fails worst-split gate | 9/12 > 0.10 | HIGH |
| Raw head-level policy (L2) infeasible | 0/12 feasible | HIGH |
| pos_weight NOT primary cause | log-range < 0.10 | HIGH |
| Platt can reduce L1 for SOME splits | 3/12 pass, 3 near-miss | MEDIUM |
| Release recall after Platt is excellent | 0.94-0.99 @ tau=0.3 | HIGH |

### What is NOT established:

| Claim | Reason |
|-------|--------|
| V5 scheduler cannot resolve | L3/L4 UNTESTED |
| Platt calibration fails | Only basic grid search tested, not cross-validated |
| Identity composition is root cause | Confounded with checkpoint/fold |
| Model architecture must change | Representation viable, decision layer unresolved |

### Corrected Case:

```
CASE_4_CANDIDATE → HOLD_PENDING_ACTUAL_SCHEDULER_AND_PROPER_CALIBRATION
```

**DO NOT interpret as:**
- "Model objective/feature failure confirmed"
- "Exact-W32 structure must be replaced"
- "No calibration method can work"

**Correct interpretation:**
- Raw head-level threshold/persistence strategies fail at L2
- Platt calibration improves L1 for some splits
- Whether the actual V5 scheduler (dwell, 3-of-5, veto) reduces L2→L3/L4 sufficiently is UNTESTED

## 9. GO / HOLD Summary

| Item | Status |
|------|--------|
| 12-split confirmation | **DONE** (12/12 PASS) |
| Per-step (L1) analysis | **DONE** |
| Episode-level (L2) analysis | **DONE** |
| Platt calibration (basic) | **DONE** |
| L3/L4 (V5 scheduler) | **AWAITING CODEX** |
| Engineering Full-FIT | **HOLD** |
| Modify loss | **HOLD** |
| V3 loss experiments (design only) | **GO** |
| V3 loss experiments (training) | **HOLD** |
| Passive shadow | **HOLD** |
| Active smoke | **HOLD** |

## 10. Explicit Statement

- Student training launched? **NO**
- OpenVLA rollout launched? **NO**
- Attack launched? **NO**
- CAL/CHECK accessed? **NO**
- Old artifacts modified? **NO**

## 11. Output Files

**Committed:**
- `docs/reports/EXACT_W32_FSM_REPLAY_ERRATUM_20260722.md`
- `docs/reports/EXACT_W32_CALIBRATION_DIAGNOSIS_20260722.md`
- `analysis/student_trigger_calibration/metric_contract.json`
- `tests/detector_v5/test_known_mask_contract.py` (10/10 PASS)

**Server-side (OFFICIAL_V3_FACTORIZED_STUDENT_V2_RECOMMENDED_EXACT_W32_V1_20260721/analysis/student_trigger_calibration/):**
- `per_split_metrics.csv`
- `per_split_pareto.csv` (1500 rows)
- `common_operating_point_search.csv`
- `identity_failure_breakdown.csv` (2160 rows)
- `class_weight_shift_audit.json`
- `calibration_transfer_results.csv`
- `long_streak_case_audit.csv`
- `pareto_summary.json`
- `replay_contract.json`
- `platt_calibration_results.json`

## 12. Key Conclusion

> **Exact-W32的表征仍然有效（release AUPRC 0.83-0.88稳定）；失败的是当前raw head-level决策规则（L2: 0/12 feasible）。Platt校准可以改善L1 step-level安全（3/12 pass，3 near-miss）且保持release recall优秀（0.94-0.99）。是否需要修改训练目标，必须等真实V5 scheduler replay（L3/L4）和无泄漏交叉验证Platt校准完成后再决定。当前不应启动大规模V3重训。**
