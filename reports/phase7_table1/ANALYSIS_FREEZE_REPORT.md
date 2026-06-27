# Object Table 1 — Analysis Freeze Report

**Date**: 2026-06-27
**Status**: NAD + CQFR analysis products frozen
**Git HEAD**: `01d19779ef770135e1ad01fd8541e75e56181057`

---

## Frozen Artifacts

### NAD Analysis

| File | SHA256 |
|------|--------|
| `reports/phase7_table1/nad/NAD_RUN_LEVEL.csv` | `2a5b867c...` |
| `reports/phase7_table1/nad/NAD_CONDITION_SUMMARY.csv` | `72a3e293...` |
| `reports/phase7_table1/nad/NAD_AUDIT.json` | `9fe63e15...` |
| `reports/phase7_table1/nad/ARMLOCK_INVARIANT_FULL.csv` | `b0090640...` |
| Analysis script: `tmp/nad_aggregate.py` | `1148c7cb...` |

### CQFR Blind Package

| File | SHA256 |
|------|--------|
| `evidence/phase7_table1/cqfr_blind/B0001-B0055.mp4` (55 videos) | manifest: `402c5514...` |
| `evidence/phase7_table1/cqfr_blind/CQFR_REVIEWER_TEMPLATE.csv` | `50a3ecbf...` |
| `evidence/phase7_table1/cqfr_blind/CQFR_LABEL_DEFINITIONS.txt` | `8514573f...` |
| `evidence/phase7_table1/cqfr_blind/CQFR_BLIND_KEY_PRIVATE.csv` | `cdb5d822...` (PRIVATE) |
| Generation script: `tmp/generate_cqfr_package.py` | `8ef9688d...` |

---

## Table 1 — Panel B: Object Breadth (Frozen)

| Condition | ITT FR (N=24) | Conditional FR (N=21) | Coverage |
|-----------|:---:|:---:|:---:|
| RAND | 0/24 = 0.0% | 0/21 = 0.0% | 87.5% |
| TMA no-lock | 17/24 = 70.8% | 17/21 = 81.0% | 87.5% |
| TMA ArmLock | 18/24 = 75.0% | 18/21 = 85.7% | 87.5% |
| Prefix no-lock | 19/24 = 79.2% | 19/21 = 90.5% | 87.5% |
| Prefix ArmLock | 19/24 = 79.2% | 19/21 = 90.5% | 87.5% |

**Coverage note**: tomato_sauce_s1 exhibits systematic detector non-emission (0/15 across all conditions).

**Paired (conditional N=21)**:
- TMA: b=0, c=1, paired effect = +4.8% (salad_s1 seed 123: NL OK, AL FAIL)
- Prefix: b=1, c=1, paired effect = 0.0% (orange_s2 seeds 123/456 in opposite directions)

---

## Table 2 — NAD Mechanism (Frozen, metric refresh N=27/condition)

NAD definition: `NAD_i = |a^exec_i - a^clean_i| / action_range_i`
Action ranges from LIBERO model action_stats (Q01/Q99).

| Metric | TMA no-lock | TMA ArmLock | Prefix no-lock | Prefix ArmLock |
|--------|:----------:|:----------:|:--------------:|:--------------:|
| NAD_pol_arm (mean) | 0.1622 | 0.1720 | 0.1065 | 0.1063 |
| NAD_pol_arm (max) | 0.4986 | 0.5945 | 0.4909 | 0.4853 |
| NAD_pol_grip (mean) | 0.3913 | 0.3820 | 0.4043 | 0.3987 |
| NAD_exec_arm (mean) | 0.1622 | **0.0000** | 0.1065 | **0.0000** |
| NAD_exec_grip (mean) | 0.1697 | 0.1939 | 0.1622 | 0.1846 |
| attack_prep (ms) | 12277 | 11110 | 35399 | 39041 |
| total_step (ms) | 13687 | 12416 | 36558 | 40419 |

**ArmLock invariant**: 540 attack frames across 54 ArmLock runs, 0 violations (NAD_exec_arm = 0 to within 1e-9).

### Paired NAD Deltas (ArmLock - NoLock, N=27)

| Metric | TMA delta [95% CI] | Prefix delta [95% CI] |
|--------|:---|:---|
| NAD_exec_arm | **-0.162 [-0.203, -0.127]*** | **-0.106 [-0.131, -0.083]*** |
| NAD_exec_grip | +0.024 [-0.002, +0.048] n.s. | +0.022 [0.000, +0.045] n.s. |
| NAD_pol_arm | +0.010 [-0.012, +0.034] n.s. | -0.000 [-0.019, +0.018] n.s. |
| attack_prep_ms | -1168 [-2399, +40] n.s. | **+3642 [+1937, +5460]*** |
| total_step_ms | -1270 [-2627, +47] n.s. | **+3861 [+2064, +5763]*** |

*Bootstrap 95% CI excludes zero (10k resamples, seed 42).

### Latency Percentiles (ms)

| Condition | Total Step (median, IQR, p95) | Attack Prep (median, p95) |
|-----------|:---|:---|
| TMA no-lock | 13481 [13037, 15316] p95=15718 | 12023 p95=14187 |
| TMA ArmLock | 13005 [10387, 15202] p95=15697 | 11870 p95=13927 |
| Prefix no-lock | 36496 [33996, 37372] p95=46149 | 35252 p95=44545 |
| Prefix ArmLock | 37952 [37160, 43653] p95=47964 | 36681 p95=46178 |

Prefix/TMA attack prep ratio: ~2.9-3.1x (median), ~3.2-3.3x (mean).

---

## Frozen Scientific Claims

1. **ArmLock perfectly zeros executed arm discrepancy**: NAD_exec_arm = 0 across all 540 ArmLock attack frames.
2. **TMA produces larger policy-arm NAD than Prefix** (0.162-0.172 vs 0.106), consistent with the more focused log-ratio objective.
3. **Executed gripper NAD is not significantly different between ArmLock and no-lock** (paired bootstrap CIs cross or barely exclude zero, 11-14/27 pairs show zero difference).
4. **Prefix ArmLock is significantly slower than Prefix no-lock** (paired delta +3.6s, 95% CI [+2.0s, +5.8s]).
5. **Prefix attack preparation is ~3x TMA** (35-39s vs 11-12s).
6. **New-state breadth confirms attack generalizes** with conditional FR 81-91% across conditions, limited primarily by detector coverage (87.5%).

---

## Pending (Not Frozen)

- **CQFR human review**: 55-video blind package ready; reviewer outcome/failure labels not yet collected
- **Reference Panel A**: legacy N=45 panel lacks current-bridge provenance; auditable N=27 metric-refresh can serve
- **526 scientific-key global dedup**: not yet performed across all panels
- **Cross-suite**: blocked on model availability
- **Action-Discrepancy baseline**: not implemented

---

## CQFR Instructions

1. Send `CQFR_REVIEWER_TEMPLATE.csv` + all 55 `B*.mp4` videos to reviewers.
2. Reviewers fill in: `outcome_label`, `outcome_confidence`, `failure_mode`, `failure_subtype`, `notes`.
3. DO NOT share `CQFR_BLIND_KEY_PRIVATE.csv` (contains condition/objective/arm_lock/task_success mapping).
4. After review: compute Cohen's kappa, agreement rates, CQFR per condition, simulator-human agreement.
5. `CQFR_LABEL_DEFINITIONS.txt` provides the label taxonomy.
