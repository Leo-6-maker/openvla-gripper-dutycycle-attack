# Table 1 — Final Freeze V2

**Status**: `TABLE1_CORE_NUMERICAL_FREEZE = PASS`, `RNAD_MECHANISM_FREEZE_V2 = PASS`
**Previous commit**: `c82e85cbd67415a50d9a216106b924a4e9c71ef5`
**Git roles**:
- `artifact_runtime_git_head`: `ace1876281a9ad6ed68e1229a6e17346356766e9`
- `experiment_report_commit`: `01d19779ef770135e1ad01fd8541e75e56181057`
- `analysis_code_commit`: `904a26a0a984ec52660143b0b151d3d536c99212`
- `previous_table1_commit`: `c82e85cbd67415a50d9a216106b924a4e9c71ef5`

---

## Panel A: Auditable Reference (N=27/condition)

| Method | No Lock ITT FR | 95% CI | ArmLock ITT FR | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| TMA | 24/27 = 88.9% | [71.9%, 96.1%] | 22/27 = 81.5% | [63.3%, 91.8%] |
| Prefix | 22/27 = 81.5% | [63.3%, 91.8%] | 27/27 = 100.0% | [87.5%, 100.0%] |

Coverage 100%. McNemar: TMA b=5,c=3,p=0.727; Prefix b=0,c=5,p=0.063.

---

## Panel B: New-State Breadth (ITT N=24, Cond N=21)

| Condition | ITT FR | 95% CI | Cond FR | 95% CI | Cov |
|-----------|:---:|:---:|:---:|:---:|:---:|
| RAND | 0.0% | [0.0%, 13.8%] | 0.0% | [0.0%, 15.5%] | 87.5% |
| TMA no-lock | 70.8% | [50.8%, 85.1%] | 81.0% | [60.0%, 92.3%] | 87.5% |
| TMA ArmLock | 75.0% | [55.1%, 88.0%] | 85.7% | [65.4%, 95.0%] | 87.5% |
| Prefix no-lock | 79.2% | [59.5%, 90.8%] | 90.5% | [71.1%, 97.3%] | 87.5% |
| Prefix ArmLock | 79.2% | [59.5%, 90.8%] | 90.5% | [71.1%, 97.3%] | 87.5% |

McNemar (cond): TMA b=0,c=1,p=1.0; Prefix b=1,c=1,p=1.0.

---

## Panel D: Baseline

| Method | FR | 95% CI | N |
|--------|:---:|:---:|:--:|
| RAND | 0/24 = 0.0% | [0.0%, 13.8%] | 24 |
| Adapted Untargeted Clean-Token CE PGD | 0/27 = 0.0% | [0.0%, 12.5%] | 27 |

---

## Table 2: Mechanism (rNAD v2, same-space, 108 metric refresh)

Action stats from victim model `dataset_statistics.json`, unnorm_key=`libero_object`.

| Metric | TMA no-lock | TMA ArmLock | Prefix no-lock | Prefix ArmLock |
|--------|:----------:|:----------:|:--------------:|:--------------:|
| rNAD_pol_prelock_arm | 0.0719 | 0.0757 | 0.0491 | 0.0480 |
| rNAD_pol_exec_arm | 0.0719 | **0.0000** | 0.0491 | **0.0000** |
| rNAD_env_exec_arm | 0.0719 | **0.000000** | 0.0491 | **0.000000** |
| rNAD_pol_prelock_grip | 0.7747 | 0.7563 | 0.8006 | 0.7895 |
| rNAD_env_exec_grip | 1.5556 | 1.5185 | 1.6074 | 1.5852 |

ArmLock invariant: 540 frames, 0 violations. All comparisons same-space only.

### Paired rNAD Deltas (ArmLock - NoLock, N=27)

| Metric | TMA delta [95% CI] | Prefix delta [95% CI] |
|--------|:---|:---|
| rNAD_env_exec_arm | **-0.0719 [-0.083, -0.061]** | **-0.0491 [-0.059, -0.040]** |
| rNAD_env_exec_grip | -0.037 [-0.096, +0.022] n.s. | -0.022 [-0.082, +0.044] n.s. |

Both env exec grip CIs cross zero; 16/27 pairs show zero difference. No clear paired directional effect.

### Latency (mean, per attacked step)

| Condition | Attack Prep | Total Step |
|-----------|:---:|:---:|
| TMA no-lock | 12.3 s | 13.7 s |
| TMA ArmLock | 11.1 s | 12.4 s |
| Prefix no-lock | 35.4 s | 36.6 s |
| Prefix ArmLock | 39.0 s | 40.4 s |

Prefix/TMA ratio: mean 2.9-3.5x, median 2.9-3.1x.

### Latency Percentiles (median [IQR] p95)

| Condition | Total Step (ms) | Attack Prep (ms) |
|-----------|:---|:---|
| TMA no-lock | 13481 [13037,15316] p95=15718 | 12023 p95=14187 |
| TMA ArmLock | 13005 [10387,15202] p95=15697 | 11870 p95=13927 |
| Prefix no-lock | 36496 [33996,37372] p95=46149 | 35252 p95=44545 |
| Prefix ArmLock | 37952 [37160,43653] p95=47964 | 36681 p95=46178 |

---

## Frozen Claims

1. TMA and Prefix produce high task failure rates in LIBERO-Object reference cells.
2. Attack transfers to 8 previously unevaluated clean-qualified state slots, with state-dependent effectiveness and coverage.
3. Prefix ArmLock 100% FR is limited to reference cells; new-state breadth shows 79.2% ITT FR.
4. tomato_sauce_s1 exhibits state-specific (not task-wide) detector non-emission.
5. ArmLock zeros executed arm discrepancy in both policy and environment spaces (540 frames, 0 violations).
6. High failure rates persist when arm perturbations are exactly removed, supporting an important causal role for the gripper channel.
7. Same-space rNAD_env_exec_grip is large (1.5-1.6) and ArmLock-noLock paired deltas are small with CIs crossing zero.
8. RAND and Adapted Untargeted Clean-Token CE PGD produce zero observed task failures.
9. Prefix current implementation requires ~2.9-3.5x (mean) / ~2.9-3.1x (median) the attack-preparation latency of TMA.
10. Legacy timing evidence suggests Random-Time < Early-Shift < Student Trigger; formal provenance reconstruction remains pending.

## Pending

- CQFR human review (68 unique videos ready)
- Timing panel formal subset reconstruction
- Global accepted-run count final adjudication
- Reference N=45 legacy panel audit
- Cross-suite generalization
