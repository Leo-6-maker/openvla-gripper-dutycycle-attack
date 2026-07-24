# Table 1 — Final Freeze V3.3

**Analysis commit**: `f7a18b425844d8cd39729d00de060fecd8b3e8bc`
**Artifact runtime**: `ace1876281a9ad6ed68e1229a6e17346356766e9`
**Experiment input**: `01d19779ef770135e1ad01fd8541e75e56181057`

---

## Panel A: Auditable Reference (N=27/condition)

| Method | No Lock ITT FR | 95% CI | ArmLock ITT FR | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| TMA | 24/27 = 88.9% | [71.9%, 96.1%] | 22/27 = 81.5% | [63.3%, 91.8%] |
| Prefix | 22/27 = 81.5% | [63.3%, 91.8%] | 27/27 = 100.0% | [87.5%, 100.0%] |

Coverage 100%. McNemar: TMA b=5,c=3,p=0.727; Prefix b=0,c=5,p=0.063.

## Panel B: New-State Breadth (ITT N=24, Cond N=21)

| Condition | ITT FR | 95% CI | Cond FR | 95% CI | Cov |
|-----------|:---:|:---:|:---:|:---:|:---:|
| RAND | 0.0% | [0.0%,13.8%] | 0.0% | [0.0%,15.5%] | 87.5% |
| TMA no-lock | 70.8% | [50.8%,85.1%] | 81.0% | [60.0%,92.3%] | 87.5% |
| TMA ArmLock | 75.0% | [55.1%,88.0%] | 85.7% | [65.4%,95.0%] | 87.5% |
| Prefix no-lock | 79.2% | [59.5%,90.8%] | 90.5% | [71.1%,97.3%] | 87.5% |
| Prefix ArmLock | 79.2% | [59.5%,90.8%] | 90.5% | [71.1%,97.3%] | 87.5% |

## Panel D: Baseline

| Method | FR | 95% CI | N |
|--------|:---:|:---:|:--:|
| RAND | 0/24 = 0.0% | [0.0%,13.8%] | 24 |
| Adapted Untargeted Clean-Token CE PGD | 0/27 = 0.0% | [0.0%,12.5%] | 27 |

---

## Table 2: Mechanism (rNAD V3.1, representation-aware, same-space)

Action stats from victim model `dataset_statistics.json` (SHA: `a4b953c2`).
Env gripper range = 2.0 (policy [0,1] -> postprocess -> env [-1,+1]).

| Metric | TMA NL | TMA AL | Prefix NL | Prefix AL |
|--------|:------:|:------:|:---------:|:---------:|
| rNAD_pol_prelock_arm | 0.0719 | 0.0757 | 0.0491 | 0.0480 |
| rNAD_pol_exec_arm | 0.0719 | 0.0000 | 0.0491 | 0.0000 |
| rNAD_env_exec_arm | 0.0719 | 0.000000 | 0.0491 | 0.000000 |
| rNAD_env_exec_grip | 0.7778 | 0.7593 | 0.8037 | 0.7926 |

540 ArmLock frames, 0 violations. 108/108 parsed, 1080 attack frames, 0 skipped.

Paired env_exec_grip (N=27): TMA -0.019 [-0.048,+0.011], Prefix -0.011 [-0.041,+0.022]. Both CIs cross zero.

---

## CQFR V3.3 (Ready for Human Review)

- 108 runs -> 68 unique videos (12 duplicate groups, 0 conflicts)
- Public ZIP: `cb09fc7fe7b4fceab604a5a46d59bc8cbf4fb38fb4944934f2b800a7f208eb2e` (2.9MB)
- 73 members (68 videos + 5 aux files), UID/GID stripped, 72 checksums verified
- Dual-axis: task_outcome + contact_quality_failure (independent)
- Controlled_placement as independent placement quality (not failure subtype)
- Separate confidence per axis
- Subtype values: yes/no/ambiguous/not_applicable
- Pre-registered CQFR/CQSR formulas
- Two-reviewer protocol with adjudication
- Cluster statistics: 68 unique videos, not 108 independent observations

## Status

```
RNAD_NUMERICAL_FREEZE_V3_1      = PASS
RNAD_MECHANISM_FREEZE_V3_1      = PASS
CQFR_ARCHIVE_INTEGRITY_V3_3     = PASS
CQFR_SCIENTIFIC_PROTOCOL_V3_3   = PASS
CQFR_HUMAN_REVIEW               = PENDING
TABLE1_PUBLICATION_FREEZE       = HOLD
GPU_NEW_EXPERIMENTS             = HOLD
```
