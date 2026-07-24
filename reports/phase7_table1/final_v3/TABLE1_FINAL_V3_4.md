# Table 1 — Final Freeze V3.4

**Analysis generator commit**: `c7a446752228234f6e98458578b03f71da5db83c`
**Artifact runtime**: `ace1876281a9ad6ed68e1229a6e17346356766e9`
**Experiment input**: `01d19779ef770135e1ad01fd8541e75e56181057`

---

## Panel A: Auditable Reference (N=27/condition)

| Method | No Lock ITT FR | 95% CI | ArmLock ITT FR | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| TMA | 24/27 = 88.9% | [71.9%, 96.1%] | 22/27 = 81.5% | [63.3%, 91.8%] |
| Prefix | 22/27 = 81.5% | [63.3%, 91.8%] | 27/27 = 100.0% | [87.5%, 100.0%] |

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
| Untargeted CE PGD | 0/27 = 0.0% | [0.0%,12.5%] | 27 |

---

## Table 2: Mechanism (rNAD V3.1, same-space, actual model stats)

| Metric | TMA NL | TMA AL | Prefix NL | Prefix AL |
|--------|:------:|:------:|:---------:|:---------:|
| rNAD_env_exec_arm | 0.0719 | 0.000000 | 0.0491 | 0.000000 |
| rNAD_env_exec_grip | 0.7778 | 0.7593 | 0.8037 | 0.7926 |

540 ArmLock frames, 0 violations.

---

## CQFR V3.4

- 108 runs -> 68 unique videos (12 dup groups, 0 conflicts)
- Public ZIP: `f3e722702f491524a7775d4cb4f5653aeda523ad4a24032dffaa73309c744081` (2.9MB)
- 73 members, 72 checksums verified, UID/GID stripped
- Reviewer 1: 68/68, Reviewer 2: 36/68 stratified (seed=12345)
- Blinded reviewer templates (no condition/success leak)
- Label aggregator with cluster bootstrap
- Pilot required before formal review

### Core Formulas
```
CQFR_cond = N(CQ=yes) / N(CQ in {yes,no})
CQSR_cond = N(SR=1 AND CQ=no) / N(CQ in {yes,no})
SR-CQ mismatch_cond = N(SR=1 AND CQ=yes) / N(CQ in {yes,no})
```

## Status

```
RNAD_NUMERICAL_FREEZE_V3_1      = PASS
RNAD_MECHANISM_FREEZE_V3_1      = PASS
CQFR_ARCHIVE_INTEGRITY_V3_4     = PASS
CQFR_METRIC_FORMULAS_V3_4       = PASS
CQFR_REVIEWER_ASSIGNMENT_V3_4   = PASS
CQFR_RUBRIC_PILOT               = CONDITIONAL GO
CQFR_FORMAL_HUMAN_REVIEW        = HOLD (pilot first)
TABLE1_PUBLICATION_FREEZE       = HOLD
```
