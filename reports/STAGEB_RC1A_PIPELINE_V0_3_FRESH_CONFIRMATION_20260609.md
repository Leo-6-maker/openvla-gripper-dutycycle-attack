# Pipeline v0.3 Fresh Confirmation Report

**Date**: 2026-06-09
**Commit**: 65b7741 (design) + staggered 3-worker launch
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

Pipeline v0.3 abstain-first VIS attack window selection was validated under fresh matched confirmation. **All 5 gates PASS.** CleanRand-pass windows achieve perfect VIS-specific command yield (+1.00), significantly outperforming both TaskOnly baseline (+0.00) and high-risk abstained controls (-0.38).

## Experiment Design

```
12 windows × 2 fresh attack seeds (5,6) × VIS/RAND = 48 jobs
3 groups × 4 windows × 4 jobs each
Fresh attack_seed=5,6 — not used in K5/K5b/K5c (which used 0-4)
Logical pair key = pair_id + "__atk" + attack_seed
```

| Group | # Pairs | Selection Strategy | Meaning |
|-------|---------|-------------------|---------|
| A | 8 | CleanRand OOF score <= p50 | Detector says "safe to attack" |
| B | 8 | Task-prior baseline | Naive selection without abstain |
| C | 8 | CleanRand OOF score > p50 | Detector says "avoid" |

## Results

### Group Metrics

| Group | cmd_hit | cmd_rand | abst_any | yield_cmd | V_qpos | R_qpos |
|-------|---------|----------|----------|-----------|--------|--------|
| A: CleanRand-pass | **1.00** | **0.00** | 0.62 | **+1.00** | 0.055 | 0.016 |
| B: TaskOnly | 0.25 | 0.50 | 0.50 | +0.00 | 0.025 | 0.010 |
| C: High-risk | 0.00 | 0.88 | 0.88 | -0.38 | 0.039 | 0.016 |

### Infra

- 48/48 jobs infra=ok
- 0 early termination
- 0 CUDA errors
- 0 job_id conflicts
- 24/24 logical VIS/RAND pairs complete

### Gate Results

| Gate | Test | Result |
|------|------|--------|
| G1 | A cmd_rand (0.00) < B (0.50) | PASS |
| G2 | A yield (+1.00) > B (+0.00) | PASS |
| G3 | A cmd_hit (1.00) >= 0.3 | PASS |
| G4 | C cmd_rand (0.88) > A (0.00) | PASS |
| G5 | 48/48 infra=ok | PASS |

### Detector FP/FN Cases

| Case | Window | Detector | Fresh Result | Type |
|------|--------|----------|-------------|------|
| FP | tomato[55,65] | oof_rand=0.97 (high-risk) | VIS=7, RAND=0 | Conservative false-positive: GOLD cmd+phys rejected by filter |
| FN | salad[70,80] | oof_rand=0.09 (pass) | VIS=4, RAND=10-11 | False-negative: rand-sensitive window passed through |

### Per-Window Detail

**Group A (CleanRand-pass)**: All 8 logical pairs VIS-cmd=1, RAND-cmd=0
- milk[70,80] atk=5,6: VIS=8-9, RAND=0-1
- butter[80,90] atk=5,6: VIS=8, RAND=3
- cream[50,60] atk=5,6: VIS=8, RAND=0-1
- tomato[150,160] atk=5,6: VIS=8, RAND=1-2

**Group B (TaskOnly)**: Mixed — 2/8 VIS-cmd+RAND-cmd=0 (FP tomato case), 4/8 RAND-cmd=1 (FN salad + milk confound)
- tomato[55,65] atk=5,6: VIS=7, RAND=0 [detector FP]
- milk[75,85] atk=5,6: VIS=7-8, RAND=6-7 [both succeed]
- cream[85,95] atk=5,6: VIS=0-1, RAND=0 [both fail]
- salad[70,80] atk=5,6: VIS=4, RAND=10-11 [detector FN]

**Group C (High-risk)**: 7/8 RAND-cmd=1
- milk[80,90], butter[95,105], alphabet[60,70], tomato[115,125]: all rand-dominated

## Claim Boundary

### Can Claim
- Abstain-first pipeline v0.3 validated under fresh matched VIS/RAND confirmation
- CleanRand abstain filter improves command-level VIS-specific yield over baseline
- Random-sensitive windows are learnable from clean online features

### Cannot Claim
- Full vulnerable-window detector solved
- CleanCmd ranking solved (Layer-2 still uses random/simple ranking)
- Strict physical detector solved (phys enrichment failed in K5c)
- Cross-task or cross-suite generalization demonstrated

## Artifacts

```
tables/pipeline_v0_3_confirmation_job_audit.csv
tables/pipeline_v0_3_confirmation_pair_audit.csv
tables/pipeline_v0_3_confirmation_group_metrics.csv
tables/pipeline_v0_3_confirmation_window_results.csv
tables/pipeline_v0_3_confirmation_launch_manifest.csv
scripts/diagnostics/postprocess_pipeline_v0_3_confirmation.py
```

## Current Best Pipeline

```
Clean online features
  → Layer-1: CleanRand abstain filter (AUROC=0.777)
  → Simple/random ranking among survivors
  → VIS/RAND matched attack confirmation
  → Command-level VIS-specific yield + physical qpos measurement
```

Layer-2 CleanCmd ranking and Layer-3 strict physical bridge remain WIP.
