# Stage-B v1.1 Corrected Smoke3-B — Final Report

**Date**: 2026-06-07
**Commit**: RC1a provenance (source_snapshot_id=f9840cb1)
**Status**: Command-level VIS effect FOUND; physical bridge NOT ESTABLISHED

## 1. Execution

| Metric | Value |
|--------|-------|
| Windows | 3 |
| Jobs (VIS + RAND) | 6/6 |
| infra=ok | 6/6 |
| Validator PASS | 6/6 |
| Postprocess | 6 traces, 0 rejected |
| Labels | 3 paired |

## 2. Window Results

| Window | Stratum | VIS open | VIS streak | VIS qpos | RAND open | RAND streak | RAND qpos | Label |
|--------|---------|----------|------------|----------|-----------|-------------|-----------|-------|
| cream_cheese s2 [45,55] | hard_neg | **8** | **4** | -0.001 | 0 | 0 | -0.071 | **cmd_susceptible** |
| bbq_sauce s0 [60,70] | medium | 2 | 1 | +0.0001 | 0 | 0 | +0.0001 | weak VIS |
| alphabet_soup s1 [45,55] | high | 1 | 1 | +0.0004 | **9** | **7** | -0.001 | random_confounded |

## 3. Scientific Interpretation

### cream_cheese s2 — command_susceptible POSITIVE

VIS PGD20 with corrected objective produces **8 OPEN commands** (streak 4), while matched random produces **0**. This is the first confirmed cmd_susceptible window under RC1a corrected semantics.

However, VIS qpos_delta_shifted = -0.001 (below 0.01 physical threshold). **Command bridge exists; physical bridge not yet demonstrated.**

### bbq_sauce s0 — weak VIS signal

VIS produces 2 opens vs RAND 0. Not enough for cmd_susceptible threshold (>=6). Directionally correct.

### alphabet_soup s1 — random_dominant

RANDOM produces 9 opens vs VIS 1. Random control dominates. This "high_opportunity" candidate based on clean-trajectory features did not translate to VIS susceptibility.

### Candidate stratum quality

Clean-trajectory stratum (high/medium/hard) does NOT predict VIS susceptibility:
- hard_negative → cmd_susceptible (cream_cheese)
- high_opportunity → random_confounded (alphabet_soup)
- medium_opportunity → weak VIS (bbq_sauce)

## 4. Verdict

| Gate | Status |
|------|--------|
| Schema / validator | PASS |
| Corrected open semantics | PASS |
| Command-level VIS effect | **PASS** (1/3 positive) |
| Physical qpos bridge | **NOT ESTABLISHED** |
| Candidate stratum quality | Weak |
| Allow 12-row pilot | **Yes** |
| Allow overnight batch | **No** |

## 5. Key Statement

> VIS PGD20 with corrected objective successfully induces **command-level OPEN**; physical qpos response remains weak. The corrected objective is no longer inverted, but the command→physical bridge chain is not yet demonstrated.
