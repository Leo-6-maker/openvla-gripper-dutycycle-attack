# Stage-B RC1a S12b Cream VIS/RAND Bridge — 0/3 FAIL

**Date**: 2026-06-10
**GitHub HEAD**: b81498a (S12b handoff) → TBD (this freeze)
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

**Verdict: 0/3 FAIL.** Cream_s2_w50-60 is the first window to clear all three pre-VIS gates (Layer1 + ORACLE + RAND-clean) yet fail the VIS physical bridge. VIS command OPEN transfers weakly (2-3/10 vs milk's 6-7/10), and qpos response is near zero (norms 0.002-0.044, all far below 0.2 threshold). RAND remains command-clean (0/0/0 OPEN) but natural drift on seed 23 exceeds VIS response. **RAND-cleanliness is necessary but not sufficient for physical bridge.**

## Infrastructure

| Gate | Result |
|------|--------|
| 6/6 summary JSON | PASS |
| 6/6 trace CSV (81 rows each) | PASS |
| 6/6 infra_status=ok | PASS |
| 0 FAILs / CUDA / EGL / pgd_error | PASS |
| GPU released | PASS |

## Results

ORACLE reference (S11a): `qpos_pos_area = 0.2838`

| Cond | Seed | Baseline | pos_area | neg_area | Open | Streak | ORACLE norm | Verdict |
|------|------|----------|----------|----------|------|--------|-------------|---------|
| VIS | 21 | 0.000952 | 0.002040 | 0.002406 | 2/10 | 1 | 0.007 | FAIL |
| RAND | 21 | 0.000952 | 0.001763 | 0.003387 | 0/10 | 0 | 0.006 | clean |
| VIS | 22 | 0.000952 | 0.012423 | 0.001281 | 3/10 | 2 | 0.044 | FAIL |
| RAND | 22 | 0.000952 | 0.000000 | 0.007468 | 0/10 | 0 | 0.000 | clean |
| VIS | 23 | 0.000980 | 0.000497 | 0.004188 | 2/10 | 2 | 0.002 | FAIL |
| RAND | 23 | 0.000619 | 0.023674 | 0.000006 | 0/10 | 0 | 0.083 | confound |

RAND seed23 has pos_area=0.024 > VIS pos_area=0.0005 — natural drift without any OPEN commands.

## Gates

| Gate | Result |
|------|--------|
| Layer1 detector v0.3 | PASS (pre-VIS) |
| ORACLE physical reachability | PASS (0.284, S11a) |
| RAND-clean veto (S12a) | PASS (strict) |
| Command: VIS open > RAND open | Marginal (2-3 vs 0) |
| Physical: VIS pos > 0, VIS pos > RAND pos | FAIL (seed23 RAND > VIS) |
| ORACLE norm: VIS >= 0.2 for >= 2/3 | FAIL (max 0.044) |

## Trace Spot-Check

Seed22 VIS (best case): PG applied in window, OPEN decoded at steps 50, 57, 58. Qpos stays ~0.001-0.003, barely above baseline 0.000952. Post-window peak 0.0048 then decays.

Seed23 RAND: No OPEN commands. Qpos flat during window (~0.00063). Natural post-window drift to 0.0015 produces pos_area=0.024 — more than any VIS seed.

## Scientific Interpretation

Cream differs from milk in VIS transfer efficiency:
- Milk: VIS open=6-7/10, pos_area=0.08-0.12, norm=0.27-0.42
- Cream: VIS open=2-3/10, pos_area=0.000-0.012, norm=0.00-0.04

The attack objective (prefix_locked_gripper_open_margin, margin=5.0) produces fewer OPEN tokens on cream than on milk. Even when OPEN is decoded, the qpos response is negligible — the cream state-2 dynamics may dampen gripper opening or the attack token sequence may differ in effectiveness.

**This is a Layer3 VIS physical bridge FAIL despite clean pre-VIS gating.** It does not invalidate the milk POC; it constrains the claim scope to milk (and potentially other objects yet to be tested).

## Claim Boundary

**Allowed:**
- Cream clears Layer1 + ORACLE + RAND-clean pre-VIS gates
- Cream VIS physical bridge FAILS (0/3, max norm 0.044)
- RAND-cleanliness is necessary but not sufficient for Layer3 physical bridge
- Milk remains the only clean repeated physical bridge POC (3/4)

**Forbidden:**
- Non-milk POC / Layer3 general solution
- Object-wide success
- Detector solved
- Cream PASS

## Artifacts

```
/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s12b_cream_visrand/
  summary_cream_s2_w50_60_s12b_seed21_vispgd_job952100.json
  summary_cream_s2_w50_60_s12b_seed21_randomlinf_job952101.json
  summary_cream_s2_w50_60_s12b_seed22_vispgd_job952102.json
  summary_cream_s2_w50_60_s12b_seed22_randomlinf_job952103.json
  summary_cream_s2_w50_60_s12b_seed23_vispgd_job952104.json
  summary_cream_s2_w50_60_s12b_seed23_randomlinf_job952105.json
  trace_*.csv (6 files)
```
