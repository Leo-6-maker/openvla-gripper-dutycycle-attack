# Clean-Control 3R Confirmation Readiness Audit

**Date**: 2026-06-06
**Candidates**: `tables/clean_rollout_control_negative_candidates.csv` (12 rows)
**Goal**: >=6 confirmed control negatives for detector v3 training

---

## Control Types

| Type | Description | Count | Reason Non-Vulnerable |
|------|-------------|-------|----------------------|
| `natural_open` | Gripper naturally opens (end of task) | 2 | VIS cannot force what already happens |
| `stable_post_lock` | Gripper locked after grasp | 2 | VIS cannot overcome lock |
| `far_too_early` | Window before any grasp | 2 | No object interaction |
| `post_lock` | After grasp lock, stable | 2 | Physical lock prevents opening |
| `after_done` | After task completion | 2 | Episode already done |
| `no_contact` | No gripper-object contact | 2 | Cannot affect object |

---

## Per-Candidate Audit

| # | Task | State | Window | Type | Clean Source | Phase Class | Overlap w/ Gold? | Priority |
|---|------|-------|--------|------|-------------|-------------|-----------------|----------|
| 1 | ketchup | 1 | [53,70] | natural_open | adaptive pool | natural_open | No | HIGH |
| 2 | ketchup | 2 | [54,71] | natural_open | adaptive pool | natural_open | No | HIGH |
| 3 | ketchup | 3 | [53,70] | stable_post_lock | adaptive pool | stable | No | HIGH |
| 4 | ketchup | 4 | [53,70] | stable_post_lock | adaptive pool | stable | No | HIGH |
| 5 | ketchup | 5 | [55,72] | far_too_early | adaptive pool | far_closed | No | MED |
| 6 | cream_cheese | 1 | [49,66] | post_lock | adaptive pool | post_lock | No | HIGH |
| 7 | cream_cheese | 1 | [59,76] | post_lock | adaptive pool | post_lock | No | HIGH |
| 8 | cream_cheese | 3 | [64,81] | after_done | adaptive pool | natural_open | No | MED |
| 9 | cream_cheese | 8 | [44,61] | no_contact | adaptive pool | pre_lock | No | MED |
| 10 | milk | 5 | [57,74] | natural_open | adaptive pool | natural_open | No | HIGH |
| 11 | bbq_sauce | 0 | [55,72] | post_lock | adaptive pool | post_lock | No | HIGH |
| 12 | bbq_sauce | 4 | [55,72] | post_lock | adaptive pool | post_lock | No | HIGH |

---

## Overlap Check

All 12 candidates are from the "adaptive pool" — they were identified during
the adaptive candidate selection as windows that should NOT be vulnerable.
They are distinct from the 9 gold positive windows:

| Row | Gold Pos Windows (same task) | Overlap? |
|-----|---------------------------|----------|
| ketchup controls | ketchup s0 [16,33], ketchup s1 [21,38] | NO — all controls are [50+,70+] windows |
| cream_cheese controls | cream_cheese s4 [28,45] | NO — all controls are [44+,81] windows |
| milk controls | milk s1 [8,25], milk s4 [19,36] | NO — control is [57,74] |
| bbq_sauce controls | bbq_sauce s9 [22,39] | NO — controls are [55,72] |

**Verdict**: ZERO overlap with gold positives. All controls are in clearly different windows.

---

## Selection for Confirmation

Target: 8-12 candidates, aim for >=6 confirmed negatives.

**Recommended: Run all 12.** Running fewer risks not hitting >=6.

Priority order:
1. HIGH (8 candidates): natural_open, stable_post_lock, post_lock, no_contact → most likely to confirm
2. MED (4 candidates): far_too_early, after_done → slight chance of unexpected VIS opening

---

## Output

| Item | Path |
|------|------|
| 3R outputs | `/data/liuyu/outputs/clean_control_3r_confirmation_20260606/` |
| Summary CSV | `tables/clean_control_3r_confirmation_summary.csv` |
| Report | `reports/CLEAN_CONTROL_3R_CONFIRMATION.md` |

---

## GPU Requirement

- Need 1 GPU pair for ~6h (12 candidates × ~30min 3R)
- Can share GPU pair with calibration v2 (serial: calib v2 first, then clean-control)
- Or use separate pair when available

---

## Readiness Verdict: **READY — launch after calibration v2 or on separate GPU pair**
