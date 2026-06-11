# Stage-B RC1a S19 — Top-4 Candidate Confirmation

**Date**: 2026-06-11
**GitHub HEAD**: 036428d (S18 post-freeze audit)
**Type**: Multi-seed RAND-veto + ORACLE + VIS/RAND confirmation of S18 census top candidates

## Executive Summary

**ketchup_s0_w150-160 is the first confirmed non-tomato_sauce parent-level physical bridge POC.** It passes the pre-registered 2/3 gate for both command and physical bridge. Two milk candidates fail (RAND-confounded and command-weak). Orange juice fails at command level (reproducible borderline but streak < 4).

With tomato_sauce_s0_w70-80 and ketchup_s0_w150-160, the project now has **two actual-task physical bridge parents** under matched VIS/RAND control.

## Phase 1: RAND-veto + ORACLE Triage

All 4 candidates passed RAND-veto and ORACLE reachability.

| Candidate | ORACLE | RAND 71 | RAND 72 | RAND 73 | Veto |
|-----------|--------|---------|---------|---------|------|
| ketchup w150-160 | 0.597 | 0/0/0.001 | 0/0/0.001 | 0/0/0.000 | STRICT-CLEAN |
| milk w230-240 | 0.241 | 0/0/0.000 | 0/0/0.001 | 0/0/0.000 | STRICT-CLEAN |
| milk w90-100 | 0.625 | 1/1/0.012 | 2/1/0.014 | 2/2/0.014 | USABLE |
| orange_juice w50-60 | 0.391 | 0/0/0.000 | 0/0/0.004 | 0/0/0.014 | STRICT-CLEAN |

## Phase 2: Multi-Seed VIS/RAND Confirmation

### ketchup_s0_w150-160 — PASS

| Seed | VIS | Streak | RAND | Streak | gap | VIS norm | CMD | PHYS |
|------|-----|--------|------|--------|-----|----------|-----|------|
| 71 | 8 | 6 | **7** | 4 | +1 | 0.930 | ❌ RAND confounded | — |
| 72 | 7 | 5 | 0 | 0 | +7 | 0.908 | ✅ | ✅ PHYS_PASS |
| 73 | 9 | 9 | 0 | 0 | +9 | 0.344 | ✅ | ✅ PHYS_PASS |

**2/3 command-positive, 2/3 physical bridge PASS. ORACLE=0.597.**

Seed71 is RAND-confounded (RAND open=7) despite passing Phase-1 veto with different job_id. The matched-control design correctly catches this within Phase 2.

### milk_s0_w230-240 — FAIL

| Seed | VIS | Streak | RAND | Streak | gap | Result |
|------|-----|--------|------|--------|-----|--------|
| 71 | 0 | 0 | 1 | 1 | -1 | CMD_WEAK |
| 72 | 7 | 4 | **3** | 2 | +4 | RAND_CONFOUNDED |
| 73 | 7 | 4 | **5** | 4 | +2 | RAND_CONFOUNDED |

**0/3 command-positive. 2/3 seeds RAND-confounded.** Consistent with milk's historical RAND sensitivity across tested windows.

### milk_s0_w90-100 — FAIL

| Seed | VIS | Streak | RAND | Streak | gap | Result |
|------|-----|--------|------|--------|-----|--------|
| 71 | 5 | 2 | 2 | 1 | +3 | BORDERLINE |
| 72 | 4 | 2 | 1 | 1 | +3 | BORDERLINE |
| 73 | 1 | 1 | 1 | 1 | 0 | CMD_WEAK |

**0/3 command-positive.** VIS command transfer weakens substantially from the strong S18 single-seed (VIS=8/4) to multi-seed replication (VIS=1-5).

### orange_juice_s0_w50-60 — FAIL

| Seed | VIS | Streak | RAND | Streak | gap | Result |
|------|-----|--------|------|--------|-----|--------|
| 71 | 7 | 3 | 2 | 1 | +5 | BORDERLINE (streak<4) |
| 72 | 5 | 2 | 2 | 2 | +3 | BORDERLINE |
| 73 | 5 | 3 | 0 | 0 | +5 | BORDERLINE (streak<4) |

**0/3 command-positive.** VIS produces moderate OPEN (5-7/10) but lacks sustained streak (2-3, gate requires ≥4). Physical norm reaches 0.28 on seed71, but command gate must be satisfied first.

## Confirmed Physical Bridge Registry

| Task | Window | Seeds | CMD pass | PHYS pass | ORACLE | Status |
|------|--------|-------|----------|-----------|--------|--------|
| tomato_sauce | w70-80 | 6 | 6/6 | 5/6 | 0.295 | confirmed anchor POC |
| **ketchup** | **w150-160** | **3** | **2/3** | **2/3** | **0.597** | **confirmed second POC** |

## Claim Matrix

### Allowed

- ketchup_s0_w150-160 is a confirmed parent-level physical bridge POC under matched VIS/RAND control (2/3 seeds, pre-registered gate).
- Together with tomato_sauce_s0_w70-80 (5/6 PASS), two actual LIBERO Object tasks have clean physical bridge evidence.
- VIS>RAND command-duty signal exists beyond tomato_sauce and can survive multi-seed confirmation with RAND-veto.
- milk tested windows show repeated RAND sensitivity / confirmation failure.
- orange_juice_s0_w50-60 shows reproducible borderline VIS response but does not meet sustained-open command gate.

### Forbidden

- ketchup object-wide or task-wide success (single parent only).
- Task failure / object drop (Level 3 not evaluated).
- milk as object-level negative (tested windows only).
- Detector solved / Layer3 solved globally.
- Claims using pre-S16R old task labels.

## Infrastructure

- 16 Phase-1 jobs (4 candidates × 4: ORACLE + 3 RAND)
- 24 Phase-2 jobs (4 candidates × 3 seeds × 2 conditions: VIS+RAND)
- 0 infra failures
- All summaries pass provenance gate (actual_task_key matches)

## Artifacts

| Artifact | Path |
|----------|------|
| S19 report | `reports/STAGEB_RC1A_S19_TOP4_CONFIRMATION_20260611.md` |
| Phase 1 results | `tables/s19a_randveto_oracle_top4.csv` |
| Phase 2 results | `tables/s19b_multiseed_visrand_confirmation.csv` |
| Claim matrix | `tables/s19_claim_matrix.csv` |
| Bridge registry | `tables/s19_confirmed_physical_bridge_registry.csv` |

## Methodology Note

Phase-1 RAND-veto used different job_ids than Phase-2 matched RAND controls because the runner's `random_seed_str = attack_seed + job_id`. As a result, Phase-1 RAND results are indicative but not directly transferable to Phase-2 matched controls. Future protocol should fix job_ids across phases or decouple random_seed_str from job_id.

## Next Steps

1. S20a (optional): ketchup extra confirmation seeds 74/75 to strengthen from 2/3 to 4/5.
2. S20b (optional): contact/task-effect audit on both confirmed windows.
3. S20c (optional): backup candidate triage (ketchup w230-240, butter, bbq_sauce, tomato_sauce w90-100).
