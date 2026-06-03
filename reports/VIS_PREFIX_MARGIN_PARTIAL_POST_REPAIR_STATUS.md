# VIS prefix_margin — Partial Post-Repair Status

**Date**: 2026-06-03
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Status**: Partial evidence, not final claim. Proceeding to auto-window bridge smoke.

## Gate Results

| Gate | Result |
|------|--------|
| pytest tests/v4 | 55 passed |
| grep audit | Clean |
| Physical smoke | PASS |
| No-rollout sanity | canonical_open=True, row=-2, loss_present=True |

## Post-Repair Prefix (ketchup 10-27 eps6)

| Seed | OPEN | qposΔ | armL2 | done |
|------|------|-------|-------|------|
| 1 | 18/18 | 0.03756 | 0.000000 | False |
| 2 | 18/18 | 0.03755 | 0.000000 | False |
| 3 | 18/18 | 0.03756 | 0.000000 | False |

All 3: 100% OPEN, full physical opening, zero arm drift.

## Post-Repair Random (ketchup 10-27 eps6)

| Seed | OPEN | qposΔ | done |
|------|------|-------|------|
| 0 | 0/18 | 0.0006 | True |
| 1 | 0/18 | 0.0006 | True |
| 4 | 0/18 | 0.0006 | True |
| 5 | 0/18 | 0.0006 | True |

## Strict Primary Gate Status

| Requirement | Current | Target | Status |
|-------------|---------|--------|--------|
| prefix_unique_seed_count | 3 | >=4 | MISSING |
| random_unique_seed_count | 4 | >=6 | MISSING |
| prefix_fail | 3/3 | >=4 | MISSING |
| random_fail | 0/4 | 0 | OK |
| all_random_open_zero | True | True | OK |
| canonical_open_min | 18 | >=16 | OK |
| qpos_delta_post_min | 0.038 | >=0.03 | OK |
| armL2_max | 0.0 | <=1e-6 | OK |

**Missing**: prefix seed 0, random seeds 2,3 (deferred to healthy/better server).

## GPU Xid Reason for Stopping

GPU 0,2,3 developed Xid 13/43 (SM Warp Exception / GPU channel stop) during PGD workloads. Reboot cleared all Xid errors. Remaining primary runs deferred.

## Statement

This is NOT a final repaired primary claim. The partial evidence strongly supports the prefix_margin mechanism but lacks the strict seed-count gates for advisor-facing primary claim. Proceeding to ProprioNoStep auto-window bridge smoke to advance the pipeline.
