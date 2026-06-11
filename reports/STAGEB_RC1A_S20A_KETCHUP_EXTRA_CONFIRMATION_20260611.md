# Stage-B RC1a S20a — Ketchup Extra Confirmation Under Repaired RAND Protocol

**Date**: 2026-06-11
**GitHub HEAD**: 7f7da33 (S20-0 protocol patch)
**Type**: Multi-seed VIS/RAND confirmation with explicit random_control_seed protocol

## Executive Summary

**S20a: 3/3 fresh ketchup seeds PHYS_PASS under repaired explicit random_control_seed protocol.** This independently strengthens the ketchup_s0_w150-160 physical bridge POC from S19's 2/3 (legacy RAND protocol) to a combined 5/6 PHYS_PASS with 1 RAND-confounded seed.

With tomato_sauce_s0_w70-80 (5/6) and ketchup_s0_w150-160 (5/6 combined), both confirmed physical bridge parents now have comparable multi-seed strength.

## Protocol

All RAND jobs use the repaired explicit protocol from S20-0:
- `--random_control_seed` equal to the attack seed
- RAND output is independent of job_id
- Legacy `attack_seed + job_id` behavior is bypassed

S19's ketchup seeds 71-73 used legacy protocol (RAND output dependent on job_id). S20a's fresh seeds 74-76 use the repaired explicit protocol. Results should be interpreted as an independent strengthening, not a simple seed accumulation.

## Results

ORACLE sanity: 0.5973 (matches S19 Phase-1: 0.5973).

| Seed | VIS | Streak | RAND | Streak | gap | qpos | VIS/ORACLE | CMD | PHYS |
|------|-----|--------|------|--------|-----|------|------------|-----|------|
| 74 | 8 | 6 | 0 | 0 | +8 | 0.459 | 0.77 | ✅ | ✅ |
| 75 | 7 | 5 | 2 | 2 | +5 | 0.402 | 0.67 | ✅ | ✅ |
| 76 | 7 | 4 | 1 | 1 | +6 | 0.565 | 0.95 | ✅ | ✅ |

### Combined Ketchup Status (S19 + S20a)

| Seed | Phase | Protocol | CMD | PHYS | Notes |
|------|-------|----------|-----|------|-------|
| 71 | S19 | legacy | ❌ | — | RAND open=7 (confounded) |
| 72 | S19 | legacy | ✅ | ✅ | vn=0.91 |
| 73 | S19 | legacy | ✅ | ✅ | vn=0.34 |
| 74 | S20a | explicit | ✅ | ✅ | vn=0.77 |
| 75 | S20a | explicit | ✅ | ✅ | vn=0.67 |
| 76 | S20a | explicit | ✅ | ✅ | vn=0.95 |

**5/6 PHYS_PASS, 1 RAND-confounded.** S20a independently demonstrates 3/3 fresh seeds PHYS_PASS under the repaired explicit protocol.

## Updated Physical Bridge Registry

| Task | Window | Seeds | PHYS pass | Confounded | ORACLE | Status |
|------|--------|-------|-----------|------------|--------|--------|
| tomato_sauce | w70-80 | 6 | 5 | 0 | 0.295 | confirmed anchor |
| ketchup | w150-160 | 6 | 5 | 1 | 0.597 | confirmed second |

## Claim Boundary

### Allowed

- S20a independently demonstrates 3/3 fresh ketchup PHYS_PASS under repaired explicit random_control_seed protocol.
- Combined with S19 (2/3 legacy protocol), ketchup has 5/6 PHYS_PASS with 1 RAND-confounded seed.
- Both confirmed physical bridge parents now have comparable multi-seed strength (5/6 each).
- The repaired random_control_seed protocol eliminates the job_id dependency that caused S19 Phase-1/Phase-2 RAND mismatch.

### Forbidden

- ketchup object-wide or task-wide success
- Task failure / object drop (Level 3 not evaluated)
- Detector solved / Layer3 solved globally
- Old task labels

## Next Step

S20b: Full-episode task/contact-effect audit on both confirmed physical bridge parents.

## Artifacts

| Artifact | Path |
|----------|------|
| S20a report | `reports/STAGEB_RC1A_S20A_KETCHUP_EXTRA_CONFIRMATION_20260611.md` |
| Confirmation table | `tables/s20a_ketchup_extra_confirmation.csv` |
| Updated registry | `tables/s20_confirmed_physical_bridge_registry_updated.csv` |
