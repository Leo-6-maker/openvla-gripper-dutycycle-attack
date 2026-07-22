# PR #97 — Superseded for Authoritative L3

**PR:** https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/97
**Branch:** deepseek/exact-w32-clean-l3-replay-20260722
**Status:** SUPERSEDED

## Reason

Superseded for authoritative L3 by the Factorized runtime path.

The V5 scheduler replay in this PR is:
- TEACHER_CORRIDOR_AND_HEAD_PROXY_DIAGNOSTIC_ONLY
- Uses teacher event_id as proxy for candidate_close
- Uses grasp_prob as proxy for utility
- Uses manipulation_prob as proxy for regrasp
- Uses route_supported as proxy for student_valid

## Historical Value

This PR documents:
1. The corrected known-mask metric contract
2. The 4-layer safety metric framework (L1→L4)
3. The fail-closed calibration provenance architecture
4. The staged output + SHA256SUMS sealing pattern
5. 39/39 CPU synthetic tests

These design patterns should be carried forward into the Factorized V2 L3 analysis.

## Superseding Work

- **Branch:** deepseek/factorized-v2-l3-analysis-20260722
- **Base:** 401f79a05753d970ecc803bb96abc64ce132df42
- **Scope:** analysis/student_trigger_calibration/factorized_*, tests/analysis/

## NOT Modified

This branch is frozen. No further commits will be added. Not merge-blocking for Factorized analysis.
