# VIS prefix_margin — Repair Status

**Date**: 2026-06-03
**Branch**: `exp/vis-attack-strength-upgrade-20260602`
**Target commit under audit**: `4337d5865181297ef5f12926a1753ff15676839f`

## Summary

Four blocking issues were identified in the prefix_margin evidence chain. This report
tracks repair status across three layers: pre-repair (historical), code audit (current),
and rerun (pending).

## Layer 1 — Pre-Repair Evidence

The historical results in `reports/VIS_FINAL_EXPERIMENT_SUMMARY.md` and
`reports/VIS_KETCHUP_PREFIX_FINAL_RESULT.md` are **downgraded to pre-repair candidate
evidence**. The following issues cast doubt on their validity:

| Issue | Severity | Impact |
|-------|----------|--------|
| A. OPEN semantic inconsistency | P0 | OPEN counts in reports may be inverted (counting CLOSE as OPEN via `adv_grip > 0.5`) |
| B. Prefix-locked gripper loss absent | P0 | `prefix_locked_gripper_open_margin` computed zero gripper loss — only arm preservation |
| C. Teacher-forced fallback used | P0 | Restart selection may have used unreliable teacher-forced open_prob_mass |
| D. Report count inconsistency | P1 | Summary (14/14) vs ketchup report (7/7) — aggregation error |

## Layer 2 — Repaired Code Audit

### A. Canonical gripper semantics

- **Created**: `src/gripper_attack/gripper_semantics.py`
- **Canonical definition**: `raw_gripper < 0.5` ⇔ OPEN (equivalent to `env_gripper > 0` after normalize→invert)
- **Self-check**: import-time assertions verify consistency for key values (-0.996, 0.0, 0.996)
- **Status**: ✅ Code complete, pending server-side execution test

### B. Prefix-locked loss fix

- **Modified**: `src/gripper_attack/attack_adapter.py::_loss()`
- **Change**: Gripper loss now computed directly from `logits[0, -1, :]`, independent of labels
- **Formula**: `loss = relu(max_non_open - logsumexp(open_region) + margin) + arm_weight * mean(arm_CEs)`
- **Debug fields**: `prefix_locked_gripper_loss_present`, `gripper_loss_value`, `arm_loss_value`, `gripper_open_region_token_count`, `canonical_open_semantics_version`
- **Status**: ✅ Code complete, pending server-side model test

### C. Teacher-forced fallback ban

- **Modified**: `scripts/vis_rollout_adaptive_v3.py::run_pgd_attack()`
- **Change**: Missing `adv_inputs` or re-decode failure → `RuntimeError` (not silent fallback)
- **Status**: ✅ Code complete, pending server-side integration test

### D. OPEN predicate replacement

- All OPEN checks now use `raw_gripper_is_open()` from `gripper_semantics`
- `adv_grip > 0.5` → `raw_gripper_is_open(r['adv_grip'])` (trace statistics)
- `is_open = adv_grip > 0.5` → `is_open = raw_gripper_is_open(adv_grip)` (controller)
- `_is_open = _gen_action[-1] < 0.5` → `_is_open = raw_gripper_is_open(float(_gen_action[-1]))` (restart)
- **Status**: ✅ Code complete

### Provenance script

- **Created**: `scripts/diagnostics/audit_prefix_margin_provenance.py`
- **Outputs**: `tables/vis_prefix_margin_provenance.csv`, `tables/vis_prefix_margin_group_summary.csv`, `reports/VIS_PREFIX_MARGIN_REPAIR_AUDIT.md`
- **Status**: ✅ Code complete, pending server-side execution with actual trace CSVs

### Unit tests

- `tests/v4/test_gripper_semantics_consistency.py` — semantics consistency
- `tests/v4/test_prefix_locked_loss_contains_gripper.py` — loss logic verification
- `tests/v4/test_no_teacher_forced_fallback_for_prefix.py` — fallback ban
- `tests/v4/test_provenance_aggregator_schema.py` — schema validation
- All files compile cleanly (`python -m py_compile` passes)
- **Status**: ✅ Written, pending server-side `pytest` execution

## Layer 3 — Repaired Rerun Evidence

**NOT YET AVAILABLE.** All reruns pending the minimal matrix defined below.

### Minimum rerun matrix

Task: `ketchup`, state_id=0

| Window | eps | Condition | Seeds |
|--------|-----|-----------|-------|
| 10-27 | 6 | prefix_locked_gripper_open_margin | 0,1,2,3 |
| 10-27 | 6 | random_linf | 0,1,2,3,4,5 |
| 20-37 | 6 | prefix_locked_gripper_open_margin | 0,1,2 |
| 20-37 | 6 | random_linf | 0,1,2,3,4,5 |

Optional (if core passes):
| 10-27 | 4 | prefix | 0,1,2 |
| 10-27 | 8 | prefix+random | TBD |

### Claim gate criteria (for Layer 3)

Only when ALL of the following hold:

For ketchup 10-27 eps6:
- [ ] prefix valid runs >= 4
- [ ] random valid runs >= 6
- [ ] prefix `generated_open_count_canonical` clearly reported (expected >0 and dominant in window)
- [ ] random `generated_open_count_canonical` = 0
- [ ] prefix qpos shows closed→open physical response
- [ ] random qpos near clean / no comparable opening
- [ ] prefix armL2 = 0 or near-zero
- [ ] random success/done semantics clean (all or near-all True)
- [ ] failure taxonomy = early_grasp_disruption
- [ ] NO teacher-forced open_prob_mass as evidence
- [ ] provenance CSV matches report

## Current Claim Boundary

### Allowed (after repair, post-rerun)
- Same-budget VIS-specific ketchup early_grasp_disruption at eps6 (IF gate criteria met)
- Gripper-channel selectivity (armL2=0)
- Window generalization to 20-37 (supporting only, not universal)

### Allowed (pre-repair, for historical reference only)
- Action bridge: prefix_margin induces generated gripper OPEN in rollout
- Physical bridge: qpos transitions from closed to open
- These are candidate claims needing provenance recomputation

### Forbidden
- ProprioNoStep-guided VIS
- Broad LIBERO generalization
- Pre-release drop mechanism for ketchup early windows
- Trained detector ready
- Teacher-forced probability as evidence
- Direct `< 0.5` / `> 0.5` comparisons outside `gripper_semantics`

## Commit Status

Repair changes committed as part of this audit. See commit log for details.

## Files Changed

| File | Status |
|------|--------|
| `src/gripper_attack/gripper_semantics.py` | NEW |
| `src/gripper_attack/attack_adapter.py` | MODIFIED |
| `scripts/vis_rollout_adaptive_v3.py` | MODIFIED |
| `scripts/diagnostics/audit_prefix_margin_provenance.py` | NEW |
| `tests/v4/test_gripper_semantics_consistency.py` | NEW |
| `tests/v4/test_prefix_locked_loss_contains_gripper.py` | NEW |
| `tests/v4/test_no_teacher_forced_fallback_for_prefix.py` | NEW |
| `tests/v4/test_provenance_aggregator_schema.py` | NEW |
| `tests/run_v4_tests.py` | NEW |
| `reports/VIS_PREFIX_MARGIN_REPAIR_STATUS.md` | NEW |
| `reports/VIS_FINAL_EXPERIMENT_SUMMARY.md` | MODIFIED (claim downgrade) |
