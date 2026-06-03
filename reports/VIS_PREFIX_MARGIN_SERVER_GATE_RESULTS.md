# VIS prefix_margin — Server Gate Results

**Date**: 2026-06-03
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Commit**: `451bd94` (server-side: `openvla_official_libero_20260525` env)

## Gate 1 — Unit Tests

**pytest tests/v4/**: 55 passed, 0 failed

Key tests verified:
- `test_gripper_row_is_neg2` — PASS
- `test_changing_wrong_row_does_not_change_loss` — PASS
- `test_no_teacher_forced_fallback_for_prefix` — PASS
- `test_provenance_aggregator_schema` — PASS

## Gate 2 — Grep Audit

All hits are comments or documentation explaining the canonical definition, or inside the helper/tests:

```
scripts/diagnostics/audit_prefix_margin_provenance.py:290  — comment
scripts/diagnostics/audit_prefix_margin_provenance.py:683  — report text
scripts/diagnostics/audit_gripper_physical_semantics.py:262 — report text
tests/v4/test_provenance_aggregator_schema.py:117           — docstring
```

No active code uses direct `< 0.5` / `> 0.5` comparisons outside the approved locations.

## Gate 3 — Physical Semantics Smoke

| Condition | raw_gripper | env_gripper | qpos start | qpos end | delta | Status |
|-----------|------------|-------------|------------|----------|-------|--------|
| OPEN | 0.0 | +1 | 0.0200 | 0.0013 | -0.0188 | PASS |
| CLOSE | 0.996 | -1 | 0.0210 | 0.0387 | +0.0178 | PASS |

Canonical semantics confirmed: `raw_gripper < 0.5 ⇔ env=+1 ⇔ qpos decreases (physical OPEN)`.

## Gate 4 — Single-Frame Prefix Margin Sanity

| Field | Value | Status |
|-------|-------|--------|
| canonical_open | True | PASS |
| gripper_row_index | -2 | PASS |
| prefix_locked_gripper_loss_present | True | PASS |
| prefix_locked_arm_loss_present | True | PASS |
| gripper_loss_value | 0.0 | PASS (OPEN dominates — margin satisfied) |
| arm_loss_value | 3.38e-08 | PASS (near-zero — arm preserved) |
| gripper_open_region_token_count | 127 | PASS |
| teacher_forced_fallback_used | False | PASS |
| restart_selection_metric | autoregressive_generated_open | PASS |
| arm_l2 | 8.77e-09 | PASS |
| error | (empty) | PASS |

## Verdict

**ALL 4 GATES PASSED.** Ready for repaired fixed-window rerun matrix.
