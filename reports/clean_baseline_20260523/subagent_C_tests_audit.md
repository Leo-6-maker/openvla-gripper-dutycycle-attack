# Subagent C — Tests Audit

## Summary

Audited all 14 test files in `tests/v4/` against 10 assertions plus 3 supplementary checks. **9 of 10 core assertions PASS**, 1 has a minor deviation (assertion 9). Protocol-drift regression coverage is strong. No imports from removed/renamed modules were found. One minor coverage gap exists for `TRIAGE_MATCHED_CONDITIONS` sentinel methods.

---

## Per-Assertion Results (table)

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | NO test continues to protect wrong protocol (random rho=0, targeted using gripper_logit_margin_cw, command_open rho=0) | **PASS** | All three deprecated-property references are accompanied by comments explicitly identifying them as wrong. `test_task_identity.py` lines 120-128: "this is the bug" / "this DISABLES oracle override". `test_protocol_validation.py` rejects rho=0 and wrong-objective. `test_clean_protocol.py` and `test_metadata_schema.py` assert the correct values. |
| 2 | No test asserts random rho=0 (deprecated tests verify rho=0 is WRONG, not correct) | **PASS** | `TestDeprecatedConditions.test_deprecated_have_wrong_rho()` in `test_task_identity.py` asserts `deprecated_random["rho"] == 0.0` with inline comment "# this is the bug". Test docstring also clarifies the deprecated value is wrong. No test treats rho=0 as correct. |
| 3 | No test asserts targeted uses gripper_logit_margin_cw as main protocol | **PASS** | `test_task_identity.py` lines 100-103: `test_targeted_is_not_margin_cw` explicitly asserts `!= "gripper_logit_margin_cw"`. `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL` references to `gripper_logit_margin_cw` are scoped to diagnostic/ablation context, with source code comments saying "NOT the legacy Codex main targeted attack" (condition_protocols.py line 200). |
| 4 | No test allows command_open rho=0 as valid | **PASS** | `TestCommandOpenProtocol.test_rho_positive` (task_identity.py), `TestCommandOpen.test_rho_positive` (clean_protocol.py), `TestCommandOpenValidation.test_rho_must_be_positive` (metadata_schema.py), and `TestCommandOpenRhoZeroRejected.test_rho_zero_rejected` (protocol_validation.py) all reject or demand >0. The deprecated entry is tested with explicit "disables oracle override" warning. |
| 5 | DEPRECATED / MATCHED_CONDITIONS fail-fast sentinel HAS tests | **PASS** | Two test classes cover the sentinel: `test_task_identity.py` `TestDeprecatedMatchedConditionsFailFast` (4 tests: iter, getitem, len, triage iter) and `test_protocol_validation.py` `TestDeprecatedMatchedConditionsFailFast` (5 tests: iter, getitem, len, bool, triage iter). All use `pytest.raises(RuntimeError, match="LEGACY_CODEX_STATE5")`. Total 9 tests across 2 files. Note: `__repr__` is not tested; `__bool__` tested only in `test_protocol_validation.py`. |
| 6 | Protocol validation tests use ProtocolValidationError (NOT AssertionError) | **PASS** | Every `pytest.raises` in `test_protocol_validation.py` catches `ProtocolValidationError`. Source `protocol_validation.py` line 12 confirms `class ProtocolValidationError(ValueError)` with docstring explicitly stating "NOT assert". |
| 7 | make_run_id and make_clean_detect_run_id imported from src.utils.task_identity (NOT from condition_protocols) | **PASS** | `test_task_identity.py` lines 8-12 imports both from `src.utils.task_identity`. `TestMakeRunIdImportsFromTaskIdentity` verifies `__module__ == "src.utils.task_identity"`. Source confirms both functions defined in `task_identity.py` (lines 151, 156). Grep of `condition_protocols.py` returns zero matches for either function. |
| 8 | Coverage: command_open rho>0, env_extra, same_seed, table1_prior_window, Codex targeted exact params | **PASS** | All areas covered: command_open rho>0 in 4 test classes across 3 files; env_extra in 3 tests (missing/rejected); same_seed protocol in `TestSameSeedProtocolRequired` (matching pass + mismatch raise); table1_prior_window rejection in `TestTable1PriorWindowRejected` (reject + accept); Codex targeted exact params in `TestCodexTargetedProtocolExact` (correct pass + 5 wrong-param rejections) plus coverage in `test_task_identity.py`. |
| 9 | All tests that don't require pytest.raises are runnable without pytest | **MINOR FAIL** | 8 of 9 files that don't use `pytest.raises` correctly avoid top-level `import pytest`. `test_guard.py` is the exception: it imports `pytest` at module level (line 3) but never uses `pytest.raises` or any pytest-only feature. All other non-pytest.raises files (`test_clean_protocol.py`, `test_metadata_schema.py`, `test_budget.py`, `test_directional.py`, `test_metrics.py`, `test_triggers.py`, `test_public_pipeline.py`, `test_runner_overrides.py`) have no top-level `import pytest`. |
| 10 | clean_detect protocol: attack_enabled=False, attack_objective="", rho=0.0 | **PASS** | Confirmed in both `test_clean_protocol.py` (`TestCleanProtocol`: 4 tests) and `test_task_identity.py` (`TestCleanDetectProtocol`: 3 tests). Source `condition_protocols.py` lines 20-38 match exactly: `"attack_enabled": False`, `"attack_objective": ""`, `"rho": 0.0`. |

---

## Test Coverage Gap Analysis

### Areas adequately covered

1. **Protocol identity enforcement** — `command_open`, `clean_detect`, Codex `random`, `VIS_current`, `targeted` all have exact-parameter assertions against their protocol dicts. Validation functions in `protocol_validation.py` have matched "passes" and "rejected" tests.

2. **Deprecated sentinel** — Both `MATCHED_CONDITIONS` and `TRIAGE_MATCHED_CONDITIONS` fail-fast behavior is tested for `__iter__` access. `MATCHED_CONDITIONS` additionally tested for `__getitem__`, `__len__`, and `__bool__`.

3. **Import integrity** — `make_run_id`/`make_clean_detect_run_id` module origin verified with `__module__` introspection.

4. **Clean_detect invariants** — Asserted in two independent test classes.

### Gaps found

1. **TRIAGE_MATCHED_CONDITIONS `__getitem__`, `__len__`, `__bool__` not tested** — Only `__iter__` is tested for `TRIAGE_MATCHED_CONDITIONS` (in both `test_task_identity.py` and `test_protocol_validation.py`). If semantically `TRIAGE_MATCHED_CONDITIONS` only needs iteration coverage, this is acceptable, but a complete coverage audit should note the asymmetry.

2. **`__repr__` of sentinel not tested** — The `_DeprecatedProtocolSentinel.__repr__` method returns a formatted message but has no test. This is cosmetic and low-risk.

3. **test_guard.py imports pytest unnecessarily** — It has a top-level `import pytest` but never uses `pytest.raises`, `tmp_path`, `monkeypatch`, or any pytest-only feature. The `import pytest` can simply be removed.

4. **No test for `validate_condition_config_schema` happy path (non-command_open, non-Codex conditions)** — The schema validator is tested for `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` (all pass) and for missing fields (rejected), but there is no test for a generic protocol that isn't command_open or Codex targeted (to ensure schema validation doesn't falsely require command_open/Codex-specific fields).

5. **No integration test verifying that `validate_command_open_protocol` is actually called at driver startup** — The validator exists and is unit-tested, but there is no test confirming the experiment driver imports and invokes it before launching rollouts.

---

## Specific Test Issues Found

### Failure: test_guard.py top-level `import pytest` without using pytest (Assertion 9)

- **File**: `tests/v4/test_guard.py`, line 3
- **Issue**: `import pytest` at module level, but none of the 14 test functions use `pytest.raises` or any pytest-only feature (no fixtures, no `tmp_path`, no `monkeypatch`, no `skipif`).
- **Impact**: Prevents the module from being imported or its functions called in a non-pytest context (e.g., direct `python test_guard.py`).
- **Fix**: Either remove the import (trivial) or add a `pytest.raises` test for an edge case if one is missing.

### Minor: TRIAGE_MATCHED_CONDITIONS sentinel coverage asymmetry

- **Files**: `tests/v4/test_task_identity.py`, `tests/v4/test_protocol_validation.py`
- **Issue**: Both test files test `MATCHED_CONDITIONS` for `__iter__`, `__getitem__`, `__len__`, and (in protocol_validation) `__bool__`, but only test `__iter__` for `TRIAGE_MATCHED_CONDITIONS`.
- **Severity**: Low — the sentinel class is the same for both variables.

### Minor: No `__repr__` test for sentinel

- `_DeprecatedProtocolSentinel.__repr__` returns a formatted diagnostic message but has no test. Cosmetic only.

### No issues with module imports

- All imports in all test files resolve to existing modules:
  - `src.utils.task_identity` / `condition_protocols` / `protocol_validation` — all exist under `src/utils/`
  - `gripper_attack.*` — all exist under `src/gripper_attack/`
  - `v4_run_eval_openvla` — exists at `scripts/v4_run_eval_openvla.py`
  - `pyproject.toml` sets `pythonpath = ["src", "scripts"]` so both paths are available during pytest runs.

### No tests reference old MATCHED_CONDITIONS without expecting RuntimeError

- Only two files reference `MATCHED_CONDITIONS` or `TRIAGE_MATCHED_CONDITIONS`: `test_task_identity.py` and `test_protocol_validation.py`. Both access them exclusively within `pytest.raises(RuntimeError)` blocks.

---

## Overall Verdict

**PASS with minor findings.** The test suite is comprehensive for protocol-drift regression prevention. No test silently accepts deprecated/wrong protocol values. The fail-fast sentinel for `MATCHED_CONDITIONS`/`TRIAGE_MATCHED_CONDITIONS` is well-tested (9 tests across 2 files). Protocol validation tests correctly use `ProtocolValidationError` throughout. The `make_run_id`/`make_clean_detect_run_id` functions are correctly imported from `task_identity` and verified by `__module__` introspection.

Two actionable findings:
1. Remove the unused `import pytest` from `test_guard.py` (line 3) — 30-second fix, eliminates the only assertion-9 deviation.
2. Consider adding `TRIAGE_MATCHED_CONDITIONS` `__getitem__`/`__len__`/`__bool__` sentinel tests for symmetry — optional, low priority.
