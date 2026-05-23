# Subagent A — Protocol Semantics Audit

## Summary

Audited 7 assertions across 6 source files covering protocol definitions, validation,
budget controller, triggers, and the experiment driver. All 7 assertions PASS. The
codebase has a clean separation between the legacy Codex state5 protocols and diagnostic
protocols, with fail-fast guards in `protocol_validation.py` catching all known drift
patterns. A small number of residual references to the old wrong protocol exist in
mark-deprecated structures and config files, none capable of affecting active experiments.

---

## Per-Assertion Results

| # | Assertion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `command_open` rho=0 IS REJECTED by `protocol_validation` | **PASS** | `validate_command_open_protocol()` (line 26) checks `if protocol["rho"] <= 0`. `validate_condition_config_schema()` (lines 152-155) also checks `if protocol.get("rho", 0) <= 0` when the objective contains "oracle_env_gripper_open". Test `test_protocol_validation.py::TestCommandOpenRhoZeroRejected::test_rho_zero_rejected` (line 58-61) confirms rejection with ProtocolValidationError matching "disables oracle override". |
| 2 | `command_open` objective MUST BE `oracle_env_gripper_open` | **PASS** | `COMMAND_OPEN_ORACLE_PROTOCOL` (line 152) defines `attack_objective: "oracle_env_gripper_open"`. `validate_command_open_protocol()` (line 30) raises if it differs. Test at line 63-66 confirms wrong objective is rejected. |
| 3 | `command_open` MUST HAVE `env_extra` with `V4_ORACLE_FORCE_GRIPPER_ENV_VALUE` | **PASS** | `COMMAND_OPEN_ORACLE_PROTOCOL` (lines 166-169) includes `env_extra: {V4_ORACLE_FORCE_GRIPPER_ENV_VALUE: "-1.0", V4_ORACLE_GRIPPER_PATTERN: "continuous_open"}`. `validate_command_open_protocol()` checks both the key existence (line 35-37) and value presence (line 39-42). `validate_condition_config_schema()` also checks both (lines 144-150). Tests at lines 68-78 confirm both missing-`env_extra` and missing-oracle-value cases are rejected. |
| 4 | `force_gripper_open_token_ce` and `gripper_logit_margin_cw` are CLEARLY DISTINGUISHED in `condition_protocols.py` | **PASS** | Three levels of distinction exist: (a) **Separate dicts** — `CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN` (lines 117-138) vs `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL` (lines 181-202). (b) **Different parameters** — eps=0.25/ss=0.050/asteps=60/fo=1.0 vs eps=0.10/ss=0.020/asteps=20/cw_margin=5.0/fo=None. (c) **Explicit notes** — "NOT gripper_logit_margin_cw" in targeted notes (line 136); "Uses Carlini-Wagner margin loss (logit gap), NOT token-CE" in diagnostic notes (lines 197-198). Tests in `test_metadata_schema.py` (lines 80-82) and `test_task_identity.py` (lines 92, 101-103) explicitly assert they are not the same. |
| 5 | `gripper_logit_margin_cw` is NOT the legacy Codex main targeted protocol (it is diagnostic only) | **PASS** | `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL` has `protocol_name: "diagnostic_gripper_margin"` (line 183) and `purpose: "diagnostic_ablation_gripper_logit_margin"` (line 185). Its notes explicitly state "Not the legacy Codex main targeted attack." (line 200). It is NOT included in `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` (lines 227-232) nor in `LEGACY_CODEX_STATE5_OPTIONAL` (lines 235-237), confirming it is not part of the confirmatory experiment set. The legacy targeted protocol is `CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN` with objective `force_gripper_open_token_ce`. |
| 6 | `random` and `VIS_current` ARE marked as legacy actual attacks (`is_attack=True`, `requires_execution_audit=True`), NOT no-attack controls | **PASS** | `CODEX_LEGACY_RANDOM_SAME_WINDOW`: `is_attack=True` (line 80), `requires_execution_audit=True` (line 72), `rho=1.0` (line 75), `epsilon=0.10` (line 77). `CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW`: `is_attack=True` (line 106), `requires_execution_audit=True` (line 98), `rho=1.0` (line 101), `epsilon=0.10` (line 103). Both have non-zero rho and linf perturbation, confirming they are actual attacks. Notes explicitly say "Not a clean-repeat control" (line 84) and "this is an actual attack, not a control" (line 111). The old deprecated versions in `task_identity.py` (lines 63-92) had the WRONG `is_attack=False, is_control=True` but are guarded by `_DeprecatedProtocolSentinel`. |
| 7 | `attack_objective=None` semantics is CLEAR: means OMIT CLI flag, not pass string "None" | **PASS** | `CODEX_LEGACY_RANDOM_SAME_WINDOW` defines `attack_objective: None` (line 68), `attack_objective_raw_arg: None` (line 69), and `omit_attack_objective_cli_arg: True` (line 70). The notes state: "attack_objective=None + omit_attack_objective_cli_arg=True means --attack_objective was NOT passed on CLI; effective objective resolved to 'targeted_directional_ce' via config default. Driver MUST NOT pass 'None' or empty string as --attack_objective value." (lines 85-89). `effective_attack_objective_expected: "targeted_directional_ce"` (line 71) documents the actual fallback. Test at `test_protocol_validation.py::TestNoneObjectiveOmitsCliFlag` (lines 157-170) confirms: `attack_objective is None`, `omit_attack_objective_cli_arg is True`, and `effective_attack_objective_expected` is neither "None" nor "". |

---

## Additional Findings

### A. `validate_condition_config_schema` catches command_open with rho=0

**YES.** Lines 143-155 of `protocol_validation.py` contain an explicit check block:
```python
if "oracle_env_gripper_open" in str(protocol.get("attack_objective", "")):
    ...
    if protocol.get("rho", 0) <= 0:
        raise ProtocolValidationError(
            f"{name}: command_open protocol rho must be > 0, got {protocol.get('rho')}"
        )
```
This guards against rho=0 at schema-validation time, in addition to the dedicated `validate_command_open_protocol()` function.

### B. Codex targeted protocol has the correct exact parameters

**YES.** `CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN` (lines 117-138) defines:
- `epsilon: 0.25`
- `step_size: 0.050`
- `attack_steps: 60`
- `force_open_raw_gripper: 1.0`

These are enforced by `validate_codex_targeted_protocol()` (lines 82-108) with tolerance `1e-9` for float comparisons. Tests at `test_protocol_validation.py` lines 107-135 confirm all deviations (wrong eps, wrong step_size, wrong attack_steps, wrong objective, wrong fo) are rejected.

### C. Remaining references to old wrong protocols

Several files contain references to the old wrong protocol (`gripper_logit_margin_cw` as the targeted objective rather than `force_gripper_open_token_ce`). None are actionable:

1. **`src/utils/task_identity.py` (lines 94-108)** — Contains `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS` with the old wrong params. This is explicitly guarded: the `_DeprecatedProtocolSentinel` class (lines 135-144) replaces `MATCHED_CONDITIONS` / `TRIAGE_MATCHED_CONDITIONS` and raises `RuntimeError` on any access. The deprecated list itself has `deprecated: True` and `deprecation_reason` fields explaining each error.

2. **`configs/paper_black_bowl_attack.yaml` (line 15)** — Defines `objective: gripper_logit_margin_cw`. This is NOT a condition protocol; it is a general attack optimizer config for paper ablation experiments. Whether this is intentional or stale cannot be determined from a static audit alone — it may be the correct objective for the paper's specific experimental design.

3. **`scripts/run_attack_pipeline.py` (lines 54, 64, 117, 129, 155, 167)** — Queue entries using `gripper_logit_margin_cw` for ablation/diagnostic runs. These are in a pipeline script, not in the condition protocol definitions.

4. **`reports/STATE5_EXACT_CODEX_EVIDENCE_FREEZE.md` (line 46)** — Documentation noting that older Codex runs used `gripper_logit_margin_cw` for ALL conditions (including targeted). This is a historical document, not an active config.

5. **`tables/state5_condition_outcome_compact.csv`** — CSV data from old runs with `gripper_logit_margin_cw`. Historical data, not active code.

6. **`src/utils/task_identity.py` — `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS` index 3 (lines 109-128)** — The old `command_open` protocol with `rho=0.0` and `asteps=0`. This is the exact drift pattern the validators are designed to catch. It is deprecation-guarded.

**Verdict on old references**: The deprecated definitions in `task_identity.py` are properly neutralized by `_DeprecatedProtocolSentinel`. The config file and pipeline references use `gripper_logit_margin_cw` in contexts where it may genuinely be the intended objective (diagnostic/ablation runs). No remaining reference can silently affect a confirmatory experiment using `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS`.

---

## Overall Verdict

**PASS**

All 7 asserted protocol semantics are correctly implemented and verified. The fail-fast validation guards (`validate_command_open_protocol`, `validate_condition_config_schema`, `validate_codex_targeted_protocol`) provide defense-in-depth against the known drift patterns. The distinction between main targeted protocol (`force_gripper_open_token_ce`) and diagnostic ablation (`gripper_logit_margin_cw`) is clean and documented. Old wrong protocol definitions are quarantined behind deprecation sentinels that raise on access.

### Issues by severity

- **BLOCKING**: None — all assertions pass.
- **HIGH**: None — no active code path uses incorrect protocol parameters.
- **MEDIUM**: Consider reviewing whether `configs/paper_black_bowl_attack.yaml` (line 15) truly intends `objective: gripper_logit_margin_cw` or should reference the canonical Codex targeted protocol. This is a single-line config file default and is not guarded by schema validation since it is a YAML config, not a protocol dict — a silent drift vector if the paper experiment intends to reproduce the Codex targeted attack.
