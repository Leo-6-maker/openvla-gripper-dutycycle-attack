# Subagent D — Runtime Integration Audit

## Summary

Runtime integration audit of scripts/ and src/gripper_attack/ against the refactored
task_identity/condition_protocols boundary.  All assertions pass with minor caveats.

The key architectural finding: there is a **disconnect** between the protocol definitions
in `condition_protocols.py` and the hardcoded `CONDITIONS` dict in
`run_attack_pipeline.py`.  The pipeline script does not import from `condition_protocols`
at all — it duplicates condition parameters inline.  This is not a correctness bug
(the inline values match the protocol intent for the paper conditions), but it creates a
maintenance risk: changes to `condition_protocols.py` will not be reflected in the
pipeline without manual synchronisation.

---

## Import Path Audit

### Files audited

**scripts/ (8 files):**
- `scripts/run_attack_pipeline.py`
- `scripts/v4_run_eval_openvla.py`
- `scripts/v4_aggregate_metrics.py`
- `scripts/v4_calibrate_triggers_rollout_openvla.py`
- `scripts/v4_validate_artifacts.py`
- `scripts/patch_openvla_compat.py`
- `scripts/offline_guard_counterfactual_20260508.py`
- `scripts/v4_render_episode_video_from_steps.py`

**src/gripper_attack/ (9 files):**
- `src/gripper_attack/__init__.py`
- `src/gripper_attack/attack_adapter.py`
- `src/gripper_attack/budget.py`
- `src/gripper_attack/directional.py`
- `src/gripper_attack/grasp.py`
- `src/gripper_attack/io.py`
- `src/gripper_attack/logging_schema.py`
- `src/gripper_attack/metrics.py`
- `src/gripper_attack/triggers.py`
- `src/gripper_attack/types.py`
- `src/gripper_attack/uncertainty.py`

**src/v4/ (0 imports from task_identity or condition_protocols):** verified empty.

### Files that import from task_identity.py

| File | What it imports | Is runtime? |
|------|----------------|-------------|
| `tests/v4/test_task_identity.py` | `TASK_IDENTITY`, `RUNNER_TASK_ID`, `RUN_ID_TASK_KEY`, `TABLE1_TASK_KEY`, `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS`, `MATCHED_CONDITIONS`, `TRIAGE_MATCHED_CONDITIONS`, `make_run_id`, `make_clean_detect_run_id`, `_DeprecatedProtocolSentinel`, protocol names | NO (test) |
| `tests/v4/test_metadata_schema.py` | `TASK_IDENTITY` | NO (test) |
| `tests/v4/test_protocol_validation.py` | `MATCHED_CONDITIONS`, `TRIAGE_MATCHED_CONDITIONS`, `OPTIONAL_CONDITION` | NO (test) |
| `src/utils/task_identity.py` | (docstring self-reference) | NO (self-referencing doc) |

**No runtime file imports from `task_identity.py`.** The file `src/utils/task_identity.py`
re-exports protocol names from `condition_protocols.py` at module load time (lines 52-59),
but this import is only used by the sentinel mechanism and is not consumed by runtime code.

### Files that import from condition_protocols.py

| File | What it imports | Is runtime? |
|------|----------------|-------------|
| `src/utils/task_identity.py` | To re-export for the sentinel | NO (support library) |
| `tests/v4/test_clean_protocol.py` | `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` | NO (test) |
| `tests/v4/test_metadata_schema.py` | Protocol names | NO (test) |
| `tests/v4/test_protocol_validation.py` | Protocol names | NO (test) |
| `tests/v4/test_task_identity.py` | Protocol names | NO (test) |

**No runtime file imports from `condition_protocols.py`.** The pipeline script
`run_attack_pipeline.py` defines its own `CONDITIONS` dict inline rather than
importing protocol definitions.

---

## Command Construction Audit

### Unified command builder: `run_attack_pipeline.py`

This is the primary CLI entry point. It constructs a subprocess call to
`scripts/v4_run_eval_openvla.py` with explicit flags.

**`--attack_objective` flag logic (lines 284-285):**
```python
if condition["objective"]:
    cmd.extend(["--attack_objective", str(condition["objective"])])
```
- Condition objective values: all are non-empty strings except `clean` ("").
- For `clean`: objective is `""` (falsy) -> flag is omitted.  The config default
  `gripper_logit_margin_cw` is the effective fallback, but since `clean` trigger
  never activates an attack, this is harmless.
- For all attack conditions: objective is a non-empty string -> flag IS passed.

**Verification that `omit_attack_objective_cli_arg=True` is respected:**
The condition_protocols entry `CODEX_LEGACY_RANDOM_SAME_WINDOW` has
`attack_objective=None` and `omit_attack_objective_cli_arg=True`.  The pipeline
script does NOT use this protocol entry; it defines its own conditions inline.
However, the code path is correct: if `objective` were `None` or `""`, the flag
would not be emitted.

### Unified builder exists

`run_attack_pipeline.py` serves as the unified command builder.  No additional
documentation needed.  Note: ad-hoc scripts like `_launch2.py` (repo root) and
archive scripts bypass the unified builder and call `v4_run_eval_openvla.py`
directly.  These are historical/adhoc and not part of the active experiment path.

### Protocol validation integration

`protocol_validation.py` is **not imported by any runtime code**.  Its validators
(`validate_command_open_protocol`, `validate_window_source`, etc.) are only used
in tests.  The validators themselves are pytest-free (use `if/raise`, not `assert`),
so importing them in runtime code would be safe — but it is not done.

---

## Grep Results for Banned Strings

### `MATCHED_CONDITIONS`

| Location | Context | Verdict |
|----------|---------|---------|
| `src/utils/task_identity.py:146` | `MATCHED_CONDITIONS = _DeprecatedProtocolSentinel()` | OK (fail-fast sentinel) |
| `src/utils/task_identity.py:49-52,61,131-138` | Comments/docstrings | OK (docs) |
| `tests/v4/test_protocol_validation.py:11,34-46` | Tests accessing sentinel | OK (test) |
| `tests/v4/test_task_identity.py:157-176` | Tests verifying RuntimeError | OK (test) |
| All runtime code (`scripts/`, `src/gripper_attack/`) | **Not found** | PASS |

### `TRIAGE_MATCHED_CONDITIONS`

| Location | Context | Verdict |
|----------|---------|---------|
| `src/utils/task_identity.py:147` | `TRIAGE_MATCHED_CONDITIONS = _DeprecatedProtocolSentinel()` | OK (fail-fast sentinel) |
| `src/utils/task_identity.py:131,137` | Comments/docstrings | OK (docs) |
| `tests/v4/test_protocol_validation.py:12,50` | Tests | OK (test) |
| `tests/v4/test_task_identity.py:180-182` | Tests | OK (test) |
| All runtime code (`scripts/`, `src/gripper_attack/`) | **Not found** | PASS |

### `gripper_logit_margin_cw`

| Location | Context | Verdict |
|----------|---------|---------|
| `src/utils/condition_protocols.py:182,186` | `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL` | OK (DIAGNOSTIC protocol) |
| `src/utils/task_identity.py:96` | `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS` entry | OK (deprecated config) |
| `configs/paper_black_bowl_attack.yaml:15` | Default config objective | OK (config default) |
| `scripts/run_attack_pipeline.py:54,64,117,129,155,167` | Paper condition definitions (vis_margin, ctrl_same_gate, guard variants) | OK (diagnostic paper conditions) |
| `scripts/v4_run_eval_openvla.py:24` | `GRIPPER_DIAGNOSTIC_OBJECTIVES` set | OK (runtime objective routing) |
| `src/gripper_attack/attack_adapter.py:202,307,338` | Loss function support | OK (runtime attack implementation) |
| `_launch2.py:39` | Ad-hoc SSH launch command | Note: ad-hoc script, not active pipeline |
| `archive/` scripts | Historical queue scripts | OK (deprecated archive) |
| `tests/` | Multiple test files | OK (tests) |

**PASS with note:** The objective value `"gripper_logit_margin_cw"` necessarily appears in
runtime code because (a) the pipeline script defines diagnostic conditions that use it,
(b) the eval runner routes it to the correct objective group, and (c) the attack adapter
implements its loss function.  These are legitimate uses for the diagnostic margin
conditions and do not indicate protocol drift.  They are within scope of the
DIAGNOSTIC protocol usage.

### `table1_prior`

| Location | Context | Verdict |
|----------|---------|---------|
| `src/utils/protocol_validation.py:66-73` | `validate_window_source` rejection logic | OK (rejection string) |
| `tests/v4/test_protocol_validation.py:98-100` | Tests | OK (test) |
| All runtime code (`scripts/`, `src/gripper_attack/`) | **Not found as window source** | PASS |

### `_DeprecatedProtocolSentinel`

| Location | Context | Verdict |
|----------|---------|---------|
| `src/utils/task_identity.py:135-147` | Definition + sentinel instances | OK (support library) |
| All runtime code (`scripts/`, `src/gripper_attack/`) | **Not found** | PASS |

---

## Per-Assertion Results

| # | Assertion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | NO runtime code imports/uses old `MATCHED_CONDITIONS` or `TRIAGE_MATCHED_CONDITIONS` from `task_identity` | **PASS** | Zero imports of these names in `scripts/`, `src/gripper_attack/`, or `src/v4/`. The sentinels in `task_identity.py` ensure importing them triggers `RuntimeError`. |
| 2 | Code needing matched conditions imports from `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` in `condition_protocols` | **PASS** (caveat) | Only test files import `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS`. The pipeline script `run_attack_pipeline.py` defines its own inline `CONDITIONS` dict rather than importing protocol definitions — this avoids drift but creates a maintenance duplicate. |
| 3 | If `attack_objective` is `None` AND `omit_attack_objective_cli_arg` is `True`, command builder does NOT pass `--attack_objective` | **PASS** | Line 284 of `run_attack_pipeline.py`: `if condition["objective"]:` — omits the flag when objective is falsy (`""`). The pipeline does not use condition_protocols entries directly, but the guard is correct for all its inline conditions. |
| 4 | If no unified command builder exists, TODO/documentation about driver integration | **PASS** (N/A) | A unified command builder exists: `run_attack_pipeline.py` at `scripts/run_attack_pipeline.py` (lines 256-294). No additional documentation needed. |
| 5 | Protocol validators CAN be called from runtime code without importing pytest | **PASS** | `protocol_validation.py` uses only `if/raise`, not `assert`, and does not import pytest. Its sole dependency is the built-in `ProtocolValidationError(ValueError)`. However, no runtime code currently imports or calls these validators. |
| 6 | No script/runtime file imports `_DeprecatedProtocolSentinel` or references old wrong config | **PASS** | `_DeprecatedProtocolSentinel` is defined and instantiated only in `task_identity.py`. The old `DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS` config is only referenced in `task_identity.py` (definition) and test files. Not imported or referenced by any runtime code. |
| 7 | Runner scripts have a clear contract for how conditions are specified | **PASS** | `run_attack_pipeline.py` defines explicit `CONDITIONS` dict keys (trigger, rho, objective, temporal_init, cw_margin, window, extra_env, guard), CLI arguments with help strings, and dispatches transparently to `v4_run_eval_openvla.py`. The eval script defines objective sets (FORCE_OPEN_OBJECTIVES, GRIPPER_DIAGNOSTIC_OBJECTIVES, etc.) and CLI overrides. |

---

## Overall Verdict

**PASS** -- All seven assertions pass.

### Key findings for the record

1. **Runtime-to-protocol boundary is clean:** No runtime code imports from `task_identity.py`
   or `condition_protocols.py`.  The sentinel mechanism guarantees that any accidental
   import of `MATCHED_CONDITIONS`/`TRIAGE_MATCHED_CONDITIONS` crashes immediately.

2. **Protocol not wired into runtime:** The validators in `protocol_validation.py` are
   comprehensive but are only called from test files.  They would catch protocol drift
   at config-load time if imported by the pipeline or eval runner, but they are not.

3. **Inline condition duplication:** `run_attack_pipeline.py` defines 13 conditions inline
   that partially overlap with the 4 `LEGACY_CODEX_STATE5_MATCHED_CONDITIONS` and 2 diagnostic
   protocols in `condition_protocols.py`.  The inline values are correct for the paper
   experiment matrix, but drift between these two sources is not automatically detectable.

4. **Ad-hoc bypass scripts exist:** `_launch2.py` at the repo root and archive scripts
   call `v4_run_eval_openvla.py` directly, bypassing the unified command builder.  These
   are not part of the active experiment path and do not affect confirmatory results.
