# Codex OpenVLA Official Semantics Audit

**Date**: 2026-06-07
**Mode**: CPU-only code/spec audit
**Branch**: `exp/vis-prefix-margin-repair-20260603`

## Verdict

The OpenVLA-LIBERO gripper execution standard is now frozen in
`src/gripper_attack/openvla_libero_exec_spec.py`.

The corrected standard is:

```text
raw_gripper >= 0.5 -> env_action_6 = -1.0 -> physical OPEN
raw_gripper <  0.5 -> env_action_6 = +1.0 -> physical CLOSE
env_action_6 < -0.5 -> physical OPEN
env_action_6 > +0.5 -> physical CLOSE
```

This matches the official OpenVLA LIBERO execution chain:
`normalize_gripper_action(..., binarize=True)` followed by
`invert_gripper_action(...)` before `env.step(...)`.

## Files Created

- `reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md`
- `src/gripper_attack/openvla_libero_exec_spec.py`
- `tests/stageb/test_openvla_libero_exec_spec.py`
- `reports/CODEX_OPENVLA_OFFICIAL_SEMANTICS_AUDIT.md`
- `tables/codex_openvla_official_semantics_findings.csv`

## Files Patched

- `src/gripper_attack/gripper_semantics.py`
- `src/gripper_attack/attack_adapter.py`
- `scripts/run_stageb_vis_labeling.py`
- `scripts/stageb/postprocess_patched_traces_v1.py`
- `scripts/diagnostics/run_patched_rerun_postprocess_hotfix.py`
- `scripts/diagnostics/run_policy_only_vis_audit.py`
- `scripts/run_clean_trace_recovery.py`
- `scripts/run_official_eval_artifact_rich.py`
- `scripts/vis_rollout_proprionostep_triggered.py`
- `scripts/run_active_probe_v1_temporal.py`
- `scripts/run_active_probe_v1_pgd_budget_diagnostic.py`
- `scripts/diagnostics/extract_online_window_features.py`
- `scripts/diagnostics/audit_clean_phase_events.py`
- `scripts/online_detector_window_proposals.py`
- `scripts/diagnostics/build_clean_phase_dataset.py`
- `tests/stageb/test_open_convention.py`
- `tests/stageb/test_attack_open_token_region.py`
- `tests/stageb/test_stageb_open_count_convention.py`
- `tests/v4/test_gripper_semantics_consistency.py`

## Important Fixes

1. `gripper_semantics.py` is now a compatibility wrapper over the executable
   spec. New code should import `openvla_libero_exec_spec.py` directly.

2. `attack_adapter.py` now builds OPEN token regions from decoded raw gripper
   `>= 0.5`, not the inverted old `decoded_action < 0.5` rule. Runtime
   assertions verify raw and env-space classifications agree, and saturation
   tokens are classified by decoded physical sign.

3. Stage-B open counters and postprocessors now use `env_gripper_is_open`, whose
   rule is `env_action_6 < -0.5`.

4. `oracle_open` and `sustained_command_open_proxy` overrides in audited runners
   now write `ENV_GRIPPER_OPEN_VALUE = -1.0`, not `+1.0`.

5. Policy-only VIS audit no longer treats raw `<0.5` as OPEN; its target gripper
   is raw `1.0`, and its open detection uses `raw_gripper_is_open`.

## Remaining Risks

- Some legacy scripts still contain old wording such as `raw_gripper < 0.5 =
  OPEN` in docstrings or generated report text. These are listed in
  `tables/codex_openvla_official_semantics_findings.csv` and must not be used
  for current labels without rewrite/regeneration.

- Several legacy runners still use `sim.data.qpos[-2:]`. The executable spec
  requires `obs["robot0_gripper_qpos"]` for new label-quality qpos evidence.

- The 44-row patched rerun and older overnight labels were produced before this
  spec freeze or under suspect inverted objective semantics. They should remain
  `QUARANTINED_OPEN_SEMANTICS_INVERTED_OR_UNVERIFIED` and may only be cited as
  bug-discovery / inverted-objective diagnostic evidence.

## Validation

`py_compile`:

```text
PASS
```

Compiled:

- `src/gripper_attack/openvla_libero_exec_spec.py`
- `src/gripper_attack/gripper_semantics.py`
- `src/gripper_attack/attack_adapter.py`
- `scripts/run_stageb_vis_labeling.py`
- `scripts/stageb/postprocess_patched_traces_v1.py`
- `scripts/diagnostics/run_patched_rerun_postprocess_hotfix.py`
- `scripts/diagnostics/run_policy_only_vis_audit.py`
- `scripts/vis_rollout_proprionostep_triggered.py`
- `scripts/run_official_eval_artifact_rich.py`

Tests:

```text
D:\Users\anaconda\python.exe -m pytest \
  tests/stageb/test_openvla_libero_exec_spec.py \
  tests/stageb/test_open_convention.py \
  tests/stageb/test_attack_open_token_region.py \
  tests/stageb/test_stageb_open_count_convention.py \
  tests/v4/test_gripper_semantics_consistency.py -q

16 passed in 0.24s
```

## Boundary

No GPU, VIS batch, rollout worker, watcher, or server live output mutation was
performed in this audit.

Do not use old Stage-B labels or the 44-row patched rerun as VIS selectivity
evidence. The next valid experiment step is a corrected-objective smoke under
this executable spec, with official prompt, official image preprocessing,
patched action/qpos logging, and matched random control.
