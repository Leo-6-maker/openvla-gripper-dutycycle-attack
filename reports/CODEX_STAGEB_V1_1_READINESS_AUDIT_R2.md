# Codex Stage-B v1.1 Readiness Audit R2

Date: 2026-06-07

Audited commit: `ab5d235675984738b8f9116142c138344747cd55`

Branch context: `exp/vis-prefix-margin-repair-20260603`

Mode: CPU-only static/code audit plus py_compile and unit tests. No GPU batch, VIS rerun, rollout worker, watcher, or server live output mutation was run.

## Verdict

`BLOCKED_FOR_END_TO_END_STAGEB_V1_1_LABELING`

The v1.1 runner itself is substantially hardened and now records the corrected OpenVLA/LIBERO execution semantics, official prompt/image preprocessing metadata, corrected open convention, raw/env action vectors, qpos fields, trace version, and git provenance. The token-region audit and pure-Python tests pass.

However, the committed tree is not ready for full Stage-B v1.1 paired labeling because the downstream postprocess and pair-label builder still use old v1 assumptions, and the worker script does not pass a shared `pair_id` into separate VIS/random runner invocations. A runner smoke can be useful, but v1.1 labels must not be built from the current postprocess/label-builder stack.

## Required Checks

| Check | Status | Evidence |
|---|---:|---|
| Runner imports `openvla_libero_exec_spec` | PASS | `scripts/run_stageb_vis_labeling.py` imports `OPENVLA_LIBERO_EXEC_SPEC_VERSION`, `official_prompt`, gripper semantic helpers, and `get_libero_image_official`. |
| Prompt style is official In:/Out: | PASS | Runner sets `PROMPT_STYLE = 'official_in_out'` and calls `official_prompt(instruction.lower())`. |
| Image preprocessing style is recorded | PASS | Runner records `image_preprocess_style`; official path uses `get_libero_image_official(obs)`. |
| Trace schema is complete | PASS | Runner writes the 52-column v1.1 trace schema, including raw/env actions, qpos fields, task/window fields, `decoded_open_bool`, `open_convention`, `trace_version`, `git_commit`, and execution metadata. |
| Random metadata is complete enough for trace audit | PARTIAL | Runner writes `random_seed`, `perturbation_space`, `random_noise_linf`, `random_noise_l2`, and epsilon metadata. It does not appear to seed the `torch.rand_like` generator explicitly from the recorded random seed. |
| Postprocess accepts only corrected traces | BLOCKER | `scripts/stageb/postprocess_patched_traces_v1.py` only checks for `obs_gripper_qpos_0`, does not require `trace_version == corrected_stageb_v1_1`, still parses task/condition from filename, and uses an incomplete pairing key. |
| Label builder hard-fails old labels | BLOCKER | `scripts/stageb/build_pair_labels_v1.py` still consumes `stageb_v1_windows_for_labels.csv`; docstring references `patched_stageb_v1`; no v1.1 trace-version/provenance hard gate is present. |
| Token-region audit proves OPEN tokens raw>=0.5/env=-1 | PASS | `src/gripper_attack/attack_adapter.py` uses decoded action `>= 0.5` plus `env_gripper_is_open`; tests assert saturation tokens are classified by decoded physical env sign. |
| Tests pass | PASS_WITH_SCOPE_CAVEAT | `22 passed in 0.23s`. Caveat: some tests assert intended logic with toy snippets and do not exercise the actual old v1 postprocess/label-builder scripts. |

## Blocking Issues

1. `pair_id` is generated inside each runner process.

   `scripts/run_stageb_vis_labeling.py` sets `pair_id = str(uuid.uuid4())[:8]`. `scripts/diagnostics/generate_stageb_worker_scripts.py` launches VIS and random as separate process invocations and does not pass a shared `--pair_id`. Unless another uninspected orchestration layer injects shared IDs, matched VIS/random traces will not share `pair_id`.

2. Postprocess is still old v1 and not v1.1-safe.

   `scripts/stageb/postprocess_patched_traces_v1.py` does not require `corrected_stageb_v1_1`, reads task from filename fragments, does not use summary JSON as the source of truth, pairs by `(task, window_start, window_end)` only, and computes shifted qpos from local `enumerate(att)` indices rather than `step_dict[step + 1]`.

3. Pair-label builder is still old v1 and does not quarantine old labels.

   `scripts/stageb/build_pair_labels_v1.py` reads `stageb_v1_windows_for_labels.csv` without a v1.1 trace-version/provenance gate and does not hard-fail old label inputs.

## Non-Blocking Warnings

- The validator hard-fails missing required columns, wrong trace version, wrong qpos source, wrong open convention, placeholder git commit, bad decoded-open values, and inconsistent pair ID within a trace. It does not yet hard-fail unexpected `prompt_style`, `image_preprocess_style`, `exec_spec_version`, `unnorm_key`, or missing random metadata conditional on `condition == random_linf`.
- `image_preprocess_style = official_rot180_only` is recorded. This confirms the official image path is marked, but the audit did not run a visual equivalence check.
- Random perturbation metadata is recorded, but the source code path should explicitly seed the torch random generator or record that the random perturbation is non-replayable.

## Validation Run

Command:

```bash
python -m py_compile \
  scripts/run_stageb_vis_labeling.py \
  scripts/stageb/validate_stageb_trace_v1_1.py \
  scripts/stageb/postprocess_patched_traces_v1.py \
  scripts/stageb/build_pair_labels_v1.py \
  src/gripper_attack/attack_adapter.py \
  src/gripper_attack/openvla_libero_exec_spec.py
```

Result: PASS.

Command:

```bash
PYTHONPATH=src python -m pytest \
  tests/stageb/test_trace_schema_v1_1.py \
  tests/stageb/test_old_label_quarantine.py \
  tests/stageb/test_random_linf_metadata.py \
  tests/stageb/test_pair_label_builder_v1_1.py \
  tests/stageb/test_attack_open_token_region.py \
  tests/stageb/test_openvla_libero_exec_spec.py \
  -q
```

Result: `22 passed in 0.23s`.

## Readiness Gate

Recommended status:

- Runner v1.1 static readiness: `PASS_WITH_WARNINGS`
- Validator static readiness: `PASS_WITH_WARNINGS`
- Token-region/open convention: `PASS`
- Pairing/postprocess/label generation readiness: `BLOCKED`
- Full Stage-B v1.1 batch readiness: `BLOCKED`

Minimum fixes before consuming v1.1 labels:

1. Add a runner CLI `--pair_id` or worker-side shared pair-id injection, and ensure the same `pair_id` is used for matched VIS/random traces.
2. Replace or patch postprocess so it accepts only `trace_version == corrected_stageb_v1_1`, reads task/condition/window from trace/summary fields, uses `pair_id + task/state/seed/window`, and computes shifted qpos via `step_dict[step + 1]`.
3. Replace or patch pair-label builder so it hard-fails old v1 labels/traces and consumes only corrected v1.1 postprocess outputs.
4. Extend tests to execute the actual postprocess and pair-label builder scripts, not only toy logic snippets.

## Claim Boundary

The current committed tree supports the claim that the Stage-B v1.1 runner was hardened against the major OpenVLA/LIBERO semantic regressions. It does not yet support a claim that Stage-B v1.1 paired labels are ready for scientific use.
