# Codex Stage-B v1.1 Readiness Audit R3

Date: 2026-06-07

Audited base commit: `05b381e531c70efe509c5f9aab06e49110fbd16e`

Mode: CPU-only static audit, py_compile, unit tests, and synthetic CSV postprocess/label-builder checks. No GPU, VIS batch, rollout worker, watcher, or server live output mutation was run.

## Verdict

`READY_FOR_SMALL_V1_1_SMOKE`

The three R2 blockers are closed in code after the Codex R3 patch in this audit worktree:

1. Runner supports `--pair_id`, and both worker generators now pass a deterministic shared pair ID from the VIS job to its matched random job.
2. `scripts/stageb/postprocess_traces_v1_1.py` is v1.1-only, rejects old trace versions, reads task/condition/window metadata from trace columns, and uses `step_dict[step + 1]` for shifted qpos.
3. `scripts/stageb/build_pair_labels_v1_1.py` is v1.1-only, pairs by `pair_id + task_key + state_id + seed + window_start + window_end`, rejects pre-v1.1 inputs, and now hard-fails duplicate/unpaired rows.

This is a code readiness result, not a scientific result. It does not validate Stage-B v1.1 task-level VIS effects. The next safe step is a 3-row v1.1 smoke using the corrected runner, validator, v1.1 postprocess, and v1.1 label builder.

## Check Results

| Check | Status | Evidence |
|---|---:|---|
| Runner imports `openvla_libero_exec_spec` | PASS | `scripts/run_stageb_vis_labeling.py` imports the executable spec version, official prompt helper, gripper semantic helpers, and official LIBERO image helper. |
| `prompt_style = official_in_out` | PASS | Runner sets `PROMPT_STYLE = 'official_in_out'` and calls `official_prompt(...)`. |
| `image_preprocess_style` recorded | PASS | Runner records official/legacy image preprocessing style; official path calls `get_libero_image_official(obs)`. |
| Trace schema complete | PASS | Runner writes the v1.1 trace schema with raw/env action vectors, qpos fields, task/window fields, `decoded_open_bool`, `open_convention`, git provenance, prompt/image metadata, and exec spec version. |
| Random metadata present | PASS_WITH_WARNING | Runner records random seed metadata, perturbation space, random Linf/L2 norms, and epsilon metadata. Random tensor generation should still be made explicitly replayable before claiming deterministic replay. |
| Shared pair ID orchestration | PASS_AFTER_CODEX_PATCH | Runner accepts `--pair_id`; `scripts/generate_stageb_worker_scripts.py` and `scripts/diagnostics/generate_stageb_worker_scripts.py` now pass the same `stageb_pair_<vis_job_id>` to VIS and matched random jobs. |
| Postprocess accepts only corrected traces | PASS | `postprocess_traces_v1_1.py` requires `trace_version == corrected_stageb_v1_1` and returns non-zero if old-format traces are present. |
| Postprocess reads trace columns, not filename | PASS | Metadata is read from trace rows: `pair_id`, `condition`, `task_key`, `state_id`, `seed`, `window_start`, `window_end`. |
| Shifted qpos indexing | PASS | Postprocess computes shifted qpos with `step_dict[s + 1]`. |
| Label builder hard-fails old labels | PASS | `build_pair_labels_v1_1.py` exits non-zero on any non-`corrected_stageb_v1_1` qpos input. |
| Label builder pair matching | PASS_AFTER_CODEX_PATCH | Builder pairs by full pair key and now hard-fails duplicate condition rows and unpaired rows instead of silently overwriting/skipping. |
| Token-region audit | PASS | Attack adapter uses decoded action `>= 0.5` and env action `< -0.5` as OPEN; tests cover saturation token classification. |
| Tests | PASS | Selected Stage-B v1.1 tests: `23 passed in 0.17s`. |
| Synthetic postprocess/label smoke | PASS | Synthetic valid pair produced 1 label row; old trace, duplicate pair, and unpaired pair each returned non-zero. |

## Validation Commands

```bash
python -m py_compile \
  scripts/run_stageb_vis_labeling.py \
  scripts/generate_stageb_worker_scripts.py \
  scripts/diagnostics/generate_stageb_worker_scripts.py \
  scripts/stageb/postprocess_traces_v1_1.py \
  scripts/stageb/build_pair_labels_v1_1.py \
  scripts/stageb/validate_stageb_trace_v1_1.py
```

Result: PASS.

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

Result: `23 passed in 0.17s`.

Synthetic CSV checks:

- Valid VIS/random v1.1 trace pair: postprocess rc=0, label builder rc=0, labels=1.
- Old `patched_stageb_v1` trace: postprocess rc=1.
- Duplicate VIS condition for same pair key: label builder rc=1.
- Unpaired VIS-only row: label builder rc=1.

## Remaining Warnings

- This audit did not execute model inference, rollout, VIS, random perturbation generation, or a server worker.
- Random Linf metadata is present, but random perturbation generation should use an explicitly seeded torch generator if deterministic replay is required.
- The v1.1 validator still focuses on trace schema and core semantic gates. Before a large run, it is reasonable to add condition-specific checks for random metadata and exact `prompt_style`, `image_preprocess_style`, `exec_spec_version`, and `unnorm_key`.

## Safe Next Step

DeepSeek can run a small Stage-B v1.1 smoke only after syncing this R3 patch:

1. Generate worker commands from the patched worker generator.
2. Run at most the approved 3-row v1.1 smoke.
3. Validate each trace with `scripts/stageb/validate_stageb_trace_v1_1.py`.
4. Run `scripts/stageb/postprocess_traces_v1_1.py`.
5. Run `scripts/stageb/build_pair_labels_v1_1.py`.
6. Do not mix old Stage-B labels or old patched traces into the v1.1 label set.

Full Stage-B batch should wait until the 3-row smoke passes trace validation, postprocess, label building, and manual inspection of pair IDs/open/qpos conventions.
