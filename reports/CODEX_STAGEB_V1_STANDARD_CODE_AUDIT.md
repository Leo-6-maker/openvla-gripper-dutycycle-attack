# Codex Stage-B v1 Standard Code Audit

Date: 2026-06-07

Scope: CPU/read-only audit of Stage-B v1 standard code plus a minimal LIBERO env-only open-convention smoke. No OpenVLA model, no PGD, no VIS, no rollout, no worker interruption, and no live output mutation beyond this audit report/table.

## Verdict

Stage-B v1 should be treated as **alpha**, not a reusable no-bug standard. The runner is directionally improved, but postprocess, label builder, validator, provenance, and tests need P0/P1 hardening before standard reuse.

The most important update from live verification is:

```text
LIBERO env_action_6 < -0.5 means physical OPEN
LIBERO env_action_6 >  0.5 means physical CLOSE
```

This was verified without loading OpenVLA by forcing gripper actions in a LIBERO object env and comparing both `obs["robot0_gripper_qpos"]` abs_sum and finger-pad geom distance.

## Open Convention Smoke

Command environment: `CUDA_VISIBLE_DEVICES=` with `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa`; official env python `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`; no model imports and no PGD/VIS.

| Forced env_action_6 | qpos_abs_sum start -> end | finger pad distance start -> end | Interpretation |
|---:|---:|---:|---|
| +1.0 | 0.041666 -> 0.003818 | 0.048666 -> 0.010818 | closes |
| -1.0 | 0.041666 -> 0.076226 | 0.048666 -> 0.083226 | opens |

Therefore the standard shared function should be:

```python
def is_env_open(env_action_gripper):
    return env_action_gripper < -0.5
```

## Runner Audit

Positive points in `scripts/stageb/run_paired_vis_random_v1.py`:

- requires `CUDA_VISIBLE_DEVICES`
- accepts `--pair_id`
- reads qpos from `obs.get("robot0_gripper_qpos")`
- writes `env_action_0..6`
- writes `obs_gripper_qpos_0/1`, `qpos_source`, `trace_version`, `runner_version`, `git_commit`
- measures `qpos_after` after `env.step` for summary qpos_delta
- current source has `is_env_open(grip_val) -> grip_val < -0.5`, consistent with the env-only smoke

Blocking runner gaps:

- missing per-step `raw_action_0..6`
- missing per-step `row_id`, `condition`, `task_key`, `state_id`, `seed`, `window_start`, `window_end`
- uses `attack_this_step`; required standard name `attack_active` is missing
- missing `decoded_open_bool`
- missing `open_convention`
- missing `obs_gripper_qpos_abs_sum` and `obs_gripper_qpos_abs_mean`
- `GIT_COMMIT = "stageb_v1"` is a placeholder, not real provenance

## Postprocess Audit

`scripts/stageb/postprocess_patched_traces_v1.py` is not safe for standard use:

- P0: open count/streak use `env_action_6 > 0.5`, which is reversed. Smoke confirms OPEN is `< -0.5`.
- P0: condition parsing splits the filename on `_`, so `vis_pgd` and `random_linf` are not recovered as single tokens.
- P0: task parsing via `fname.split("_")[1]` breaks for underscore tasks such as `tomato_sauce`, `bbq_sauce`, and `salad_dressing`.
- P0: shifted qpos uses `enumerate(att)` local index, not original step index; for windows not starting at step 0 it reads the wrong rows.
- P0: pairing key `(task, ws, we)` ignores `pair_id`, `state_id`, and `seed`, so it can mix episodes.

Required postprocess rule: read `pair_id`, `condition`, `task_key`, `state_id`, `seed`, `window_start`, `window_end`, and convention from summary/trace columns, not from filename guesses.

## Label Builder Audit

`scripts/stageb/build_pair_labels_v1.py` is not ready for trusted labels. It consumes aggregate postprocess rows and does not independently verify:

- `trace_version == patched_stageb_v1`
- `qpos_source == obs_robot0_gripper_qpos`
- matched VIS/random `pair_id`
- task/state/seed/window match
- no INFRA/OOM/manual/polluted rows
- random-confounded exclusion from valid matched controls

Because it inherits postprocess open/streak and shifted-qpos fields, label generation must be blocked until postprocess is fixed and validated.

## Validator And Tests

`scripts/stageb/validate_stageb_outputs_v1.py` is too permissive. It should fail if any standard trace lacks:

- `raw_action_0..6`
- `row_id`, `pair_id`, `condition`, `task_key`, `state_id`, `seed`, `window_start`, `window_end`
- `attack_active`, `decoded_open_bool`, `open_convention`
- `obs_gripper_qpos_abs_sum`, `obs_gripper_qpos_abs_mean`
- real `git_commit`, `git_dirty`, `runner_script_sha`

Current `tests/stageb/test_qpos_convention.py` is useful but partial. Add tests for open convention, shifted qpos indexing, schema completeness, and pair label construction.

## Current 44-row Rerun Boundary

The currently running 44-row selective rerun should not be stopped. However, its outputs should not be treated as standard v1 labels through the current postprocess/label builder. For temporary readout, use a hotfix postprocess that reconstructs metadata from summary + queue + trace, pairs by row/pair identity, uses `env_action_6 < -0.5` for OPEN, and computes shifted qpos by `step + 1`.

## Files

- Findings table: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/codex_stageb_v1_standard_code_findings.csv`
- Report: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/reports/CODEX_STAGEB_V1_STANDARD_CODE_AUDIT.md`
