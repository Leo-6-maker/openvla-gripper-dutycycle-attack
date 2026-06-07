# Stage-B Full OpenVLA Alignment Audit

**Date**: 2026-06-07
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Scope**: CPU-only source audit and regression tests. No GPU, VIS, rollout, watcher, or live-output mutation.

## Official Sources Checked

- OpenVLA `experiments/robot/robot_utils.py`: `normalize_gripper_action(..., binarize=True)` normalizes raw gripper from `[0, 1]` to `[-1, +1]` and then applies `np.sign`; `invert_gripper_action` flips the sign for LIBERO.
- OpenVLA `experiments/robot/libero/run_libero_eval.py`: official LIBERO loop calls `get_action`, then `normalize_gripper_action`, then `invert_gripper_action`, then `env.step(action.tolist())`; state includes `obs["robot0_gripper_qpos"]`.
- OpenVLA `experiments/robot/openvla_utils.py`: non-`openvla-v01` checkpoints use the `In: ...\nOut:` prompt; `openvla-v01` uses the older chat-style branch.
- OpenVLA `experiments/robot/libero/libero_utils.py`: `get_libero_image` uses `obs["agentview_image"]`, rotates 180 degrees, then applies the Octo-style JPEG/resize path.

Reference URLs:

- <https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/robot_utils.py>
- <https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/run_libero_eval.py>
- <https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/openvla_utils.py>
- <https://raw.githubusercontent.com/openvla/openvla/main/experiments/robot/libero/libero_utils.py>

## Main Finding

The previous Stage-B v1.1 RC1 convention treated `raw_gripper == 0.5` as OPEN. That is not aligned with official OpenVLA because official binarization uses `np.sign(2 * raw - 1)`. Therefore:

```text
raw_gripper >  0.5 -> normalized +1 -> env_action_6 -1 -> physical OPEN
raw_gripper <  0.5 -> normalized -1 -> env_action_6 +1 -> physical CLOSE
raw_gripper == 0.5 -> normalized  0 -> env_action_6  0 -> boundary / neutral
```

This audit patched the executable spec, wrapper, attack token-region logic, postprocess open counter, docs, and tests to use the official boundary rule.

## Code Alignment Results

| Area | Result |
|---|---|
| Official action chain | PASS |
| Raw/env open semantics | PASS after patch |
| Boundary raw 0.5 handling | PASS after patch |
| Attack open token region | PASS after patch |
| Prompt style for current checkpoint | PASS |
| Qpos source | PASS |
| Pre-v1.1 trace quarantine | PASS |
| Image preprocessing | BOUNDARY: explicit `official_rot180_only`, not full official Octo resize |
| Legacy diagnostics | WARNING: stale comments/helpers remain outside Stage-B v1.1 main path |
| Server snapshot | BOUNDARY: isolated server validation copy passed; live reviewed worktree was not overwritten |

## Files Patched

- `src/gripper_attack/openvla_libero_exec_spec.py`
- `src/gripper_attack/gripper_semantics.py`
- `src/gripper_attack/attack_adapter.py`
- `scripts/run_stageb_vis_labeling.py`
- `scripts/stageb/postprocess_traces_v1_1.py`
- `scripts/stageb/validate_stageb_trace_v1_1.py`
- `tests/stageb/test_openvla_libero_exec_spec.py`
- `tests/stageb/test_attack_open_token_region.py`
- `tests/stageb/test_open_convention.py`
- `tests/stageb/test_stageb_open_count_convention.py`
- `tests/stageb/test_openvla_full_alignment.py`
- `reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md`
- `reports/STAGEB_V1_1_RC1_FREEZE_REPORT.md`
- `reports/STAGEB_V1_1_RC1_SERVER_SNAPSHOT_VERIFY.md`
- `tables/stageb_v1_1_rc1_file_manifest.csv`
- `tables/stageb_v1_1_rc1_test_results.csv`
- `tables/stageb_full_openvla_alignment_findings.csv`

## Validation

Local CPU validation:

```text
py_compile: PASS
pytest tests/stageb -q: 47 passed in 0.23s
```

Server CPU validation in isolated copy:

```text
copy: /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
env: /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
py_compile: PASS
pytest tests/stageb -q: 47 passed in 0.21s
SHA table: rows=40, missing=0, py310_fail=0
```

No GPU, VIS, rollout, watcher, or live experiment output was touched.

## Claim Boundary

- Stage-B main code is now aligned with the official OpenVLA action transform and boundary semantics.
- This is a code/readiness result only, not a VIS effectiveness result.
- `official_rot180_only` remains a partial image-preprocessing implementation. It is explicitly recorded and must not be described as full official Octo preprocessing.
- Existing 44-row patched rerun and old overnight labels remain quarantined as inverted-objective or pre-v1.1 evidence.
- New Stage-B runs should wait until the live server execution worktree is resynced to the RC1a source snapshot, or should run from the isolated validation copy after DeepSeek confirms it is the execution source.
