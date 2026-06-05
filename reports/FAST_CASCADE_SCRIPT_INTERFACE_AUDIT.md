# Fast Cascade Script Interface Audit

Date: 2026-06-05

Scope: static audit plus CPU-only dry-run/interface checks. No GPU, rollout, VIS, watcher, or detector v2 training was started.

## Summary

Status: PARTIAL PASS WITH FAST OUTPUT BLOCKERS

- `scripts/diagnostics/run_policy_only_vis_audit.py`: patched for dry-run safety, local OpenVLA model loading, explicit `gpu_pair`, `runtime_sec`, `provenance_status`, `denominator_status`, `label_source`, and `label_confidence` fields.
- `scripts/diagnostics/run_command_open_proxy_replay.py`: patched for local OpenVLA model loading, fixed LIBERO Object task mapping, stable `OffScreenRenderEnv` setup, MuJoCo-primary gripper qpos measurement, and final env-step OPEN injection.
- `scripts/diagnostics/generate_phase_e_aligned_windows.py`: added CPU-only phase-aligned subwindow generator. It enumerates L8/L10/L12 windows and recommends only qpos-supported true_closed or transitional-pre-open candidates.
- `scripts/diagnostics/run_phase_e_canary.py`: patched for explicit candidate CSV windows, official action transform before every `env.step()`, action-transform provenance, MuJoCo-primary qpos audit fields, mechanism guard fields, and epsilon calibration provenance. Phase E canary v0 remains `INVALID_ACTION_SPACE_CONFOUNDED`.
- `scripts/diagnostics/compare_fast_vis_to_full_labels.py`: implemented. Current comparison remains blocked only because Fast VIS output CSVs are missing.
- `scripts/vis_phase_conditioned_attack.py`: patched unsafe default from `6,7` to `1,0` and added GPU3/GPU7/CUDA_VISIBLE_DEVICES guard.
- `scripts/vis_rollout_adaptive_v3.py`: added GPU3/GPU7/CUDA_VISIBLE_DEVICES guard. It still relies on physical GPU IDs in `max_memory` and `render_gpu_device_id`.

## Per-Script Findings

### scripts/diagnostics/run_policy_only_vis_audit.py

Findings before patch:

- `--gpu-pair` default was `0,1`.
- Script previously set `CUDA_VISIBLE_DEVICES` internally from `--gpu-pair`.
- Output did not explicitly record all audit fields needed by Fast cascade schema.
- `--dry-run` would still proceed toward model loading because the dry-run branch was absent.
- `find_cached_clean_image(...)` referenced `window_end` but did not receive it as an argument.
- Report candidate positive/negative counts used only the last loop variable.
- Commit `c17e2c9` still used a HuggingFace model id path that could trigger network access.

Patch applied:

- Added `validate_gpu_pair()`:
  - rejects GPU3/GPU7.
  - rejects `CUDA_VISIBLE_DEVICES=2,6` combined with `--gpu-pair 2,6`.
- Removed internal `CUDA_VISIBLE_DEVICES` remapping; model loading now treats `--gpu-pair` as physical GPU IDs.
- Added dry-run branch before model loading.
- Delayed `numpy`/`torch` imports so CPU-only dry-run works outside the official env.
- Fixed `find_cached_clean_image(..., window_end, ...)`.
- Added output fields: `gpu_pair`, `runtime_sec`, `provenance_status`, `denominator_status`, `label_source`, `label_confidence`.
- Added infra error classification for Xid/OOM/CUDA/CUBLAS failures.
- Model loading now uses `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` and `local_files_only=True` where supported.

Remaining caveat:

- This is a policy-only proxy screen. It must not be reported as full VIS success or gold task-level evidence.

### scripts/diagnostics/run_command_open_proxy_replay.py

Findings before patch:

- `--gpu-pair` default was `0,1`.
- Script previously set `CUDA_VISIBLE_DEVICES` internally from `--gpu-pair`.
- Output lacked explicit `gpu_pair`, `denominator_status`, `label_source`, and `label_confidence`.
- Top-level `numpy` import made dry-run depend on the experiment env.
- Commit `c17e2c9` still had P0 issues found by DeepSeek:
  - model loading used `"openvla/openvla-7b"` and could access HuggingFace.
  - task matching depended on unstable task strings.
  - environment setup depended on `task_suite.env`.
  - gripper qpos could silently use an invalid zero-valued obs path.

Patch applied:

- Added `validate_gpu_pair()` with the same GPU3/GPU7 and `CUDA_VISIBLE_DEVICES=2,6` guard.
- Removed internal `CUDA_VISIBLE_DEVICES` remapping; model loading now treats `--gpu-pair` as physical GPU IDs.
- Delayed `numpy` import to the rollout function.
- Added output fields: `gpu_pair`, `runtime_sec`, `provenance_status`, `denominator_status`, `label_source`, `label_confidence`.
- Classified Xid/OOM/CUDA/CUBLAS exceptions as `INFRA_FAILED`.
- Model loading now uses local `MODEL_PATH=/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` with `attn_implementation="eager"` and `local_files_only=True` where supported.
- Task mapping now uses the fixed LIBERO Object map: alphabet_soup=0, cream_cheese=1, salad_dressing=2, bbq_sauce=3, ketchup=4, tomato_sauce=5, butter=6, milk=7, chocolate_pudding=8, orange_juice=9.
- Environment setup now uses `benchmark_dict["libero_object"]()`, `OffScreenRenderEnv`, `initial_states[state_id]`, and `env.set_init_state(init_state)`.
- Gripper qpos measurement now uses MuJoCo gripper joint qpos as primary, with `obs["robot0_gripper_qpos"]` only as fallback/audit comparison.
- Output records `gripper_qpos_source`, `gripper_qpos_mujoco`, `gripper_qpos_obs`, `gripper_qpos_used`, and `gripper_qpos_source_priority`.
- MuJoCo/obs mismatch is recorded as `gripper_qpos_warning`; no silent zero qpos is allowed.
- `MEASUREMENT_FAILED:missing_gripper_qpos` rows use `label_confidence=not_label_measurement_failed`.
- Final env-step forced OPEN injection is preserved: clean raw action is decoded, normalized/inverted, then `env_action[-1]` is set to `forced_open_value_used=+1.0` immediately before `env.step()`.

Remaining caveat:

- Command-open proxy is an upper-bound physical/task susceptibility screen. It is not VIS and must not be treated as a gold label.

### scripts/diagnostics/compare_fast_vis_to_full_labels.py

Status: IMPLEMENTED

The script now computes positive recall, negative specificity, runtime reduction where available, false positives on controls, agreement with full VIS, recommended fast budget, and failure modes. It excludes `INFRA_FAILED`, `MEASUREMENT_FAILED`, `BLOCKED`, and `ERROR` rows from metrics and keeps proxy/silver evidence separate from gold labels.

### scripts/diagnostics/run_phase_e_canary.py

Findings before Phase E repair:

- Phase E canary v0 passed raw OpenVLA actions directly to `env.step(adv_act)` and `env.step(clean_act)`.
- This skipped the official `normalize_gripper_action(raw_action, binarize=True)` and `invert_gripper_action(env_action)` transform.
- The old canary result is therefore `INVALID_ACTION_SPACE_CONFOUNDED`.
- Phase E v1 then failed as `PHASE_MISALIGNED_COMPRESSED_WINDOW` because centered L10 entered a natural-open phase.
- Phase E v2 then failed as `PHASE_NOT_CAPTURED` because parent-start aligned L10 still had qpos near open.
- The core issue is not only budget. Compressed windows must be selected from real MuJoCo/obs qpos state, not only from centered window geometry or `phase_bin_proxy`.

Patch applied:

- Every environment step now receives `env_action = invert_gripper_action(normalize_gripper_action(raw_action, binarize=True))`.
- Added `--candidate-csv`, `--limit`, `--only-recommended`, and `--allow-unrecommended`.
- When `--candidate-csv` is provided, the script uses explicit `window_start/window_end` and does not call `centered_window()`.
- Candidate rows must provide `phase_alignment_source`; recommended filtering is enforced unless `--allow-unrecommended` is explicitly passed.
- Added `action_transform_version`.
- Added gripper action provenance fields:
  - `raw_clean_action_gripper`
  - `raw_adv_action_gripper`
  - `env_clean_action_gripper_after_transform`
  - `env_adv_action_gripper_after_transform`
  - `post_transform_gripper_action`
- Added MuJoCo-primary qpos audit fields:
  - `gripper_qpos_mujoco`
  - `gripper_qpos_obs`
  - `gripper_qpos_used`
  - `gripper_qpos_source_priority`
  - `gripper_qpos_warning`
- Added `previous_phase_e_v0_status=INVALID_ACTION_SPACE_CONFOUNDED`.
- Added Phase E mechanism fields: arm/action drift, token flips, VIS_OPEN count, env gripper OPEN count, raw/env gripper means, MuJoCo/obs qpos starts/mins/opening deltas, epsilon calibration, PGD budget provenance, `mechanism_status`, and `mechanism_reason`.
- `label_confidence=silver_candidate` is allowed only when `mechanism_status=mechanism_clean`; otherwise rows are `not_silver_candidate`.
- Added CPU-only `--dry-run` that exits before importing or loading OpenVLA/LIBERO/torch.

Remaining caveat:

- Low-budget VIS canary remains diagnostic/proxy evidence only. It cannot become a silver-candidate generator until true_closed phase alignment, mechanism_clean audit, positive/negative separation, and 5-sample smoke pass.

### scripts/diagnostics/generate_phase_e_aligned_windows.py

Status: IMPLEMENTED

The script enumerates L8/L10/L12 windows at parent_start_aligned, parent_start_plus_2, parent_start_plus_4, centered, and parent_end_aligned positions. It records MuJoCo/obs qpos fields when available, classifies windows as true_closed, transitional-pre-open, natural_open, or missing, and does not recommend missing-qpos rows.

Remaining caveat:

- On the local checkout `labels_v2` and server trace roots may be absent, so current local output may be a missing-qpos planning artifact. DeepSeek should rerun this script on the server after syncing the reviewed branch.

### scripts/vis_phase_conditioned_attack.py

Findings before patch:

- Default `--gpu_pair` was `6,7`, unsafe because GPU7 is permanently blacklisted.

Patch applied:

- Default changed to `1,0`, matching the current gold VIS reservation convention.
- Added GPU pair validation rejecting GPU3/GPU7 and the `CUDA_VISIBLE_DEVICES=2,6` plus `--gpu_pair 2,6` misuse.

### scripts/vis_rollout_adaptive_v3.py

Findings:

- Default `--gpu_pair` is `4,5`.
- Records GPU pair in stdout.
- Uses `device_map='auto'` with `max_memory` keyed by parsed physical GPU IDs.
- Uses `render_gpu_device_id=int(args.gpu_pair.split(',')[0])`.
- Does not set `CUDA_VISIBLE_DEVICES` internally.

Patch applied:

- Added GPU pair validation rejecting GPU3/GPU7 and the `CUDA_VISIBLE_DEVICES=2,6` plus `--gpu_pair 2,6` misuse.

Remaining caveat:

- Because this script uses physical IDs directly for model max memory and render GPU, the scheduler must avoid mixed logical/physical GPU mapping. Do not combine `CUDA_VISIBLE_DEVICES=2,6` with physical `--gpu_pair 2,6`.

## CPU-Only Verification

- `python -m py_compile` passed for:
  - `scripts/diagnostics/run_policy_only_vis_audit.py`
  - `scripts/diagnostics/run_command_open_proxy_replay.py`
  - `scripts/diagnostics/audit_fast_vis_outputs.py`
  - `scripts/diagnostics/run_phase_e_canary.py`
  - `scripts/vis_phase_conditioned_attack.py`
  - `scripts/vis_rollout_adaptive_v3.py`
- Dry-run passed for:
  - `run_policy_only_vis_audit.py --dry-run --candidate-csv tables/fast_vis_calibration_candidates_v0.csv --gpu-pair 2,6`
  - `run_command_open_proxy_replay.py --dry-run --candidate-csv tables/fast_vis_calibration_candidates_v0.csv --gpu-pair 2,6`
  - `run_phase_e_canary.py --dry-run`
  - `vis_phase_conditioned_attack.py --dry-run`
- Guard check passed:
  - `CUDA_VISIBLE_DEVICES=2,6` plus `--gpu-pair 2,6` exits with `INFRA_FAILED` before entering GPU code.

## Claim Boundary

- No Fast cascade result is validated yet.
- No policy-only result proves task failure.
- No command proxy result proves VIS.
- No silver/proxy label may be treated as gold.
- Old Phase D v0 command-proxy results are invalid and must be discarded.
- Old Phase E canary v0 results are invalid as `INVALID_ACTION_SPACE_CONFOUNDED`.
- DeepSeek microcheck v2 validates MuJoCo qpos measurement and final env-step OPEN injection, but Codex did not run GPU/rollout in this patch.
