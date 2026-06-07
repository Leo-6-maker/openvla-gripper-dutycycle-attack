# StageB Patched Runner Live Audit

Date: 2026-06-07

Scope: read-only audit of live 44-row selective patched rerun. No worker was stopped, no runner source was modified, and no GPU/VIS/rollout command was launched by Codex.

## Provenance

- Server repo: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605`
- Output root: `/data/liuyu/outputs/stageb_selective_rerun_patched_20260607`
- Queue: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/stageb_selective_rerun_queue.csv`
- Git branch: `exp/vis-prefix-margin-repair-20260603`
- Git HEAD: `fb064b36b47ca1a2cf3b8ca04675fc802cec9cfc`
- Repo dirty: `no`

## Worker Commands

- worker_10: script `/tmp/run_stageb_rerun_worker_10.sh`, CUDA_VISIBLE_DEVICES=`1,0`, command_count=30, output_root=`/data/liuyu/outputs/stageb_selective_rerun_patched_20260607`
- worker_26: script `/tmp/run_stageb_rerun_worker_26.sh`, CUDA_VISIBLE_DEVICES=`2,6`, command_count=30, output_root=`/data/liuyu/outputs/stageb_selective_rerun_patched_20260607`
- worker_45: script `/tmp/run_stageb_rerun_worker_45.sh`, CUDA_VISIBLE_DEVICES=`4,5`, command_count=28, output_root=`/data/liuyu/outputs/stageb_selective_rerun_patched_20260607`

Running worker PIDs observed from `ps`:

```text
15997     1       21:11 bash /tmp/run_stageb_rerun_worker_26.sh
15998     1       21:11 bash /tmp/run_stageb_rerun_worker_45.sh
15999     1       21:11 bash /tmp/run_stageb_rerun_worker_10.sh
20027 15997       06:15 /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/run_stageb_vis_labeling.py --task bbq_sauce --state-id 9 --window_start 108 --window_end 118 --condition vis_pgd --gpu_pair 0,1 --pgd_steps 20 --eps_raw_pixels 6 --max_steps 400 --job_id 11002 --output_dir /data/liuyu/outputs/stageb_selective_rerun_patched_20260607
37385 15999       01:19 /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/run_stageb_vis_labeling.py --task cream_cheese --state-id 9 --window_start 111 --window_end 120 --condition random_linf --gpu_pair 0,1 --eps_raw_pixels 6 --max_steps 400 --job_id 10003 --output_dir /data/liuyu/outputs/stageb_selective_rerun_patched_20260607
40790 15998       00:20 /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python -u /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/run_stageb_vis_labeling.py --task butter --state-id 2 --window_start 143 --window_end 153 --condition random_linf --gpu_pair 0,1 --eps_raw_pixels 6 --max_steps 400 --job_id 12011 --output_dir /data/liuyu/outputs/stageb_selective_rerun_patched_20260607
42301 42298       00:00 python3 /tmp/codex_stageb_live_audit.py
45221     1    01:49:21 bash /data/liuyu/outputs/overnight_stageb_labels_20260607/CHAIN_CATCHUP_0.sh
```

## Source-Code Schema Verdict

**Verdict: HARD_FAIL for downstream label readiness.** The runner uses patched obs qpos and transformed env action, but the per-step schema is incomplete for the requested patched audit contract.

Key findings:
- HARD_FAIL: raw_action_full_vector - missing raw_action_0..6 in trace writer
- HARD_FAIL: obs_qpos_mean_abs_sum - missing obs_gripper_qpos_mean and obs_gripper_qpos_abs_sum
- HARD_FAIL: decoded_open_bool - missing decoded_open_bool; only env_grip/open summary available
- WARN: attack_active_field - runner writes attack_this_step, not attack_active
- HARD_FAIL: condition_trace_field - condition is summary-only, not per-step trace
- HARD_FAIL: row_window_trace_fields - missing row_id and/or per-step window_start/window_end
- HARD_FAIL: pair_id_random_matched_id - missing random_matched_id/pair_id
- WARN: qpos_delta_post_step - qpos_after is read from obs before env.step, so summary qpos_delta is not a post-step physical response metric

## Completed Trace Schema

- Completed summaries found: 12
- Completed traces found: 12
- unknown_or_smoke: vis_pgd=1, random_linf=3
- worker_10: vis_pgd=2, random_linf=1
- worker_26: vis_pgd=1, random_linf=1
- worker_45: vis_pgd=2, random_linf=1

Completed trace audit notes:
- `env_action_0..6`, `obs_gripper_qpos_0/1`, and `qpos_source=obs_robot0_gripper_qpos` are present in completed traces.
- `raw_action_0..6`, `decoded_open_bool`, `obs_gripper_qpos_mean`, `obs_gripper_qpos_abs_sum`, per-step `condition`, `row_id`, `window_start`, `window_end`, and pair id fields are missing.
- qpos values are not all placeholder 0/0.5/1 in sampled completed traces.

## Qpos/Action Sanity

- Trace-level qpos values are finite and vary over time for completed worker traces.
- Summary `qpos_delta` is zero for completed traces even when trace qpos varies; source inspection shows `qpos_after` is read before `env.step`, so summary qpos_delta is not a post-step physical response metric.
- Use `tables/stageb_patched_runner_qpos_action_audit.csv` for recomputed pre/window/post qpos and open streak metrics.

## Pair Matching

- Parsed worker-script expected windows: 44
- Pairing status counts: {'PASS': 44, 'FAIL': 4}
- Job status counts: {'pending': 69, 'failed_VIS_FAIL': 4, 'failed_RAND_FAIL': 4, 'running': 3, 'completed': 8, '': 4, 'completed_unmatched': 4}
- VIS/random expected pairs match by task/state/window in the worker scripts.
- Several completed `job9xxx` traces exist in the output root and are not part of the 44-row worker scripts; they are marked `old_unpatched_mix_status=FAIL`/unmatched in the pairing audit until quarantined or explicitly documented as smoke.

## Label Script Readiness

**Verdict: NOT READY.** Existing label builder `scripts/diagnostics/build_labels_v3_candidate.py` does not consume patched trace-level qpos/action fields or recompute matched VIS/random qpos_delta_attack/post.
- FAIL: uses_patched_obs_gripper_qpos - build_labels_v3_candidate.py does not parse patched trace qpos fields
- FAIL: uses_env_action_decoded_open_bool - does not parse env_action_0..6 or decoded_open_bool
- WARN: uses_matched_vis_random_pair - script consumes summary/candidate labels and does not explicitly validate patched VIS/random trace pairs
- FAIL: computes_qpos_delta_attack_post - does not compute qpos_delta_attack/post from patched traces
- WARN: excludes_random_confounded - random_confounded not explicitly used in script

## Blocking Issues

- Source schema raw_action_full_vector: missing raw_action_0..6 in trace writer
- Source schema obs_qpos_mean_abs_sum: missing obs_gripper_qpos_mean and obs_gripper_qpos_abs_sum
- Source schema decoded_open_bool: missing decoded_open_bool; only env_grip/open summary available
- Source schema condition_trace_field: condition is summary-only, not per-step trace
- Source schema row_window_trace_fields: missing row_id and/or per-step window_start/window_end
- Source schema pair_id_random_matched_id: missing random_matched_id/pair_id
- Label readiness uses_patched_obs_gripper_qpos: build_labels_v3_candidate.py does not parse patched trace qpos fields
- Label readiness uses_env_action_decoded_open_bool: does not parse env_action_0..6 or decoded_open_bool
- Label readiness computes_qpos_delta_attack_post: does not compute qpos_delta_attack/post from patched traces
- Summary qpos_delta is not post-step; use recomputed trace qpos metrics for audit only until runner writes causal pre/post fields.
- Unmatched job9xxx traces are present in the patched output root and must not be mixed into 44-row patched rerun labels.

## Non-Blocking Observations

- Runner correctly avoids `qpos[-2:]` and records `qpos_source=obs_robot0_gripper_qpos`.
- Runner applies `normalize_gripper_action(..., binarize=True)` followed by `invert_gripper_action(...)` before `env.step(env_action_full)`.
- `CUDA_VISIBLE_DEVICES=2,6` with `--gpu_pair 0,1` follows visible-index convention for worker_26.
- worker_45 log includes OOM failures for some jobs; these should remain infra failures and must not become labels.

## Files Written

- `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/stageb_patched_runner_live_schema_audit.csv`
- `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/stageb_patched_runner_pairing_audit.csv`
- `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/tables/stageb_patched_runner_qpos_action_audit.csv`
- `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/reports/STAGEB_PATCHED_RUNNER_LIVE_AUDIT.md`
