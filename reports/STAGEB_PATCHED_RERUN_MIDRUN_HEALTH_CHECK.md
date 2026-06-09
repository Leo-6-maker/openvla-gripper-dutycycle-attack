# Stage-B Patched Rerun — Mid-Run Health Check

**Timestamp**: 2026-06-07 13:27–13:35 CST
**Server**: klfy-SYS-4028GR-TR2
**Read-only**: No workers interrupted, no live outputs mutated.

## 1. Worker Liveness & GPU Health

| Worker | GPU Pair | PID | Current Job | GPU Mem (0/1 or 2/6 or 4/5) | Temp |
|--------|----------|-----|-------------|------------------------------|------|
| worker_10 | 1,0 | 17290 | ketchup s3 [112,122] random_linf job10019 | 8229+8719 MiB | 66/56°C |
| worker_26 | 2,6 | 23470 | tomato_sauce s6 [98,108] vis_pgd job11018 | 8717+8249 MiB | 66/74°C |
| worker_45 | 4,5 | 20887 | butter s7 [143,153] random_linf job12025 | 8719+8249 MiB | 69/62°C |

- GPU3, GPU7: **1 MiB each, idle** (blacklist respected)
- **No fresh Xid** in dmesg (historical Xid13, Xid31 from prior incidents only)
- **No CUDA OOM** observed in worker logs
- All workers use `python -u` (unbuffered), stdout/stderr → `worker_*.log` in output root
- Model loads in ~13s per job, 4 checkpoint shards, action_dim=7

### Worker Script Wrappers

All 3 wrappers at `/tmp/run_stageb_rerun_worker_*.sh`:
- `CUDA_VISIBLE_DEVICES=1,0` / `2,6` / `4,5`
- `--gpu_pair 0,1` with `--eps_raw_pixels 6 --max_steps 400`
- Exit code checked: `rc=$?; if [ $rc -ne 0 ]; then echo "VIS_FAIL $jid rc=$rc"; fi`
- Source: reviewed repo at `openvla-gripper-dutycycle-attack-reviewed-20260605`

## 2. Waiter Status

- PID 25401, started 12:37, script at `/tmp/wait_rerun_and_postprocess_v2.sh`
- Logic: `while ps aux | grep -q '[v]is_labeling'; do sleep 60; done`
- On completion: runs `run_patched_rerun_postprocess_hotfix.py`
- Current status: **STILL WAITING** (3 vis_labeling processes found)
- Waiter stdout/stderr → pipes (not a file), so no persistent log

## 3. Output Counts

| Category | Count |
|----------|-------|
| Trace CSVs (total) | 60 |
| Trace CSVs (good, job_id >= 10000) | 56 |
| Trace CSVs (smoke, job_id < 10000) | 4 |
| Summary JSONs (total) | 60 |
| Summary JSONs (good) | 56 |
| Summary JSONs (smoke) | 4 |
| Unique windows in good summaries | 28 |
| Complete pairs (VIS+RAND) | 27 |
| VIS-only | 1 (butter s7 [143,153] job12024 — currently running) |
| Windows not yet started | 16 |
| **Overall progress** | **56/88 = 63.6%** |

## 4. Contamination: 4 Smoke Traces (job9xxx)

| File | job_id | task | condition | qpos_delta | Trace rows |
|------|--------|------|-----------|------------|-------------|
| trace_alphabet_soup_random_linf_job9203.csv | 9203 | alphabet_soup s4 [25,35] | random_linf | 0.0 | 40 |
| trace_milk_random_linf_job9105.csv | 9105 | milk s7 [30,40] | random_linf | 0.0 | 45 |
| trace_salad_dressing_random_linf_job9201.csv | 9201 | salad_dressing s2 [49,59] | random_linf | 0.0 | 64 |
| trace_salad_dressing_vis_pgd_job9200.csv | 9200 | salad_dressing s2 [49,59] | vis_pgd | 0.0 | 64 |

- All 4 have **qpos_delta=0.0** (old buggy placeholder)
- Trace row counts (40-64) are shorter than patched rerun (100-220+)
- **Must be excluded** from hotfix postprocess
- Origin: leftover from smoke test run before patched rerun launched

## 5. Schema Audit (56 good traces)

### Columns Present (19 cols, all good traces):
- step, in_window, attack_this_step, env_grip, arm_l2
- pgd_applied, attacks_applied, gripper_qpos, done
- **env_action_0..6** (7 cols)
- **obs_gripper_qpos_0, obs_gripper_qpos_1** (2 cols)
- **qpos_source** = `obs_robot0_gripper_qpos` (verified in all 56 traces)

### Columns Missing (v1.1 gap — all 56 traces):

| Missing Column | Needed For |
|----------------|------------|
| trace_version | Provenance tracking |
| runner_version | Runner provenance |
| pair_id | Cross-reference VIS↔random |
| condition | Self-contained trace metadata |
| task_key | Self-contained trace metadata |
| state_id | Self-contained trace metadata |
| window_start | Self-contained trace metadata |
| window_end | Self-contained trace metadata |
| raw_action_0..6 | Action replay reconstruction |

**Conclusion**: Runner uses pre-v1.1 trace format. Metadata exists only in summary JSONs. Hotfix postprocessor reads from summaries (not filename), so label generation is SAFE. But v1.1 trace schema upgrade is needed for standalone trace portability.

## 6. Open Convention

**Correct convention confirmed**: `env_action_6 < -0.5` → OPEN

All 56 traces use this convention. No trace shows the old wrong convention (`env_action_6 > 0.5` → OPEN).

Sample verification from trace `alphabet_soup_vis_pgd_job11008.csv`:
- 10/10 window steps have `env_action_6 < -0.5` → all OPEN
- 0/10 have `env_action_6 > 0.5` → none using wrong convention

## 7. Qpos Analysis

### Summary qpos_delta is UNRELIABLE (confirmed)

Multiple traces show qpos_delta=0.0 in summary but real non-zero shifts in per-step trace data:

| Window | Summary VIS qpos_delta | Trace VIS total_delta |
|--------|----------------------|----------------------|
| bbq_sauce s9 [108,118] | 0.0 | -0.00223 |
| butter s2 [143,153] | 0.0 | -0.00164 |
| butter s3 [39,49] | 0.0 | -0.00043 |

**Root cause**: Summary qpos_delta was computed before the P0 fixes (wrong indices, timing, signed mean). Trace-level shifted qpos (step_dict[s+1], abs_sum) shows real physical responses.

### Hotfix Postprocessor is ESSENTIAL

The hotfix postprocessor recomputes qpos from trace CSVs using:
- `abs(q0) + abs(q1)` → abs_sum
- `step_dict[s+1]` for shifted measurement
- Proper after-env-step timing

This will produce DIFFERENT (correct) qpos_delta values than the summary JSON fields.

## 8. Postprocess Logs

- `postprocess_hotfix.log` (42 bytes, mtime 12:37): "Waiting for rerun to finish..." — from current waiter
- `postprocess.log` (42 bytes, mtime 12:11): "Waiting for rerun to finish..." — from earlier waiter instance
- `smoke.log`, `smoke_fast.log`, `smoke_v3.log`: from 3-row smoke at 11:24–11:40

**No hotfix postprocess has run yet.** These logs are waiter echo output, not postprocess results.

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| 4 smoke traces in output | LOW | Hotfix filters by job_id >= 10000 |
| Summary qpos_delta wrong | MEDIUM | Hotfix recomputes from traces |
| Trace schema missing v1.1 cols | LOW (for labels) | Postprocess reads summaries for metadata |
| Filename task parsing (split bug) | LOW | Postprocess reads from summary JSON, not filename |
| Waiter uses grep pattern matching | LOW | Pattern `[v]is_labeling` is specific enough |

## 10. Verdict

**Rerun is HEALTHY.** No worker errors, no GPU faults, correct open convention, correct qpos_source in all traces. The 4 smoke contaminants are benign (hotfix will exclude). Schema gaps are known v1.1 items and don't block label generation. Proceed with waiting for completion → hotfix postprocess → audit.
