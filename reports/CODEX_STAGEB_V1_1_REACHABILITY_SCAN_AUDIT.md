# Codex Stage-B v1.1 Reachability Scan Audit

**Date**: 2026-06-07
**Mode**: CPU-only audit. No GPU, VIS, random, PGD, rollout worker, watcher, or live-output mutation.
**Input root**: `/data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/`

## Conclusion

**BLOCKED_WITH_FINDINGS**

The clean traces themselves pass RC1a provenance/schema checks, but the package is not ready for 3-row corrected smoke because no RC1a candidate/window artifact was present and 12/27 full-scan traces ended before the requested `window_end=295`. Candidate windows must be generated from these RC1a clean traces and clipped inside actual rollout ranges before smoke.

## Summary

| Check | Result |
|---|---:|
| Trace CSVs | 27/27 |
| Summary JSONs | 27/27 |
| `trace_version=corrected_stageb_v1_1` | 27/27 |
| `source_snapshot_id=f9840cb1` | 27/27 |
| `prompt_style=official_in_out` | 27/27 |
| `image_preprocess_style=official_rot180_only` | 27/27 |
| RC1a open convention | 27/27 |
| `qpos_source=obs_robot0_gripper_qpos` | 27/27 |
| Clean condition only | 27/27 |
| Traces with attack/PGD/random evidence | 0/27 |
| `n_window_steps >= 8` | 27/27 |
| Full requested window inside actual rollout range | 15/27 |
| Candidate/window artifact found | 0 |

## Clean Scan Integrity

All 27 traces use the expected RC1a metadata:

```text
trace_version = corrected_stageb_v1_1
source_snapshot_id = f9840cb1
prompt_style = official_in_out
image_preprocess_style = official_rot180_only
open_convention = env_action_6_lt_neg_0p5_means_OPEN
qpos_source = obs_robot0_gripper_qpos
condition = clean
```

The scan log contains no `VIS`, `random`, or `PGD` command evidence. Per-trace counters also show `attack_this_step_count=0`, `pgd_applied_count=0`, and `attacks_applied_nonzero_count=0` for every trace.

## Blocking Findings

1. **Missing candidate artifact**: no RC1a reachability candidate/window table was found in the output root or obvious repo artifact paths. Therefore this audit cannot verify that candidates are completely derived from RC1a clean traces.
2. **Requested scan window exceeds actual rollout range in 12 traces**: those traces succeeded/ended before step 295, so any downstream candidate table must clip candidate windows to `max_step` for each trace. The affected traces still have at least 114 in-window clean steps, so they can support earlier windows, but not the full `[1,295]` requested range.

Affected early-ended traces are listed in `tables/codex_stageb_v1_1_reachability_findings.csv` with `window_out_of_actual_range`.

## Early-Done Explanation

The 12 short traces have `summary_success=1` and `infra_status=ok`, which is consistent with successful clean rollouts ending early. However, the summary schema does not include an explicit `done_reason` or `success_reason`, and the trace-level `done` column stayed `0` in parsed rows. This is not a schema/provenance failure for the clean scan, but future summaries should record an explicit reason.

## Required Next Step Before Smoke

Generate or provide a candidate/window table from this exact RC1a clean scan with, at minimum:

```text
source_trace
trace_version
source_snapshot_id
task_key
state_id
window_start
window_end
actual_min_step
actual_max_step
n_window_steps
candidate_source=rc1a_clean_reachability_scan
```

Gate requirements for that table:

- every source trace is `corrected_stageb_v1_1` and `f9840cb1`;
- every candidate window is within `actual_min_step..actual_max_step`;
- every candidate has `n_window_steps >= 8`;
- no old labels, old windows, patched rerun outputs, or pre-v1.1 traces are referenced.
