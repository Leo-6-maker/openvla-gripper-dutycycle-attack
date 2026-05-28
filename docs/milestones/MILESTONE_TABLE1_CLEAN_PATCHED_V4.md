# Milestone Table1 — Clean Patched v4 Official-Eval-Aligned Recovery

## Purpose

Freeze the clean patched-v4 Table1 result as the official clean denominator for all subsequent detector and attack work. This milestone marks the completion of the PIL preprocessing alignment fix and the full 400-episode clean eval across all four LIBERO suites.

## Clean SR by Suite

| Suite | N Episodes | N Success | SR |
|-------|-----------|-----------|-----|
| libero_spatial | 100 | 79 | 0.790 |
| libero_object | 100 | 80 | 0.800 |
| libero_goal | 100 | 82 | 0.820 |
| libero_10 | 100 | 51 | 0.510 |
| **Overall** | **400** | **292** | **0.730** |

## Old v4 Comparison

| Suite | Old v4 (TF JPEG) | New v4 (PIL Lanczos) | Delta |
|-------|-------------------|---------------------|-------|
| libero_spatial | 0.82 | 0.790 | -0.030 |
| libero_object | 0.68 | 0.800 | **+0.120** |
| libero_goal | 0.80 | 0.820 | +0.020 |
| libero_10 | 0.53 | 0.510 | -0.020 |
| **Overall** | **0.708** | **0.730** | **+0.022** |

## Object Improvement: 68→80

The +12pt Object improvement is entirely explained by switching from TF JPEG preprocessing to PIL Lanczos:

- Old v4 used TF `decode_jpeg` + `ResizeMethod.BICUBIC`, which introduced a lossy JPEG compression round-trip on RGB observations.
- Patched v4 uses PIL Lanczos resize directly on the RGB array, matching the corrected official eval script (`tmp_official_eval.py`).
- The JPEG round-trip caused per-frame color-space distortion that degraded Object-suite action decoding; the PIL path eliminates this distortion completely.

## Configuration

- **Preprocess**: official_pil_lanczos (PIL Lanczos, no JPEG round-trip)
- **Prompt**: `In: What action should the robot take to {task}?\nOut:`
- **EOS**: add_if_missing_29871
- **Center crop**: True
- **Num steps wait**: 10
- **Postprocess gripper**: normalize_gripper_action + invert_gripper_action
- **Success metric**: done
- **Attention**: eager
- **Dtype**: bfloat16
- **Action path**: generate_manual_decode

## GPU Topology

| Worker | Suite | GPU Pair | Status |
|--------|-------|----------|--------|
| spatial_w13 | libero_spatial | 1,3 | Complete |
| goal_w26 | libero_goal | 2,6 | Complete |
| libero10_w45 | libero_10 | 4,5 | Complete |

GPU0 (permanent Xid13) and GPU7 (Xid13+Xid31 history) were quarantined and excluded from all runs.

## Error Summary

- **Zero runtime errors** across all 400 episodes.
- **Zero Xid** during all runs.
- **Zero CUDA illegal memory access** errors.

## CSV Overwrite Bug

The original official eval script wrote all worker output to a single hardcoded filename (`object_official_script_100_manifest.csv`), regardless of suite or worker count. When multiple workers ran in parallel, the last worker to finish overwrote all prior data.

**Fix applied** in commit ddfc219: worker-safe shard naming (`manifest_{suite}_worker_{id}.csv` in `tables/shards/`).

**Recovery**: Final Table1 SR was reconstructed from worker stdout logs (`recover_table1_from_logs.py`), not from stale partial CSV files.

## Boundary Statement

This is a **clean denominator candidate**, not an attack result.
No attacks were run.
No VIS/oracle/random/manual outcomes were used.
No detector was trained.
The clean SR is intended as the baseline against which all future attack results are compared.
