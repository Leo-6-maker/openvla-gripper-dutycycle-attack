# Table1 Clean Patched v4 Artifact Index

This page indexes the Table1 clean patched-v4 output artifacts. The artifacts are stored outside git and should not be committed to the repository.

## Output Root

```bash
/data/liuyu/outputs/table1_official_eval_recovery_20260527
```

## Clean SR

| Suite | N Episodes | N Success | SR |
|-------|-----------|-----------|-----|
| libero_spatial | 100 | 79 | 0.790 |
| libero_object | 100 | 80 | 0.800 |
| libero_goal | 100 | 82 | 0.820 |
| libero_10 | 100 | 51 | 0.510 |
| **Overall** | **400** | **292** | **0.730** |

## Old vs New Comparison

| Suite | Old v4 | New v4 | Delta |
|-------|--------|--------|-------|
| libero_spatial | 0.82 | 0.790 | -0.030 |
| libero_object | 0.68 | 0.800 | +0.120 |
| libero_goal | 0.80 | 0.820 | +0.020 |
| libero_10 | 0.53 | 0.510 | -0.020 |
| **Overall** | **0.708** | **0.730** | **+0.022** |

## Key Tables

- `tables/table1_recovered_summary_by_suite.csv` — Per-suite SR with source notes
- `tables/table1_old_vs_new_comparison.csv` — Old v4 vs new v4 delta by suite
- `tables/table1_worker_status.csv` — Worker-level log recovery status
- `tables/table1_artifact_validity_audit.csv` — Stale artifact exclusion audit

## Key Reports

- `reports/TABLE1_CLEAN_PATCHED_V4_REPORT.md` — Final combined Table1 report
- `reports/TABLE1_RECOVERY_AUDIT.md` — Log recovery methodology and worker status
- `reports/CSV_OVERWRITE_BUG_AUDIT.md` — Root cause analysis of CSV overwrite bug

## Stale Artifacts Excluded

The following stale partial CSVs from a crashed Goal worker (wrong GPU indices, 0/100) were excluded from final aggregation:

- `/data/liuyu/outputs/table1_clean_patched_v4_remaining3_20260527/tables/object_official_script_100_manifest.csv`
- `/data/liuyu/outputs/table1_clean_patched_v4_remaining3_20260527/tables/object_official_script_100_summary_by_task.csv`

## Git Boundary

Do not commit the output directory, shards, CSVs, or any generated artifacts to git. Only source, tests, and documentation belong in the repository.
