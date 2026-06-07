# Codex Stage-B v1.1 RC1 SHA Refresh Audit

Date: 2026-06-07

Audited base commit: `8108195e19e5d4f05dc65f390740996f005052b3`

Mode: CPU-only focused audit and provenance-table repair. No GPU, VIS, rollout, watcher, or server live output mutation was run.

## Verdict

`SHA_TABLE_PASS_VALIDATOR_EXACT_CHECK_PASS`

The source snapshot field and validator exact-value check are now usable:

- runner records `source_snapshot_id = 4fe01c43`
- validator hard-fails if `source_snapshot_id != 4fe01c43`
- test coverage now exercises the validator with a wrong snapshot ID
- server SHA table has 19 rows and matches all 19 live server files
- local and server `tests/stageb` both pass: `42 passed`

Codex found and fixed two provenance gaps during this audit:

1. `tables/stageb_v1_1_rc1_server_file_sha.csv` still had the pre-exact-check validator SHA.
2. `test_source_snapshot_id_exact` originally checked only constants; it now constructs a synthetic trace and calls `validate_trace()`.

## Server Verification

Server path:

```text
/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605
```

Official env:

```text
/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
```

Results:

```text
SHA_ROWS 19
SHA_MISMATCH_COUNT 0
pytest tests/stageb: 42 passed in 0.20s
```

Key server SHA rows:

| File | SHA |
|---|---|
| `scripts/run_stageb_vis_labeling.py` | `cf2a2320976240a2dade0d6862130e250c48be19b7c8a00a072de2590b6b7c59` |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | `6c1add49cc998cec07c3fce62e92c388077a975b56288e1f4b82e4cbdf7b325c` |
| `tests/stageb/test_trace_schema_v1_1.py` | `970115a6aa6afe995efaa24ea7146165fed86de622164d3d1d79539e3be64d32` |

## Local Verification

```text
py_compile: PASS for validator and trace schema test
pytest tests/stageb: 42 passed in 0.22s
```

## Remaining Caveat

The server worktree still uses a manually synchronized source tree rather than a clean GitHub checkout. Treat the 19-row SHA table as the provenance lock for RC1 server execution, not `git status`.

## Recommendation

RC1 provenance is now sufficient for a v1.1 clean reachability scan, provided future traces record:

```text
git_commit = 3985809a
source_snapshot_id = 4fe01c43
git_dirty = 0
trace_version = corrected_stageb_v1_1
exec_spec_version = openvla_libero_exec_spec_v1_20260607
```

Do not use old queue windows as negative labels; select windows from v1.1 official clean trajectories.
