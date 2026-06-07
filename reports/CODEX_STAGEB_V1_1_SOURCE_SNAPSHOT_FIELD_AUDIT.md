# Codex Stage-B v1.1 Source Snapshot Field Audit

Date: 2026-06-07

Audited commit: `a400a1b864e4c6aaeb4f3f222a8249ea5f456df2`

Server path: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605`

Mode: CPU-only focused audit. No GPU, VIS, rollout, watcher, or server live output mutation was run.

## Verdict

`SOURCE_SNAPSHOT_TRACE_FIELD_PASS_SHA_TABLE_STALE`

The `source_snapshot_id` trace-field fix is implemented and present on the server:

- runner defines `SOURCE_SNAPSHOT_ID = '4fe01c43'`
- trace rows include `source_snapshot_id`
- summary JSON includes `source_snapshot_id`
- `TRACE_COLUMNS` includes `source_snapshot_id`
- validator `REQUIRED_COLUMNS` includes `source_snapshot_id`
- server runner SHA is `cf2a2320976240a2dade0d6862130e250c48be19b7c8a00a072de2590b6b7c59`
- server `test_trace_schema_v1_1.py` passed under the official env

The remaining issue is provenance bookkeeping: `tables/stageb_v1_1_rc1_server_file_sha.csv` still contains pre-a400 SHA values for the runner, validator, and trace schema test. Therefore the existing SHA CSV must not be used as the current a400 server snapshot lock until it is regenerated.

## Checks

| Check | Status | Evidence |
|---|---:|---|
| Runner has `SOURCE_SNAPSHOT_ID` | PASS | `scripts/run_stageb_vis_labeling.py` defines `SOURCE_SNAPSHOT_ID = '4fe01c43'`. |
| Trace row writes `source_snapshot_id` | PASS | Runner row dictionary writes `'source_snapshot_id': SOURCE_SNAPSHOT_ID`. |
| Summary writes `source_snapshot_id` | PASS | Summary JSON includes `'source_snapshot_id': SOURCE_SNAPSHOT_ID`. |
| `TRACE_COLUMNS` includes `source_snapshot_id` | PASS | Runner column order includes `source_snapshot_id` after `git_dirty`. |
| Validator requires `source_snapshot_id` | PASS | `scripts/stageb/validate_stageb_trace_v1_1.py` includes `source_snapshot_id` in `REQUIRED_COLUMNS`. |
| Missing manifest files uploaded | PASS | Server now has `test_stageb_open_count_convention.py`, `OPENVLA_LIBERO_EXECUTABLE_SPEC.md`, and `CODEX_STAGEB_V1_1_SMOKE_R3_LIVE_AUDIT.md`. |
| Server test schema passes | PASS | Server official env: `tests/stageb/test_trace_schema_v1_1.py` returned `5 passed`. |
| Server full stageb tests | PASS | Server official env previously in this audit sequence: `39 passed`; local a400 worktree: `41 passed`. |
| Server SHA table current | FAIL_STALE | SHA CSV still lists old runner SHA `d37d8f49...`, old validator SHA `838159d2...`, and old test schema SHA `b959f8...`, while a400/server files have new SHA values. |
| Server git worktree clean | FAIL | Server `git status --short` still reported 30 lines; file SHA must remain the provenance source. |

## Server Observations

Observed server file hashes:

| File | Live server SHA |
|---|---|
| `scripts/run_stageb_vis_labeling.py` | `cf2a2320976240a2dade0d6862130e250c48be19b7c8a00a072de2590b6b7c59` |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | `decd14f32d88b0a14832b2b4700ee2fd25c703245b0d6644f86652b5b80e0a38` |
| `tests/stageb/test_trace_schema_v1_1.py` | `b959f8da1b15f1b2079da61b02ff67397d3209a4e5d1ccb01d955f7e4b77e9e9` |
| `tests/stageb/test_stageb_open_count_convention.py` | `0f7bc25bc65d5e8c8682563c9f9f267c7645b5c20941acae1e44b04e614606fb` |
| `reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md` | `76774976d24d81d0f4d01f8a27f1666ff999eb13be9cd5ba95bbea800fbdb05f` |
| `reports/CODEX_STAGEB_V1_1_SMOKE_R3_LIVE_AUDIT.md` | `de6554cb7c8255aabd4b860f94624ab71f754070421b61e84c4c448b58a7211e` |

The server validator contains the new required column but does not check the specific value `4fe01c43`. It currently verifies column presence, not exact snapshot identity.

## Validation

Local a400 worktree:

```text
py_compile: PASS for runner, validator, and spec
pytest tests/stageb: 41 passed in 0.23s
```

Server official env:

```text
py_compile: PASS for runner and validator
pytest tests/stageb/test_trace_schema_v1_1.py: 5 passed
```

## Recommendation

The `source_snapshot_id` fix is ready for the next v1.1 smoke/reachability scan, but update the server SHA table before treating a400 as a locked snapshot.

Minimum next provenance cleanup:

1. Regenerate `tables/stageb_v1_1_rc1_server_file_sha.csv` for the a400 server files.
2. Update `reports/STAGEB_V1_1_RC1_SERVER_SNAPSHOT_VERIFY.md` with the new runner/validator/test schema SHA values.
3. Consider making validator hard-fail if `source_snapshot_id != 4fe01c43`.
