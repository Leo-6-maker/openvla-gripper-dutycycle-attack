# Codex Stage-B v1.1 RC1 Server Snapshot Audit

Date: 2026-06-07

Audited branch: `exp/vis-prefix-margin-repair-20260603`

Audited commit: `6f8170cc921ea9794d8c7f207b91750c610c0bcc`

Server path: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605`

Runtime used for verification: `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python`

Mode: CPU-only server snapshot audit. No GPU, VIS, rollout, watcher, or server live output directory was modified.

## Verdict

`CORE_SNAPSHOT_SHA_PASS_FULL_MANIFEST_PARTIAL`

The server RC1 core runtime snapshot is materially improved versus the previous audit:

- `scripts/diagnostics/generate_stageb_worker_scripts.py` is now in the manifest.
- 16 core file SHA rows were recorded and all 16 matched the live server files.
- Server `tests/stageb` passed under the official env: `39 passed in 0.20s`.
- Runner still records `trace_version = corrected_stageb_v1_1` and the corrected Stage-B v1.1 gripper semantics.

However, the live server git worktree is still not clean by `git status --short` and the full manifest is still not completely present on the server. File SHA now replaces git cleanliness for the 16 core files only, not for the full 39-row manifest.

## Required Checks

| Check | Status | Evidence |
|---|---:|---|
| Server file SHA matches manifest/SHA table | PASS_CORE | `tables/stageb_v1_1_rc1_server_file_sha.csv` has 16 rows; all 16 matched live server SHA256. |
| Previously missing 8 manifest files fixed | PARTIAL | 5 of the previously missing 8 were present. 3 manifest paths were still missing on server. |
| `generate_stageb_worker_scripts.py` is in manifest | PASS | Manifest includes both `scripts/generate_stageb_worker_scripts.py` and `scripts/diagnostics/generate_stageb_worker_scripts.py`. |
| Server `tests/stageb` pass | PASS | Official env pytest: `39 passed in 0.20s`. |
| Server trace runner uses `corrected_stageb_v1_1` | PASS | Server runner contains `TRACE_VERSION = 'corrected_stageb_v1_1'`; postprocess/label-builder require the same version. |
| `source_snapshot_id` written into freeze/report docs | PASS | `STAGEB_V1_1_RC1_FREEZE_REPORT.md` and `STAGEB_V1_1_RC1_SERVER_SNAPSHOT_VERIFY.md` record `source_snapshot_id = 4fe01c43`. |
| Dirty/manual upload replaced by file SHA | PARTIAL | Core 16 SHA rows are verified, but live server `git status --short` still reported 27 lines and 3 manifest paths were missing. |

## Server Verification Details

Observed live server state:

```text
GIT_HEAD ca3a97e73965e2582e28066a93892e3dd9c24617
GIT_STATUS_LINES 27
MANIFEST_COUNT 39
MANIFEST_MISSING_COUNT 3
SHA_ROWS 16
SHA_MISMATCH_COUNT 0
```

Missing manifest paths on server:

```text
tests/stageb/test_stageb_open_count_convention.py
reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md
reports/CODEX_STAGEB_V1_1_SMOKE_R3_LIVE_AUDIT.md
```

Server compile/test:

```text
py_compile: PASS for spec, runner, postprocess, label_builder, validator, and both worker generators
pytest tests/stageb: 39 passed in 0.20s
```

## Provenance Contract Audit

The reports define the intended trace provenance contract:

```text
git_commit = 3985809a
source_snapshot_id = 4fe01c43
git_dirty = 0
trace_version = corrected_stageb_v1_1
exec_spec_version = openvla_libero_exec_spec_v1_20260607
```

The current runner records `git_commit`, `git_dirty`, `trace_version`, and `exec_spec_version`. It does not currently include a `source_snapshot_id` trace field in the 52-column trace schema. If future traces must satisfy the full provenance contract literally, the runner schema should add `source_snapshot_id`.

## Findings

No core semantic blocker was found for the 16 SHA-verified runtime files.

Remaining issues:

- Server live git status is not clean despite the server verify report saying `git_dirty=0`; use SHA table as provenance, not server git status.
- The SHA table covers 16 core files, while the manifest has 39 rows. The un-hashed manifest files are not fully provenance-locked.
- Three manifest files are still missing from the live server path.
- `source_snapshot_id` is documented but not emitted by the runner trace schema.

## Recommendation

Treat RC1 server snapshot as valid for core runtime smoke/reachability scans only if the run records or externally pins the verified SHA table.

Before calling the server path a complete RC1 source snapshot:

1. Sync the 3 missing manifest files or remove them from server-snapshot scope.
2. Either make the server worktree genuinely clean or stop claiming `git_dirty=0` from `git status`; instead claim `core_file_sha_verified`.
3. Add `source_snapshot_id` to runner trace rows if that field is required for trace-level provenance.
