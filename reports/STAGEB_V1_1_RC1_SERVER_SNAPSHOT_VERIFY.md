# Stage-B v1.1 RC1 Server Snapshot Verification

**Date**: 2026-06-07
**Commit**: `3985809a`
**Server**: klfy-SYS-4028GR-TR2

## Snapshot Identity

| Field | Value |
|-------|-------|
| git_commit | `3985809a` |
| branch | `exp/vis-prefix-margin-repair-20260603` |
| source_snapshot_id | `4fe01c43` (spec module SHA256 prefix) |
| git_dirty | `0` (clean checkout from tar upload) |
| Server source path | `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605` |

## File Count

27 RC1 files synced to server.

## py_compile

| Interpreter | Result |
|-------------|--------|
| system py3.6 | 6/9 scripts PASS; spec/gripper_semantics/attack_adapter require py3.7+ (`from __future__ import annotations`) |
| conda py3.10 | spec: PASS; all other files verifiable |

## Tests

```
39 passed in 0.21s
```

All 39 tests across all `tests/stageb/test_*.py` files pass on server.

## Server File SHA256

See `tables/stageb_v1_1_rc1_server_file_sha.csv` for full listing.

## Trace Provenance Contract

Every subsequent v1.1 trace MUST record:

| trace field | value |
|-------------|-------|
| `git_commit` | `3985809a` |
| `source_snapshot_id` | `4fe01c43` |
| `git_dirty` | `0` |
| `trace_version` | `corrected_stageb_v1_1` |
| `exec_spec_version` | `openvla_libero_exec_spec_v1_20260607` |

## Verification Checklist

- [x] 27 RC1 files uploaded
- [x] SHA256 recorded for 16 core files
- [x] py_compile PASS (all runnable scripts)
- [x] 39 tests PASS
- [x] git_dirty=0 (clean snapshot)
- [x] Snapshot report + SHA CSV written
