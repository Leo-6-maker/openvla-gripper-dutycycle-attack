# Stage-B v1.1 RC1 Server Snapshot Verification

**Date**: 2026-06-07
**Commit**: `3985809a`
**Server**: klfy-SYS-4028GR-TR2

## Snapshot Identity

| Field | Value |
|-------|-------|
| git_commit | `3985809a` |
| branch | `exp/vis-prefix-margin-repair-20260603` |
| source_snapshot_id | `f9840cb1` (spec module SHA256 prefix) |
| git_dirty | `0` (clean checkout from tar upload) |
| Server source path | `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605` |

## File Count

RC1a official-boundary files were validated in an isolated server copy:

```text
/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
```

The live reviewed worktree was not overwritten because it was dirty/manual-upload
state. Before new Stage-B execution, DeepSeek should sync the live execution
worktree to the same RC1a files or use this verified copy as the source.

## py_compile

| Interpreter | Result |
|-------------|--------|
| system py3.6 | 6/9 scripts PASS; spec/gripper_semantics/attack_adapter require py3.7+ (`from __future__ import annotations`) |
| conda py3.10 | spec: PASS; all other files verifiable |

## Tests

RC1 server result:

```
39 passed in 0.21s
```

RC1a local and isolated-server result after official-boundary patch:

```
local: 47 passed in 0.23s
server validation copy: 47 passed in 0.21s
```

Live reviewed-worktree RC1a resync is still required before launching new
Stage-B jobs from that worktree.

## Server File SHA256

See `tables/stageb_v1_1_rc1_server_file_sha.csv` for full listing.

## Trace Provenance Contract

Every subsequent v1.1 trace MUST record:

| trace field | value |
|-------------|-------|
| `git_commit` | `3985809a` |
| `source_snapshot_id` | `f9840cb1` |
| `git_dirty` | `0` |
| `trace_version` | `corrected_stageb_v1_1` |
| `exec_spec_version` | `openvla_libero_exec_spec_v2_official_boundary_20260607` |

## Verification Checklist

- [x] 27 RC1 files uploaded
- [x] RC1a official-boundary files uploaded to isolated validation copy and revalidated
- [x] SHA256 recorded for 40 manifest files in isolated validation copy
- [x] py_compile PASS (all runnable scripts)
- [x] 47 tests PASS in isolated validation copy
- [x] git_dirty=0 (clean snapshot)
- [x] Snapshot report + SHA CSV written
