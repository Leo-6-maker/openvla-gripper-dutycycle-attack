# Stage-B v1.1 RC1 Freeze Report

**Date**: 2026-06-07
**Status**: RC1 (not final — known limitations acknowledged)
**Review**: Codex R3 smoke live audit PASS (schema only, not scientific result)

## 1. Freeze Coordinates

| Field | Value |
|-------|-------|
| Branch | `exp/vis-prefix-margin-repair-20260603` |
| Commit SHA | `ca570704b56c7e424a92a37813036f7b2c631419` |
| Tag | None (RC1, not final) |
| Spec version | `openvla_libero_exec_spec_v2_official_boundary_20260607` |
| Trace version | `corrected_stageb_v1_1` |
| Runner version | `stageb_vis_labeling_v1_1_spec_aligned_20260607` |

## 2. Canonical Semantics

| Rule | Value |
|------|-------|
| raw_gripper OPEN threshold | `> 0.5` |
| env_action_6 OPEN | `< -0.5` (env = -1.0) |
| env_action_6 CLOSE | `> +0.5` (env = +1.0) |
| raw_gripper boundary | `== 0.5` → neutral/excluded |
| open_token_ids | `{tid \| decoded raw > 0.5}` |
| qpos source | `obs["robot0_gripper_qpos"]` |
| qpos aggregation | `abs(q0) + abs(q1)` |
| shifted qpos | `step_dict[s+1]` |
| prompt style | `official_in_out` (`In: ...?\nOut:`) |
| image preprocess | `official_rot180_only` |
| unnorm key | `libero_object` |
| perturbation space (VIS) | `processor_pixel_values_linf` |
| perturbation space (random) | `random_linf_processor_pixel_values` |

## 3. Old Conventions — Quarantined

```
env_gripper > 0 = OPEN              # WRONG
raw_gripper < 0.5 = OPEN            # WRONG
decoded_action < 0.5 = OPEN         # WRONG
qpos = sim.data.qpos[-2:]           # WRONG
chat-style prompt                   # LEGACY only
no image rotation                    # LEGACY only
filename-based condition parsing     # BROKEN
old summary qpos_delta               # UNRELIABLE
pre corrected_stageb_v1_1 traces    # REJECTED
```

## 4. RC1 File Manifest

See `tables/stageb_v1_1_rc1_file_manifest.csv` for full listing.

Core files:

| File | Purpose |
|------|---------|
| `src/gripper_attack/openvla_libero_exec_spec.py` | Executable spec — single source of truth |
| `src/gripper_attack/gripper_semantics.py` | Spec wrapper |
| `src/gripper_attack/attack_adapter.py` | Token region + PGD attacker |
| `scripts/run_stageb_vis_labeling.py` | v1.1 labeling runner (53-column trace) |
| `scripts/generate_stageb_worker_scripts.py` | Worker script generator with shared --pair_id |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | Trace schema validator |
| `scripts/stageb/postprocess_traces_v1_1.py` | v1.1-only qpos postprocess |
| `scripts/stageb/build_pair_labels_v1_1.py` | v1.1-only pair label builder |
| `tests/stageb/test_*.py` | Test suite |

## 5. Test Results

See `tables/stageb_v1_1_rc1_test_results.csv` for per-test details.

- Pure-Python (local): 13/13 Stage-B test files PASS
- Local full suite (Codex): 47 passed
- Server validation copy: 47 passed
- Key assertions verified:
  - `raw_gripper_is_open(0.996)` → True
  - `env_gripper_is_open(-1.0)` → True
  - `open_token_ids` correctly classify saturation tokens
  - v1.1 trace columns all present (53 cols)
  - old trace_version rejected
  - unreachable windows rejected by label builder
  - raw gripper boundary `0.5` is neutral/excluded, matching official `np.sign`

## 6. Smoke Validation (R3)

- 6/6 traces passed schema validator
- 2/3 windows unreachable (butter/ketchup window too late for official trajectory)
- 1/3 reachable (cream_cheese s7), VIS=0 open
- Label builder correctly rejects unreachable windows (n_window_steps=0 → hard-fail)

## 7. Known Limitations

| Limitation | Impact |
|------------|--------|
| `official_rot180_only` — no full Octo resize | Minor: resizing differs from official eval |
| Old queue windows may be unreachable under v1.1 | Requires v1.1 clean trajectory rescan |
| Server repo at `ca3a97e` (no git pull access) | Requires manual upload for sync |
| `git_dirty=1` on server | Trace provenance slightly degraded |
| No full 44-row v1.1 rerun | Only 3-window smoke validated |
| cream_cheese only reachable smoke pair | Cannot generalize VIS effectiveness |

## 8. Server Sync Record

| Field | Value |
|-------|-------|
| Server | klfy-SYS-4028GR-TR2 |
| Source path | `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605` remains untouched; RC1a validation copy is `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607` |
| git_commit | `3985809a` |
| git_dirty | `0` (clean tar upload) |
| source_snapshot_id | `f9840cb1` |
| Files synced | RC1a validated in isolated server copy, not live reviewed worktree |
| py_compile (system py3.6) | RC1 result: 6/6 scripts PASS; 3 src files need py3.7+ |
| py_compile (conda py3.10) | RC1a local: PASS; RC1a server validation copy: PASS |
| pytest (conda py3.10) | RC1a local: **47 passed** in 0.23s; server validation copy: **47 passed** in 0.21s |

### Trace provenance contract

Every subsequent v1.1 trace MUST record:
- `git_commit = 3985809a`
- `source_snapshot_id = f9840cb1`
- `git_dirty = 0`
- `trace_version = corrected_stageb_v1_1`
- `exec_spec_version = openvla_libero_exec_spec_v2_official_boundary_20260607`

Full SHA256 table: `tables/stageb_v1_1_rc1_server_file_sha.csv`
Server verification report: `reports/STAGEB_V1_1_RC1_SERVER_SNAPSHOT_VERIFY.md`

## 9. Quarantine Rule

All Stage-B labels generated before this RC1 freeze must carry:
```
QUARANTINED_OPEN_SEMANTICS_INVERTED_OR_UNVERIFIED
```

This includes:
- Old overnight labels (Batch 2b/3b/3c)
- 44-row patched rerun (inverted objective diagnostic)
- Any pre corrected_stageb_v1_1 traces

## 9. RC1 Acceptance Criteria

- [x] Spec module self-check passes
- [x] 24+ tests pass
- [x] py_compile on all staged Python files
- [x] v1.1 trace schema validated with live smoke
- [x] Label builder hard-fails on unreachable windows
- [x] open_token_ids verified against spec
- [x] pair_id shared between VIS/random in worker scripts
- [x] RC1a isolated server validation copy uploaded and verified
- [ ] Live reviewed worktree resynced for execution
- [ ] v1.1 clean reachability scan completed
- [ ] Windows selected from v1.1 clean trajectories
