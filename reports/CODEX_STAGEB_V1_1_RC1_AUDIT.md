# Codex Stage-B v1.1 RC1 Audit

Date: 2026-06-07

Audited branch: `exp/vis-prefix-margin-repair-20260603`

Audited remote commit: `bbd2d989adab61bd60a68c2f0a78932414eed8ff`

Mode: CPU-only release-candidate audit. No GPU, VIS, rollout, watcher, or server live output mutation was run.

## Verdict

`RC1_CODE_SEMANTICS_PASS_SERVER_SNAPSHOT_PARTIAL`

The RC1 code semantics are consistent with the Stage-B v1.1 contract:

- runner imports `openvla_libero_exec_spec`
- trace version is `corrected_stageb_v1_1`
- prompt style is `official_in_out`
- image preprocessing style is explicitly recorded
- qpos source is `obs_robot0_gripper_qpos`
- `decoded_open_bool` is derived from `env_gripper_is_open`
- v1.1 postprocess and label builder hard-fail pre-v1.1 traces
- old labels are explicitly quarantined

The release-candidate package is not fully reproducible from the server snapshot alone: the server reviewed worktree is dirty/manual-uploaded, and 8 files listed in the RC1 manifest were missing from the server path during this audit. Core runtime file hashes recorded in the freeze report do match the server snapshot, but the full manifest does not.

## Required Checks

| Check | Status | Evidence |
|---|---:|---|
| Freeze manifest contains all critical files | PARTIAL | Manifest has 38 rows and includes core runner/spec/postprocess/label-builder files, but omits `scripts/diagnostics/generate_stageb_worker_scripts.py`, which is also a Stage-B worker generator. |
| Runner imports `openvla_libero_exec_spec` | PASS | `scripts/run_stageb_vis_labeling.py` imports `OPENVLA_LIBERO_EXEC_SPEC_VERSION`, `official_prompt`, gripper semantic helpers, and `get_libero_image_official`. |
| `trace_version` is `corrected_stageb_v1_1` | PASS | Runner sets `TRACE_VERSION = 'corrected_stageb_v1_1'`; postprocess/label-builder require the same value. |
| `prompt_style` is `official_in_out` | PASS | Runner sets `PROMPT_STYLE = 'official_in_out'` and uses `official_prompt(...)`. |
| `image_preprocess_style` is explicit | PASS | Runner records `official_rot180_only` or `legacy_direct_agentview_no_rotation`; field is not empty. |
| `qpos_source` is obs gripper qpos | PASS | Runner writes `qpos_source = 'obs_robot0_gripper_qpos'`. |
| `decoded_open_bool` derives from spec helper | PASS | Runner computes `decoded_open_bool = int(env_gripper_is_open(env_action_6))`. |
| Postprocess hard-fails pre-v1.1 traces | PASS | `postprocess_traces_v1_1.py` rejects non-`corrected_stageb_v1_1` traces and exits non-zero when old-format traces are present. |
| Label builder hard-fails pre-v1.1 traces | PASS | `build_pair_labels_v1_1.py` exits non-zero for non-v1.1 rows, unexpected conditions, duplicate pairs, unpaired pairs, and unreachable/no-intervention windows. |
| Old labels quarantine is explicit | PASS | Freeze report declares `QUARANTINED_OPEN_SEMANTICS_INVERTED_OR_UNVERIFIED` for all pre-RC1 Stage-B labels and pre-`corrected_stageb_v1_1` traces. |
| Server snapshot matches RC1 manifest | PARTIAL_FAIL | Server core runtime SHA values match the freeze report, but the server worktree is dirty/manual-uploaded and 8 manifest paths were missing on server. |

## Local Validation

Local `py_compile` on the audited worktree:

```text
src/gripper_attack/openvla_libero_exec_spec.py
scripts/run_stageb_vis_labeling.py
scripts/stageb/postprocess_traces_v1_1.py
scripts/stageb/build_pair_labels_v1_1.py
scripts/stageb/validate_stageb_trace_v1_1.py
```

Result: PASS.

Local test run over the 12 RC1 test files:

```text
41 passed in 0.27s
```

The freeze test table lists 12 test files, all PASS.

## Server Snapshot Audit

Observed server path:

```text
/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605
```

Observed server git state:

```text
HEAD=ca3a97e73965e2582e28066a93892e3dd9c24617
dirty/manual-uploaded files present
```

Core server SHA values:

| File | Server SHA | RC1 freeze report |
|---|---|---|
| `src/gripper_attack/openvla_libero_exec_spec.py` | `4fe01c430b2f0ec0125c77bd7cbcb7f34de24440ffa85db87f007117ac2aa5ac` | match |
| `scripts/run_stageb_vis_labeling.py` | `d37d8f49c2a903ce8ab729309c17f6d83a72ea8ad9eab5e77d3949ca24b15329` | match |
| `scripts/stageb/build_pair_labels_v1_1.py` | `0dbe18a3a593b94fce4eeb1a6ca9dd1491432d21b69113ed3fe23aa390303718` | match |
| `scripts/stageb/validate_stageb_trace_v1_1.py` | `838159d289eec586dca1a24d77cca01cbd4890adae7a6f42bfa2e0aec04b54dd` | match |

The local Windows raw SHA for `openvla_libero_exec_spec.py` and `validate_stageb_trace_v1_1.py` differs because the worktree uses CRLF line endings. LF-normalized local SHA matches the server/freeze SHA.

Manifest paths missing on server:

```text
tests/stageb/test_attack_open_token_region.py
tests/stageb/test_trace_schema_v1_1.py
tests/stageb/test_pair_label_builder_v1_1.py
tests/stageb/test_old_label_quarantine.py
tests/stageb/test_random_linf_metadata.py
tests/stageb/test_stageb_open_count_convention.py
reports/OPENVLA_LIBERO_EXECUTABLE_SPEC.md
reports/CODEX_STAGEB_V1_1_SMOKE_R3_LIVE_AUDIT.md
```

## Blocking Issues

No blocker was found in the RC1 core executable semantics.

The release package has a reproducibility issue: the server snapshot is not a complete copy of the RC1 manifest. This does not invalidate the smoke runner output already audited, but it should be fixed before calling the server checkout a complete RC1 source snapshot.

## Non-Blocking Issues

- Freeze coordinates list the code baseline commit `ca570704...`, while the freeze report itself was committed at `bbd2d98...`. This is acceptable if documented as "code baseline vs report commit", but it is currently easy to misread.
- Manifest row count is 38 in the audited CSV, not the 32-file count stated in the handoff note.
- `official_rot180_only` remains a known limitation and is not full Octo resize.
- Old queue windows remain unsuitable under v1.1 official preprocessing; reachability scan is still required.

## Recommendation

RC1 can be used as the code baseline for a v1.1 clean reachability scan, provided the run uses the synced core runtime files and does not consume old labels.

Before a larger smoke or batch, sync the missing manifest test/report files to the server or clearly scope the server snapshot as "runtime-core only", not "full RC1 manifest".
