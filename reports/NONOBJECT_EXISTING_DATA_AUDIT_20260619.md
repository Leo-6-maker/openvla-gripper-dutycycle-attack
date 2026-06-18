# Non-Object Existing CLEAN Data Audit - 2026-06-19

## Scope

Read-only metadata-only audit of `/data/liuyu/outputs`. No GPU, OpenVLA load, rollout, file mutation, file move, or large MP4/NPZ full SHA was performed.

## Run Metadata

```json
{
  "active_end": {
    "active_root": "/data/liuyu/outputs/cross_suite_clean_300_20260619_r1_6379397",
    "complete_count": 51,
    "output_root_mtime": 1781809083.258488,
    "queue_pid": "26816",
    "queue_status_exists": true,
    "rows": 52,
    "running_count": 1
  },
  "active_root": "/data/liuyu/outputs/cross_suite_clean_300_20260619_r1_6379397",
  "active_start": {
    "active_root": "/data/liuyu/outputs/cross_suite_clean_300_20260619_r1_6379397",
    "complete_count": 51,
    "output_root_mtime": 1781809083.258488,
    "queue_pid": "26816",
    "queue_status_exists": true,
    "rows": 52,
    "running_count": 1
  },
  "audit_end_time": "2026-06-19T06:52:59+08:00",
  "audit_source_commit": "14894f4c04c8269bbb327e51856b0b8e787cf7b1",
  "audit_start_time": "2026-06-19T06:52:49+08:00",
  "outputs_root": "/data/liuyu/outputs",
  "reference_schema_commit": "63793972743f667c6a6bcc12e9700f322f261147",
  "rules": {
    "active_root_separate": true,
    "anonymous_pose_not_identity": true,
    "condition_unverified_not_clean": true,
    "metadata_only": true,
    "no_gpu": true,
    "no_large_mp4_npz_full_sha": true
  },
  "scan_counters": {
    "bytes_seen": 35921005119,
    "dirs_scanned": 11669,
    "files_seen": 462267
  },
  "sentinels": [
    "detector_telemetry.csv",
    "episode_manifest.json",
    "episode_records.jsonl",
    "episode_summary.json",
    "results.json",
    "run_manifest.json",
    "step_records.jsonl",
    "step_telemetry.csv",
    "summary.json",
    "video_manifest.json"
  ],
  "server_hostname": "klfy-SYS-4028GR-TR2"
}
```

## Summary

- Candidate sentinel directories discovered: 3575
- Clean denominator usable episodes: 9
- Detector-transfer usable episodes: 9
- Teacher/offline relabeling usable episodes: 2
- Tier counts: {'TIER_X_REJECTED': 3486, 'CURRENT_IN_PROGRESS': 51, 'TIER_D_LEGACY_REFERENCE_ONLY': 29, 'TIER_B_DETECTOR_COMPATIBLE': 9}

## Current 300 Queue Snapshot

- Active rows observed: 52
- Active COMPLETE stable rows: 50
- Active RUNNING rows: 1

Current active root is reported separately and not mixed with legacy counts.

## Reusability By Suite

```text
verified_spatial_unique_clean_states=2
verified_goal_unique_clean_states=2
verified_libero10_unique_clean_states=2
```

| Suite | Clean usable | Detector usable | Teacher relabel usable | Unique task-states | Missing 10x10 | Missing 10x50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_10 | 3 | 3 | 1 | 2 | 98 | 498 |
| libero_goal | 2 | 2 | 0 | 2 | 98 | 498 |
| libero_spatial | 4 | 4 | 1 | 2 | 98 | 498 |

## Decision Table

- Directly reusable: `TIER_A_FULL_CURRENT_COMPATIBLE` and `TIER_B_DETECTOR_COMPATIBLE`, depending on whether same-freeze comparison is required.
- Detector analysis but not Teacher: episodes with `usable_detector_transfer_analysis=true` and `usable_teacher_relabeling=false`.
- Clean SR only: `TIER_C_CLEAN_DENOMINATOR_ONLY`.
- Rerun required: `TIER_X_REJECTED`, partial/current-running episodes, attack-contaminated outputs, and any episode with unresolved condition or suite.
- Offline relabel without rerun: episodes with generic sim-state or complete frames but `privileged_valid=false`.

## Claim Limits

- This audit does not prove detector timing transfer.
- Anonymous pose streams were not treated as object identity evidence.
- Different source commits are not same-freeze compatible unless explicitly marked.
- Active queue COMPLETE snapshots are separated from legacy inventory.
