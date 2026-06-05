# Server Read-Only Verification 2026-06-05

Command boundary: SSH read-only checks only. No GPU command, rollout, VIS job, watcher, training job, or server output mutation was run.

SSH route:

```bash
ssh -J scene@10.60.133.3 liuyu@10.60.133.4
```

Server repo checked:

```bash
/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
```

## Key Result

**Blocking mismatch:** the server repo is not currently on the handoff branch.

| Check | Server Result |
|---|---|
| Host | `klfy-SYS-4028GR-TR2` |
| Server time | `2026-06-05 10:52:15 CST` |
| Current branch | `exp/vis-payload-upgrade-validation-20260601` |
| Current HEAD | `653ed33d78578aa0f0af96539a9c8b4c2a6d4c08` |
| Expected handoff branch | `exp/vis-prefix-margin-repair-20260603` |
| Branch present locally on server | Not listed |
| Local Codex commit present on server | `3a35239` not present |
| Python env path | `/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python` |
| Python version | `Python 3.10.13` |

## Server File Presence

| File | Server Status |
|---|---:|
| `reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` | Missing |
| `reports/HANDOFF_CONSISTENCY_AUDIT_20260605.md` | Missing |
| `reports/DETECTOR_TRAINING_SCRIPT_AUDIT.md` | Missing |
| `reports/DETECTOR_V2_AND_VISUAL_TRANSFER_DESIGN.md` | Missing |
| `reports/LABEL_BUILDER_PATCH_PLAN.md` | Missing |
| `reports/CODEX_PARALLEL_REVIEW_SUMMARY_20260605.md` | Missing |
| `scripts/diagnostics/audit_label_schema.py` | Missing |
| `scripts/diagnostics/generate_window_compression_candidates.py` | Missing |
| `scripts/diagnostics/finalize_phase_response_labels.py` | Exists, untracked |
| `scripts/train_vulnerability_ready_detector_v1.py` | Exists, untracked |
| `tables/object_phase_response_batch3_vis_summary.csv` | Exists, untracked |
| `tables/object_phase_response_labels_v1.csv` | Exists |
| `tables/object_phase_response_labels_v2.csv` | Missing |
| `tables/object_window_compression_candidates.csv` | Missing |

## Script Capability Check

`finalize_phase_response_labels.py --help` on the server supports:

- `--batch1-merged`
- `--batch2b-vis`
- `--batch3-vis`
- `--output-labels`

It does not show:

- `--batch3b-vis`
- `--batch3c-vis`

`train_vulnerability_ready_detector_v1.py --help` on the server supports:

- `--labels-csv`
- `--freeze-v1-hardcoded`
- `--min-rows`

The server copy is untracked and should not be assumed to match the committed local script without diff inspection.

## Output Directory Check

| Output Dir | Server Status |
|---|---:|
| `/data/liuyu/outputs/nightly_object_batch3_20260604` | Exists |
| `/data/liuyu/outputs/nightly_object_batch3b_20260604` | Exists |
| `/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604` | Exists |

No files in those output directories were modified.

## Implications

1. DeepSeek should not train detector v2 from this server checkout yet.
2. The server must first switch/sync to the intended branch or receive a controlled patch bundle.
3. `object_phase_response_labels_v2.csv` is still missing on server.
4. The server label builder still lacks Batch3b/Batch3c CLI support.
5. The local Codex audit commit `3a35239` is not present on server, so server-side scripts/reports are stale relative to this review.

## Safe Next Step

Before any server-side label merge or detector training:

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git fetch origin
git checkout exp/vis-prefix-margin-repair-20260603
git log --oneline -3
```

If that branch is not available on the server, do not improvise from untracked scripts. Sync the reviewed commit/patch deliberately, then rerun only CPU-only schema checks first.
