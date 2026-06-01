# VIS Payload Continuation — Phase 0 Status

**Date**: 2026-06-01 | **Continuation start**: 12:13:44

## Environment State

- **Branch**: `exp/vis-payload-upgrade-validation-20260601`
- **HEAD**: `8ff150d8d11e4e4001d1a3fc786ea09738471ae5`
- **Python env**: `openvla_official_libero_20260525` (corrected, NOT openvla_compat)
- **Model**: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object` — present

## Modified Tracked Files
- `scripts/run_official_eval_artifact_rich.py` (1339 lines changed)
- `src/gripper_attack/attack_adapter.py` (938 lines changed)

## Key Untracked Artifacts
- `reports/VIS_PAYLOAD_CONTACT_FRAME_VALIDATION.md`
- `reports/VIS_PAYLOAD_UPGRADE_PHASE0_PREFLIGHT.md`
- `reports/VIS_ARM_DRIFT_DIAGNOSTIC_PLAN.md`
- `reports/VIS_TOKEN_FLIP_THRESHOLD_DIAGNOSTIC_PLAN.md`
- `reports/VIS_TOKEN_FLIP_THRESHOLD_DIAGNOSTIC_RUN.md`
- `scripts/diagnostics/` (8 diagnostic scripts)
- `tests/v4/` (VIS-related tests)
- `tables/` (uncommitted tables)
- `configs/` (several schema/config files)

## GPU Status
| GPU | Model | Memory | Utilization | Status |
|-----|-------|--------|-------------|--------|
| 0 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE (avoid: lgzhou processes) |
| 1 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE, OK |
| 2 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE, OK |
| 3 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE, OK |
| 4 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE, OK (used in prior sweep) |
| 5 | RTX 2080 Ti | 0/11264 MiB | 1% | IDLE, OK (used in prior sweep) |
| 6 | RTX 2080 Ti | 0/11264 MiB | 0% | IDLE, OK |
| 7 | RTX 2080 Ti | 0/11264 MiB | 0% | QUARANTINED (Xid 13 @ 2026-06-01 10:00:34) |

## Xid Status
- GPU7: Xid 13 (Misaligned Address) + Xid 43 @ 2026-06-01 10:00:34 — confirmed historical, still present
- No new Xid since last session
- All other GPUs clean

## Output Roots
- **Valid**: `/data/liuyu/outputs/vis_payload_upgrade_contact_frames_20260601/`
- **Valid**: `/data/liuyu/outputs/vis_contact_frame_dump_clean_20260531/`
- **INVALID**: `/data/liuyu/outputs/vis_payload_contact_frame_collection_20260601/` (wrong env: openvla_compat)
- **INVALID**: `/data/liuyu/outputs/vis_payload_contact_frame_collection_20260601_official/` (verify status)

## Backup Status
- Patch: `/data/liuyu/outputs/code_backups/vis_payload_upgrade_20260601/worktree_diff_before_continuation.patch` (130 KB)
- Git status: `/data/liuyu/outputs/code_backups/vis_payload_upgrade_20260601/git_status_before_continuation.txt`
- HEAD: `/data/liuyu/outputs/code_backups/vis_payload_upgrade_20260601/head_before_continuation.txt`

## Disk
- `/data/liuyu`: 612G used / 1.8T total (36%) — healthy
