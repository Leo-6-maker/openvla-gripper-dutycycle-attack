# VIS Correct Decode — Final Handoff

**Date**: 2026-06-01 | **Branch**: `exp/vis-payload-upgrade-validation-20260601` | **HEAD**: `8ff150d8`

## 1. Environment

- Python env: `openvla_official_libero_20260525` (corrected)
- Model: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`
- Invalid outputs: `/data/liuyu/outputs/vis_payload_contact_frame_collection_20260601/` (wrong env)

## 2. Decode Bug and Fix

**Bug**: Direct `processor(text, image)` without OpenVLA `prompt()` wrapper and missing action prefix token 29871. All old no-rollout diagnostics produced invalid grip values (EOS token mapped to bin 254 = 0.996078).

**Fix**: Apply `prompt(instruction)` → `processor(prompt_text, image)` → append token 29871 → generate 7 action tokens.

## 3. Key Results (Correct Decode Only)

### Deterministic Repeat (Phase B, 10 seeds)
- ketchup_0098: **10/10** grip 0.0→0.996, armL2=0.839
- tomato_0134: 0/10 no change
- Random baseline: **0/5** on all frames

### Multi-Frame (Phase C, 3 seeds each)
- **Tomato HIGH-SENSITIVE positive frames**: step_0130 (armL2=0.13), step_0138 (armL2=0.16)
- **Cream Cheese HIGH-SENSITIVE positive frames**: step_0070 (armL2=**0.11**, BEST), step_0065, 0080, 0085
- Ketchup positive frames: step_0096, 0098, 0110

### Arm Specificity
- Best: cream_cheese_0070 armL2=0.11 with full grip flip
- Cream_cheese_0075 proves specificity: armL2=1.07 but grip delta=0

## 4. GPU Status

| GPU | Status |
|-----|--------|
| 0 | Historical Xid — avoid |
| 1 | OK |
| 2 | OK |
| 3 | **Xid 31 @ 14:15 — QUARANTINED** |
| 4 | OK |
| 5 | OK |
| 6 | OK |
| 7 | **Xid 13 — PERMANENTLY DAMAGED** |

Healthy: 1,2,4,5,6 (5 GPUs, 2 usable pairs)

## 5. Decision

**Case A — Strong No-Rollout PASS**

Visual-input perturbation (`gripper_open_region_ce`, eps=4/255, steps=20) can reliably change the decoded gripper action from neutral (0.0) to fully open (0.996) on contact-phase frames across multiple LIBERO-Object tasks, including high-sensitive tasks (tomato_sauce, cream_cheese). Random same-Linf perturbation does not reproduce the effect. Arm drift is controlled on best frames (0.11 L2) and the effect is not arm-drift-dominated.

## 6. Rollout Status

**BLOCKED** — Do not execute rollout. Write controlled rollout proposal only. Requires Leon approval.

## 7. Artifacts

### Reports
- `reports/VIS_DECODE_PATH_BUGFIX_AUDIT.md`
- `reports/VIS_GRIPPER_ACTION_SEMANTICS_AUDIT.md`
- `reports/VIS_REPEATABILITY_CORRECT_DECODE.md`
- `reports/VIS_ARM_DRIFT_AUDIT.md`
- `reports/VIS_CORRECT_DECODE_DECISION.md`
- `reports/VIS_CORRECT_DECODE_FINAL_HANDOFF.md`

### Tables
- `tables/vis_decode_path_before_after_examples.csv`
- `tables/vis_repeatability_correct_decode.csv` (partial, recovered from log)
- `tables/vis_cc_scan.csv` (cream_cheese results)

### Backup
- `/data/liuyu/outputs/code_backups/vis_payload_upgrade_20260601/` (Phase 0 backup)
- Patch: `worktree_diff_before_continuation.patch` (130 KB)

## 8. Next Recommendation

Write controlled rollout proposal for tomato_sauce and cream_cheese (high-sensitive) with:
- Best frames identified (tomato_0130/0138, cream_cheese_0070)
- eps=4/255, steps=20, gripper_open_region_ce
- Validated decode path
- Random baseline
- Max 16 rollouts
- GPU2,4 or GPU5,6 pair
