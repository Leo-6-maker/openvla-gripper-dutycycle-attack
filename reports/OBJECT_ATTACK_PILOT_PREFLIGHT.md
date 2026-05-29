# Object Matched Attack Pilot — Preflight Checklist

**Date**: 2026-05-29
**Status**: BLOCKED on GPU2,6 (Goal-100: 63/100, ~3h ETA)

## 1. Hardware

| Check | Status | Detail |
|-------|--------|--------|
| GPU2,6 free | ⏳ WAITING | Goal-100: 63/100 eps, ETA ~3h |
| GPU2 Xid history | ⚠️ Xid31 May 28 | Historical only. Process survived. |
| GPU0 excluded | ✓ | Confirmed CUDA illegal memory access |
| GPU7 excluded | ✓ | OOM at 10.66/10.75 GB for inference |
| GPU3 status | ⚠️ Xid31 May 29 11:42 | MMU fault on L10-B. Process survived. Affected shards after 11:42 should be quarantined. |
| Disk | ✓ | 35% (603G/1.8T), 1% inodes |
| Fresh Xid | ⚠️ GPU3 only | GPU2,6 clean today |

## 2. Software

| Check | Status | Detail |
|-------|--------|--------|
| Detector checkpoint | ✓ | `ProprioNoStep_baseline.pt`, sha256: `4b3f3d47...dd9c7b1f` |
| Patched runner | ✓ | `run_official_eval_artifact_rich.py`, hash: `8a150437` |
| Repo commit | ✓ | `da4b297 Freeze Milestone 2C proprio causal student baseline` |
| Attack conditions | ✓ | clean, oracle_open, random_control, VIS_targeted |
| Detector config | ✓ | ht=0.1, dur=5, cooldown=0 |

## 3. Detector Verification

| Check | Status |
|-------|--------|
| Shadow validation coverage | 97.8% ✓ |
| Shadow miss rate | 0/9 ✓ |
| Shadow FP on failed | 0 ✓ |
| Offline replay coverage | 99.1% (ProprioNoStep) |
| Fusion best strategy | VisualOnly (84.5%), ProprioNoStep primary |
| normalized_step in student | NO ✓ |
| Privileged state in student | NO ✓ |
| Attack outcomes in detector | NO ✓ |

## 4. Attack Pilot Design

- 2 tasks × 5 states × 4 conditions = **40 rollouts**
- Tasks: cream_cheese, milk
- Conditions: clean, oracle_open, random_control, VIS_targeted
- One detector trigger for ALL conditions
- Attack module does NOT reselect windows

## 5. Actions After GPU2,6 Free

1. Verify no fresh Xid on GPU2 or GPU6
2. Verify Goal-100 worker exited cleanly
3. `mkdir -p /data/liuyu/outputs/milestone_2f_object_detector_matched_attack_pilot_20260529/{logs,tables,reports}`
4. Launch 4 conditions sequentially on GPU2,6
5. Monitor for Xid after each condition

## 6. Blocking Conditions

- [ ] GPU2,6 free
- [ ] No fresh Xid on GPU2,6
- [ ] Goal-100 postprocess not started (don't mix outputs)
- [ ] User approval for attack pilot launch
