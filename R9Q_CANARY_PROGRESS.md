# R9Q Corrected Online Canary — Progress Report

**Date:** 2026-07-13 11:25 CST
**Server:** pm-364c0001 (dty_user@10.60.2.56:33571)

## Git

```
BRANCH = deepseek/c2g-r9q-corrected-online-canary-20260713
HEAD = a89db95af98ce3a23fd4e759136df753e4563b4f
DRAFT_PR = #73 (base: PR #72 codex/c2g-r9q-final-detector-attack-20260713)
WORKTREE = /mnt/sdc/dty_user/openvla_attack_deepseek_r9q_corrected_canary_20260713
WORKTREE_CLEAN = true
```

## Changes from PR #72 baseline

1. **`tools/multisuite_detector/build_c2g_r9q_attack_manifest.py`**: Fixed GPU suite mapping — object+spatial → GPU6, goal+l10 → GPU7 (was interleaved 8-worker)
2. **`scripts/stageb/run_c2g_r9q_attack_scheduler.py`**: Reduced to 4 workers, dynamic max_wpg, default 2 workers/GPU
3. **`tools/multisuite_detector/rebuild_c2g_r9q_corrected_canary_manifest.py`**: Manifest rebuilder — preserves parent identities from old canary, corrects GPU assignment

## Detector Bundle (frozen)

```
Path: c2g_r9q_final_detector_bundle_5576d46_20260713_v1
checkpoint.pt SHA256: 336a7723...
detector_config.json SHA256: 196966c8...
normalization.json SHA256: 1b31424a...
τ_critical=0.7  τ_release=0.4  τ_ground=0.3  persistence=2-of-3  burst_length=10
susceptibility_gate_enabled = false
```

## Canary Manifest

```
NEW_CANARY_ROOT = c2g_r9q_corrected_online_canary_a89db95_20260713_v3
Source: Original canary manifest c2g_r9q_attack_canary_8ddbd035_20260713_v1
PARENTS = 8 (all match old canary) ✅
CELLS = 32 (8 parents × 4 conditions) ✅
```

| Parent | Suite | Conditions |
|--------|-------|-----------|
| libero_object/task_01/state_015/clean/attempt_01 | object | CLEAN, R9Q, RAND, ORACLE |
| libero_object/task_03/state_025/clean/attempt_01 | object | CLEAN, R9Q, RAND, ORACLE |
| libero_spatial/task_02/state_022/clean/attempt_01 | spatial | CLEAN, R9Q, RAND, ORACLE |
| libero_spatial/task_07/state_018/clean/attempt_01 | spatial | CLEAN, R9Q, RAND, ORACLE |
| libero_goal/task_01/state_023/clean/attempt_01 | goal | CLEAN, R9Q, RAND, ORACLE |
| libero_goal/task_01/state_024/clean/attempt_01 | goal | CLEAN, R9Q, RAND, ORACLE |
| libero_10/task_02/state_016/clean/attempt_01 | l10 | CLEAN, R9Q, RAND, ORACLE |
| libero_10/task_04/state_043/clean/attempt_01 | l10 | CLEAN, R9Q, RAND, ORACLE |

## Worker Deployment

```
GPU6_CAP = 2 (g6_object + g6_spatial)
GPU7_CAP = 2 (g7_goal + g7_l10)
Worker budget: 16000 MiB
GPU reserve: 4000 MiB

GPU6 free: 29.2 GB → 10.8 GB (after load)
GPU7 free: 40.0 GB → 7.8 GB (after load)
```

## Progress: 8/32 cells (25%)

**All cells: 0 failed, susceptibility_gate_enabled=False, runtime_valid=True**

| Condition | Completed | Key Findings |
|-----------|-----------|-------------|
| CLEAN | 4/8 | All true clean (no attack), trig detection working |
| R9Q_DETECTOR_T10 | 4/8 | All exact T10 (atk=10), one-shot, varied trigger steps |
| RAND_T10 | 0/8 | Pending |
| COMMAND_OPEN_ORACLE | 0/8 | Pending |

### R9Q Trigger Evidence

| Parent | Trigger Step | Attack Count | Success |
|--------|-------------|-------------|---------|
| libero_object/task_01/state_015 | **64** (matches known evidence) | 10 | False (task not completed) |
| libero_spatial/task_02/state_022 | 58 | 10 | True |
| libero_goal/task_01/state_023 | 165 | 10 | False |
| libero_10/task_02/state_016 | 81 | 10 | False |

### Validation

- [x] susceptibility_gate_enabled = False on ALL 8 cells
- [x] detector_effective_valid tracked per-step
- [x] R9Q triggers: one-shot, exact T10, no multi-trigger
- [x] Known regression targets confirmed (object/task_01/state_015 step 64)
- [x] Trigger diversity: 4 different trigger steps (58, 64, 81, 165) — not uniform
- [ ] object/task_03/state_025 R9Q (known trigger step 136) — pending
- [ ] All 8 RAND_T10 cells — pending
- [ ] All 8 COMMAND_OPEN_ORACLE cells — pending
- [ ] Pre-trigger prefix consistency audit — pending

## Estimated Completion

~60 minutes total runtime for 32 cells. Currently ~15 min elapsed, 8/32 done. ETA 11:50 CST.

## Next Steps

1. Canary complete (32/32 cells)
2. Formal audit with cross-suite gate
3. GPU release
4. Server-side code commit
