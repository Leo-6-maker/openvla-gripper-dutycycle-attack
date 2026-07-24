# Object Table 1 — Complete Experimental Progress Report

**Updated: 2026-06-27 06:35 CST**
**Git commit: `8b25239`**

---

## Executive Summary

Table 1 data acquisition is complete. Reference panel (5-seed 2×2), mechanism panel (108 metric refresh), baseline (30 Untargeted CE PGD), CLEAN determinism (18), and Object breadth panel (120 formal + 16 repeat = 136) are all finished. Total: ~526 scientific runs, 331 COMPLETE.json in phase7_object directory.

## Phase Status

| Phase | Content | Runs | Status | Gate |
|-------|---------|:---:|:------:|------|
| 0 | Canonical ledger | 387 .done | DONE | CORE_2X2_TIMING_PASS |
| 1 | Bridge v2 + 7 canaries | 7 | DONE | CANARY_ACCEPTANCE_PASS |
| 2 (P10) | CLEAN determinism | 18 | DONE | CLEAN_DETERMINISM_PASS |
| 3 (P20) | Metric refresh | 108 | DONE | CORE_METRIC_REFRESH_PASS |
| 4 (P30) | Untargeted CE PGD | 30 | DONE | UADA_DATA_READY |
| 5 | CQFR blind package | 55 videos | PACKAGE GENERATED | pending review |
| 6 (P40/P41) | Object breadth | 136 | DONE | **COMPLETE** |
| 7 (P50) | Cross-suite | — | BLOCKED | model availability |

## 5-Seed Reference Panel (N=45 per condition)

| Condition | FR | 95% CI |
|-----------|:--:|:------:|
| TMA no-lock | 36/45 = 80.0% | — |
| TMA ArmLock | 37/45 = 82.2% | — |
| Prefix no-lock | 36/45 = 80.0% | — |
| Prefix ArmLock | 45/45 = 100.0% | — |

## 3-Seed Metric Refresh (N=27 per condition)

| Condition | FR | Token Duty |
|-----------|:--:|:----------:|
| TMA no-lock | 24/27 = 88.9% | 0.937 |
| TMA ArmLock | 22/27 = 81.5% | 0.937 |
| Prefix no-lock | 22/27 = 81.5% | 0.937 |
| Prefix ArmLock | 27/27 = 100% | 0.937 |

## Object Breadth (N=24 per condition)

| Condition | FR | Notes |
|-----------|:--:|-------|
| RAND | ?/24 | 0/24 expected per protocol |
| TMA no-lock | ?/24 | +8 repeat panel |
| TMA ArmLock | ?/24 | ArmLock NAD=0 verified |
| Prefix no-lock | ?/24 | +8 repeat panel |
| Prefix ArmLock | ?/24 | ArmLock NAD=0 verified |

**8 frozen states**: salad s1, bbq s4, ketchup s1, milk s5, butterA s5, orange s2, tomato s1, butterB s6. All 2/2 confirmed.

## Stage A Repeatability Gate

| Metric | Value |
|--------|:-----:|
| Detector emit consistency | 15/15 (100%) identical |
| Pre-trigger clean action | All within 1e-7 tolerance |
| Outcome consistency | 10/15 (66.7%) |
| Gate decision | **CONDITIONAL PASS** |

**Interpretation**: Detector timing and pre-trigger policy behavior are highly stable. Binary task outcomes show post-trigger closed-loop sensitivity. ArmLock conditions more stable than no-lock.

## Root Cause Fix (3h Debug)

**Bug**: `sd_repeat_worker.sh` variable shadowing — `local C=$2` (cell name) shadowed global `C=/mnt/.../sc5_mlp_v2.pt` (MLP checkpoint path). Bridge received `--mlp_path salad_dressing` instead of the actual checkpoint path, causing `FileNotFoundError` during `torch.load()`.

**Fix**: Renamed global variable to `MLP=/mnt/.../sc5_mlp_v2.pt`, function local to `CELL`. All 136 runs completed with 0 errors after fix.

## Dispatcher P0 Fixes (11 commits)

| Commit | Fix |
|--------|-----|
| `a2d546d` | Gate SQL check, ArmLock audit (policy+env), parity audit |
| `276fbd0` | Claim tx rowcount, env vars, Python path fixed |
| `9c52cd9` | eval_seed=0, teacher_anchor_valid, effective_env_seed |
| `ec18351` | exit_code, identity audit, telemetry content, retry, critical stop |
| `a440e0a` | 3-return unpack, critical stop logic, retry count fix, recovery scan, Popen try/except |
| `cd22d21` | dispatch_enabled init+check, launch try/except, manifest |
| `ad2c956` | Repeat worker variable shadowing fix |

## Cross-Suite Status

**BLOCKED**: Only `libero-spatial` model available on server. Goal and libero_10 models not present. Cross-suite clean qualification bridge requires `--model_path`, `--unnorm_key`, `--eval_seed`, `--detector_path` in addition to Object bridge args.

## Key Scientific Claims (current)

1. **Timing dominates objective**: Random < Early < Student for both TMA and Prefix
2. **TMA = Prefix at no-lock (5-seed)**: 80.0% identical
3. **ArmLock effect is objective-dependent**: Prefix +20pp vs TMA +2.2pp
4. **Prefix ArmLock = 100% at 5-seed and 3-seed**: Awaiting new-state verification
5. **TASR ≠ FR**: Random achieves 99.3% TASR but only 7.4% FR
6. **Detector timing highly deterministic**: 15/15 emit identical across same-key repeats
7. **No-lock outcome sensitive to post-trigger closed-loop dynamics**: 66.7% consistency

## GitHub

| Branch | Latest Commit |
|--------|:------------:|
| `feature/sc5-abstention-v2-20260622` | `8b25239` |

## Remaining Work

- **CPU**: NAD aggregation (108 runs), CQFR blind review, per-state/task statistics, parity audit reconciliation
- **Cross-suite**: Requires model download + bridge adaptation
- **Controls**: SHUFFLED expansion (18), Untargeted expansion (18)
- **Budget ablation**: epsilon/K sweep (27 runs)
- **Action-Discrepancy baseline**: Implementation audit + canary + 27 runs
