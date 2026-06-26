# Phase 7 Object Table 1 — Full Experimental Progress Report

**Generated: 2026-06-26**

## Executive Summary

138 scientific runs completed across 4 phases. 108 metric refresh runs produce full NAD/latency/response data. 30 Adapted Action-Discrepancy PGD runs establish UADA-equivalent baseline. CQFR blind review package (55 videos) generated.

## Phase Status

| Phase | Content | Runs | Status | Gate |
|-------|---------|:---:|:------:|------|
| 0 | Canonical ledger freeze | — | DONE | PHASE_AB_LEDGER_PASS |
| 1 | Bridge v2 + 7 canaries | 7 | DONE | CANARY_ACCEPTANCE_PASS |
| 2 (P10) | CLEAN determinism | 18 | DONE | CLEAN_DETERMINISM_PASS |
| 3 (P20) | Metric refresh | 108 | DONE | CORE_METRIC_REFRESH_PASS |
| 4 (P30/P31) | Adapted Action-Disc. PGD | 30 | DONE | UADA_DATA_READY |
| 5 | CQFR blind package | — | DONE | 55 videos |
| **6 (P40/P41)** | **Object breadth** | **120** | **QUALIFYING** | pending |
| 7 (P50/P51) | Cross-suite smoke | 90 | pending | — |
| 8 (P60/P70) | Control + Ablation | 36 | pending | — |

## Phase 0: Canonical Ledger

- **387 .done files**, 0 duplicate keys, 0 SHA violations
- **SHUFFLED 33 runs** not found in directory tree (discrepancy documented)
- **TMA Early FR**: 12/27 = 0.444 (matches Prefix Early)
- **5-seed 2x2**:
  - TMA no-lock: 36/45 = 80.0%
  - TMA ArmLock: 37/45 = 82.2%
  - Prefix no-lock: 36/45 = 80.0%
  - Prefix ArmLock: 45/45 = 100.0%
- **Gate status**: `CORE_2X2_TIMING_PASS` — core 2x2 and timing complete.
  - `FULL_TABLE1_LEDGER_PASS` on HOLD pending SHUFFLED artifact recovery.
  - Prior `PHASE_AB_LEDGER_PASS.json` records `match: false` and must be split.

## Phase 1: Telemetry V2 + Dispatcher

- Telemetry v2 bridge: 83 CSV fields including clean/adv policy actions, token IDs, timing breakdown (clean_forward_ms, pgd_optimization_ms, adv_decode_ms, arm_lock_ms), NAD-compatible action traces
- COMPLETE.json atomic protocol (fixed `os.fsync(f.fileno())` bug)
- 7 canaries all PASS:
  - ArmLock NAD_exec_arm = 0 confirmed
  - Early-Shift/Random-Time override audit PASS
  - No-emit cell (cream_cheese) attack_frames=0 confirmed
- Dispatcher v2 created (SQLite WAL + BEGIN IMMEDIATE), not yet deployed
- Gate: `PHASE1_CANARY_ACCEPTANCE_PASS.json`

## Phase 2 (P10): CLEAN Determinism

- 9 canonical CLEAN: all 9 cells SUCCESS (clean baseline confirmed)
- 3 cells × 4 repeats (1 canonical + 3 determinism):
  - salad_dressing_s0: all 4 identical (hash 731c4149)
  - butter_s0: all 4 identical (hash 1967c423)
  - tomato_sauce_s0: all 4 identical (hash 0e6aeb19)
- **3 representative cells have deterministic repeat trajectories** (salad, butter_s0, tomato_s0).
  - Remaining 6 cells have not undergone repeat auditing.
  - Scope limited to: "On the three repeat-audited cells, fixed-protocol rollouts are identical."
- Gate: `CLEAN_DETERMINISM_PASS.json`

## Phase 3 (P20): Core Metric Telemetry Refresh

108 runs = 4 conditions × 9 cells × 3 seeds (42, 123, 456)

**3-seed FR (metric panel, N=27 per condition):**

| Condition | FR | Token Duty |
|-----------|:--:|:----------:|
| TMA no-lock | 24/27 = 88.9% | 0.937 |
| TMA ArmLock | 22/27 = 81.5% | 0.937 |
| Prefix no-lock | 22/27 = 81.5% | 0.937 |
| Prefix ArmLock | 27/27 = 100% | 0.937 |

**All runs include:**
- Full telemetry v2 (83 fields)
- Video (H.264, fps=10, stride=2)
- COMPLETE.json atomic protocol
- Timing breakdown per step
- ADV policy action before/after ArmLock

**Note**: These are 3-seed metric panel numbers (N=27), NOT the canonical 5-seed FR (N=45). The 5-seed canonical FR for Prefix ArmLock is also 100% (45/45).

Gate: `CORE_METRIC_REFRESH_PASS.json`

## Phase 4 (P30/P31): Adapted Untargeted Clean-Token CE PGD

30 runs = 3 canaries + 27 full (9 cells × 3 seeds)

**Implementation Audit**: No faithful UADA-DoF7 objective exists in codebase. No farthest-bound target selection. No soft action-discrepancy loss. The actual objective used is `untargeted_clean_token_ce` — a token-level untargeted CE attack. Correctly named **Adapted Untargeted Clean-Token CE PGD**.

**Status**: Rollouts complete, analysis pending. 3 method canaries + 27 full runs. Cannot be called UADA-equivalent, Action-Discrepancy, or Adapted UADA.

All 30 runs complete with telemetry v2 + video. FR pending per-run audit.

## Phase 5: CQFR Blind Review Package — GENERATED, NOT BLIND

- **55 videos selected** from 108 metric refresh pool
- **ISSUE**: Current package CSV contains `condition`, `objective`, `arm_lock`, `task_success` columns.
  Video paths directly expose method names. NOT truly blind.
- **Fix required**: Regenerate with opaque `blind_videos/BXXXX.mp4` paths and reviewer CSV
  containing only `blind_id`, `opaque_video_path`, `review_label`, `confidence`, `notes`.
- **Status**: Package generated, needs blind regeneration. Review not yet conducted.

## Phase 6 (P40/P41): Object New-State Breadth — IN PROGRESS

**Clean qualification status:**

| Slot | Task | State | Result |
|------|------|:-----:|:------:|
| salad | 2 | s1 | True ✅ FROZEN |
| bbq | 3 | s1 | False → testing s2 |
| ketchup | 4 | s1 | testing |
| milk | 7 | s5 | testing |
| butterA | 6 | s1 | False → need higher |
| orange | 9 | s1 | testing |
| butterB | 6 | TBD | after butterA |
| tomato | 5 | s1 | testing |

After qualification: 8 states × 3 seeds × 5 conditions = 120 attack runs.

## Scientific Claims (current)

1. **Timing dominates objective**: Random < Early < Student for both TMA and Prefix
2. **TMA = Prefix at no-lock**: 80.0% identical at 5-seed
3. **ArmLock effect is objective-dependent**: Prefix +20pp vs TMA +2.2pp
4. **Prefix ArmLock = 100% at 5-seed and 3-seed panels**
5. **TASR != FR**: Random achieves 99.3% TASR but only 7.4% FR
6. **Environment deterministic**: All repeat runs identical

## Claim Boundary

Claims limited to current 9-cell Object panel with fixed perturbation seeds (attack/PGD seeds, not independent env seeds). Cross-suite generalization and state-independent claims require Phase 6-7 completion.

## Next Steps

1. Complete clean qualification → freeze state manifest
2. Launch 120 Object breadth attack runs
3. Cross-suite smoke (90 runs)
4. Control expansion (36 runs)
5. Budget ablation
