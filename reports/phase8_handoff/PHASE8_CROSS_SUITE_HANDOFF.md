# Phase 8 Cross-Suite Generalization — Handoff

**Generated**: 2026-06-27 16:00 CST
**Status**: PAUSE DISPATCH — MODEL TRANSFER CONTINUES
**Handoff commit**: PENDING

---

## 1. Executive Status

```
DISPATCHERS      = STOPPED (4 PIDs killed)
ACTIVE BRIDGES   = 6 python processes (allow to complete)
QUEUED JOBS      = 0 (dispatchers dead)
COMPLETED (.done)= 0
ATTEMPTED DIRS   = 67
VALID OUTPUTS    = 0
BRIDGE OK        = NO — object-site lookup is Object-specific
LIBERO-10        = 800MB / 15GB transferred, no active SCP
MANIFEST SHA     = VERIFIED (fce2bc3d...)
```

## 2. Immutable Git Identifiers

```
branch:     experiments/cross-suite-generalization-v1
HEAD:       5313ef24cce5b0923ff940b75bc8c03d77617bcf
remote:     origin
status:     clean working tree (no uncommitted changes)
prev tags:  table1-object-v3.5-b1bbf54
```

## 3. Object-Suite Frozen Boundary

Object suite frozen at `b1bbf54`. Do not modify Object results.
Phase 8 output root: `/mnt/sdc/dty_user/openvla_attack/evidence/phase8_cross_suite_v1/`

## 4. Models and Exact Paths

| Suite | Path | Status |
|-------|------|:------:|
| Spatial | `models/libero-spatial/spatial_c8f03f4_20260620/` | Ready |
| Goal | `models/libero-goal` → `table1_dependencies/openvla-7b-finetuned-libero-goal/` | Ready (symlink) |
| LIBERO-10 | `models/libero-10/openvla-7b-finetuned-libero-10/` | 800MB/15GB |

## 5. Manifest

```
Path: evidence/phase8_cross_suite_v1/manifests/ALL_630_JOBS.jsonl
SHA256: fce2bc3da942e6d0daab3ab04685b582c5030cb5e4ae45eb9a4aab5da8dd3ebb
Lines: 630
Breakdown:
  P1 smoke:   21 (3 suites × 1 task × 1 seed × 7 conditions)
  P2 CLEAN:   90 (3 suites × 10 tasks × 3 seeds)
  P3 attack: 360
  P4 armlock: 180
```

Per-suite manifests also available:
- `LIBERO_SPATIAL_210.jsonl` (210 jobs)
- `LIBERO_GOAL_210.jsonl` (210 jobs)
- `LIBERO_10_210.jsonl` (210 jobs)

## 6. GPU/Process State (at freeze)

```
GPU0: 56GB (other users)
GPU1: 24GB, 31% util — SPATIAL launcher idx=8, bridge PID 3950830 (task0 s1 seed123 RANDOM)
GPU2: 39GB, 10% util — GOAL launcher idx=8, bridge PID 3951994 (task0 s1 seed123 RANDOM)
GPU3: 24GB, 9%  util — SPATIAL launcher idx=6, bridge PID 3951462 (task0 s0 seed42 PREFIX_ARMLOCK)
GPU4: 47GB, 86% util (other users)
GPU5: 42GB, 90% util (other users)
GPU6: 24GB, 29% util — GOAL launcher idx=6, bridge PID 3951912 (task0 s0 seed42 PREFIX_ARMLOCK)
GPU7: 39GB (other users)

Active bridge PIDs: 3950830, 3951121(?), 3951462, 3951912, 3951994, 3952752
All on task 0 (Spatial or Goal), various states and conditions.
```

## 7. Attempted Jobs — ALL FAILED

```
Total dirs: 67
.done: 0
episode_summary.json: 0
step_telemetry.csv: 0
rollout_raw.mp4: 0

Breakdown by naming:
  p8_SPATIAL_*:  9 dirs (V3 dispatcher, task0 s0/s1)
  p8_GOAL_*:     9 dirs (V3 dispatcher, task0 s0/s1)
  p8_libero_goal_*: 21 dirs (V1/V2 worker, task0 all 3 seeds)
  p8_libero_10_*:   28 dirs (V1/V2 worker, LIBERO-10 model unavailable)
```

**ALL 67 jobs = FAILED_TECHNICAL.** Root cause: bridge object-site lookup is Object-specific.

## 8. Dispatcher Incident Root Cause

The bridge at `scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_v2.py` line 171-172:

```python
_obj_key = _task_name.replace("pick_up_the_","").replace("_and_place_it_in_the_basket","")
obj_sid = env.sim.model.site_name2id(f"{_obj_key}_1_default_site")
```

This hardcodes Object task name patterns:
- Object: "pick_up_the_butter_and_place_it_in_the_basket" → _obj_key = "butter" ✓
- Spatial: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate" → _obj_key extraction fails

Additionally, MuJoCo site names differ across suites:
- Object: `butter_1_default_site`
- Spatial: `akita_black_bowl_1_default_site` (different prefix)

**The `--unnorm_key` and `--suite_name` patches were necessary but not sufficient.** Object-site lookup must be made suite-aware.

## 9. Required Code Patch (P0)

Bridge line 171-172 must handle suite-specific task name templates:
1. Detect suite from `--suite_name`
2. Use suite-specific suffix replacement (e.g., `_and_place_it_in_the_basket` for Object, `_and_place_it_on_the_plate` for Spatial, etc.)
3. Use MuJoCo site name prefix mapping per suite

OR: Use an external lookup table mapping (suite, task_idx) → object_key.

## 10. Queue Redesign Required

Current `5313ef2` dispatcher defects:
1. Counter NOT atomic (cat+echo, not mkdir lock)
2. Per-GPU counters produce duplicate work across GPUs
3. `rm -rf $OUT` before launch destroys any partial outputs
4. `set -e` may exit before EC recording
5. `|| true` silently swallows failures
6. No heartbeat, no stale lock recovery
7. No phase gating (P1→P2→P3→P4)

Required V2 design:
- 630-job immutable manifest
- Global atomic mkdir claim (not GPU-scoped)
- One job = one owner max
- Independent worker_id, HOME, TMPDIR
- No rm -rf without lock
- Technical failure → failure ledger
- Phase gating with explicit gate checks
- Stale lock heartbeat + recovery rules

## 11. P1 Restart Protocol

After bridge patch:
1. Clean ALL 67 existing run dirs → quarantine
2. Reset all counters
3. Launch 1 worker on GPU3 for Spatial P1 (7 jobs: task0, seed42)
4. Verify CLEAN completes with episode_summary.json + step_telemetry.csv
5. Verify RANDOM + TMA_ARMLOCK
6. Verify remaining 4 conditions
7. Spatial P1 Gate: 7/7 technical completion
8. Repeat for Goal P1 on GPU2
9. Repeat for LIBERO-10 P1 on GPU3 after model verification

## 12. 21/21 P1 Gate

All 3 suites × 7 conditions must pass:
- 21/21 technical completion
- 0 duplicate claims
- 0 JSON/CSV parse failures
- 0 task/suite mismatch
- ArmLock executed-arm violations = 0

## 13. HOLD / GO Matrix

```
LIBERO_10_TRANSFER              = CONTINUE
CURRENT_DISPATCHERS             = PAUSED
CURRENT_5313EF2_OUTPUTS         = ALL QUARANTINE (67 FAILED)
QUEUE_REDESIGN                  = REQUIRED
BRIDGE_OBJECT_SITE_PATCH        = REQUIRED (P0)
SPATIAL_P1_RESTART              = GO AFTER BRIDGE PATCH
GOAL_P1_RESTART                 = GO AFTER BRIDGE PATCH
LONG_P1                         = GO AFTER MODEL VERIFY + BRIDGE PATCH
EIGHT_WORKERS                   = HOLD UNTIL 21/21 P1 PASS
FULL_P2_P3_P4                   = HOLD
CQFR_WORK                       = DEFERRED
TABLE1_PUBLICATION_FREEZE       = HOLD
```

## 14. Exact Next Commands

```bash
# 1. Verify current state
git rev-parse HEAD  # should be 5313ef2 or newer (this handoff commit)

# 2. Check LIBERO-10 transfer
ls -la /mnt/sdc/dty_user/openvla_attack/models/libero-10/
du -sh /mnt/sdc/dty_user/openvla_attack/models/libero-10/

# 3. Resume LIBERO-10 transfer if incomplete
# scp -r D:/vla_attack/_archive/local_model_cache/hf_stage/openvla-7b-finetuned-libero-10/ \
#     dty_user@10.60.2.56:/mnt/sdc/dty_user/openvla_attack/models/libero-10/

# 4. Patch bridge object-site lookup (suite-aware) — CRITICAL

# 5. Quarantine existing 67 failed dirs
# mv evidence/phase8_cross_suite_v1/runs evidence/phase8_cross_suite_v1/quarantine_5313ef2

# 6. Launch P1 after bridge patch
# bash phase8_simple_launch.sh 3 SPATIAL 0   # CLEAN
# Verify .done + episode_summary.json exists
```

## 15. Required Output Files (after completion)

```
GENERALIZATION_RUN_LEVEL.csv
GENERALIZATION_CONDITION_SUMMARY.csv
GENERALIZATION_ARMLOCK_AUDIT.csv
GENERALIZATION_FAILURE_LEDGER.csv
TABLE1_CROSS_SUITE_V1.md
SHA256SUMS.txt
```
