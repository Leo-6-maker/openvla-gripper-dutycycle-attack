# PILOT V2 Pre-launch Blocking Audit

**Generated**: 2026-05-29 20:15 CST
**Server**: 10.60.133.4
**Branch**: eval/official-libero-clean-20260525

---

## A. Git / Provenance

| Check | Result |
|-------|--------|
| Server HEAD | `c62214fabe3d9991029a3a450a9a1f0f4de75f14` |
| Expected frozen | `087044390498f271fddfe009d95e6701fc9450fd` |
| Frozen exists locally? | **NO** |
| Working tree clean? | Yes (only untracked) |
| Core-file diff possible? | No — frozen commit not available |
| Remote accessible? | No — SSL cert mismatch `rukita.co` != `github.com` |

**Verdict**: ⛔ **P0 BLOCKER** — cannot verify `c62214f` == `0870443`. Core-file diff impossible.

Server commit log:
```
c62214f Fix git branch command compatibility in launch script
c078626 Fix LIBERO success predicate — use done=True not info[success]
ab6cd5c Hotfix 3: fix run_dir collision + manifest provenance
2fc98b5 Hotfix 2: fix P0/P1 bugs + add smoke flag
7032d61 Hotfix P0/P1 bugs in detector-triggered attack runner
```

---

## B. Static Code Invariants

### B1 — Success Predicate
**PASS** ✅
- `success = True` when `done=True` (line 533)
- `info["success"]` only logged as `ep_info_success`, not primary
- `success_official = done_any_LIBERO_official`
- Manifest records: `success_official`, `success_done`, `done_any`, `success_info_present`, `success_info_value`

### B2 — Detector Reset
**PASS** ✅
- Line 279: `detector = None; attack_rng = None` (init only)
- Line 357: `detector.reset()` (per-episode)
- No double `detector = None` after load

### B3 — Attack Conditions
**PASS** ✅
- choices: `clean, oracle_open, random_control, gripper_inversion_proxy`
- Line 127: backward compat for `VIS_targeted` → `gripper_inversion_proxy`
- No standalone `VIS_targeted` string in active code path

### B4 — run_dir / run_id
**PASS** ✅
- `run_id = f"{args.run_id_prefix}_{task_name_short}_s{ep_idx}"`
- Includes condition prefix in run_id_prefix
- `exist_ok=True` allows re-run overwrite (minor: known behavior)

### B5 — Launch Script
**PASS** ✅
- Conditions: `clean oracle_open random_control gripper_inversion_proxy`
- Records `RUNNER_COMMIT` from `git rev-parse HEAD`
- No `--force_detector_trigger` for formal pilot
- No `VIS_targeted`

### B6 — Aggregate Script
**PASS** ✅
- Uses `gripper_inversion_proxy`, not `VIS_targeted`

### B7 — CQ Script
**PASS** ✅
- Uses `gripper_inversion_proxy`, not `VIS_targeted`

### B8 — Clean Attack Guard
**PASS** ✅
- Line 450: `attack_condition != "clean"` guard on attack trigger
- Line 452: `attack_condition != "clean"` guard on attack application
- Clean CANNOT set `attack_applied=True`

---

## C. Tests

| Test | Result |
|------|--------|
| py_compile runner | ✅ OK |
| py_compile triggers | ✅ OK |
| unittest test_success_predicate_regression (6/6) | ✅ OK |
| bash -n launch script | ✅ OK |
| bash -n aggregate script | ✅ OK |
| bash -n evaluate script | ✅ OK |

Pytest environment broken (missing `pygments`). Workaround: py_compile + unittest + bash -n covers critical paths.

---

## D. Existing Detector-Clean Output Integrity

### Valid (7 states)
| Task | State | success_official | attack_applied | orig==attacked | No dupes |
|------|-------|-----------------|----------------|----------------|----------|
| cream_cheese | 0 | True | None | 128/128 | ✅ |
| cream_cheese | 2 | True | None | 144/144 | ✅ |
| ketchup | 0 | True | None | 150/150 | ✅ |
| salad_dressing | 0 | True | None | 122/122 | ✅ |
| salad_dressing | 1 | True | None | 128/128 | ✅ |
| salad_dressing | 2 | True | None | 137/137 | ✅ |

### Failed Clean (1 state)
| Task | State | success_official | Notes |
|------|-------|-----------------|-------|
| cream_cheese | 1 | False | Failed clean — excluded from pilot |

### Partial (1 state — Quarantined)
| Task | State | Notes |
|------|-------|-------|
| ketchup | 1 | 123 SR lines, no manifest. Killed mid-run during duplicate loop. |

### Salad Dressing Contamination Check
**PASS** ✅
- All salad_dressing mtimes (19:58-20:03) predate duplicate re-run (started 20:03)
- Pid 41719 did NOT write to salad_dressing directories
- Zero duplicate episode keys detected
- All 3 states valid

---

## Stage 2 — Blocker Classification

### P0 Blockers

| # | Blocker | Detail |
|---|---------|--------|
| 1 | **Git SHA mismatch** | Server `c62214f` ≠ frozen `0870443`. Frozen commit not available locally. Cannot verify core-file equivalence. SSL blocks remote fetch. |

### P1 Issues (non-blocking, documented)

| # | Issue | Detail |
|---|-------|--------|
| 1 | Step-level reward/done shift | reward/done logged before `env.step()` — one step behind action that caused them. Episode-level `done_any` correct. |
| 2 | pytest env broken | Missing `pygments`. Workaround: py_compile + unittest + bash -n. |
| 3 | `exist_ok=True` on run_dir | Allows re-run overwrite. Mitigated by unique run_id_prefix per condition. |

---

## Stage 3 — Detector-Clean Completion Plan (BLOCKED)

Cannot proceed until P0 Git blocker is resolved.

If resolved, the safe plan:
1. Use new output root: `/data/liuyu/outputs/milestone_2f_object_pilot_v2_detector_clean_missing_rerun_20260529`
2. Missing tasks: ketchup (s1,s2 re-run to match s0), tomato_sauce (3 states), milk (3 states)
3. Parallel: GPU4,5 → ketchup, GPU2,6 → tomato_sauce, then milk on first free

### Current Eligible States for Pilot v2 (if Git resolved)

Even without completing all 15 detector-clean states, existing valid states already meet the gate:

| Task | Eligible States | Count |
|------|----------------|-------|
| cream_cheese | s0, s2 | 2 |
| ketchup | s0 | 1 |
| salad_dressing | s0, s1, s2 | 3 |

Total: 6 eligible states across 3 tasks. Gate threshold (≥2 tasks × ≥3 states OR ≥6 states across ≥2 tasks) is **met** with cream_cheese (2) + salad_dressing (3) = 5 states across 2 tasks, or all 6 states across 3 tasks.

However, best practice is to complete all 15 detector-clean states first.

---

## Final Verdict

**⛔ PILOT V2 LAUNCH IS BLOCKED**

Reason: Cannot verify server HEAD `c62214f` is content-equivalent to frozen commit `0870443`. The frozen commit does not exist in the local repository, and GitHub remote is unreachable due to SSL certificate mismatch.

All other checks (success predicate, attack guards, condition names, tests, output integrity) pass clean.

**Next step**: Resolve Git SHA discrepancy before any pilot v2 launch.
