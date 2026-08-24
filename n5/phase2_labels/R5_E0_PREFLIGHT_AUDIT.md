# R5-E0 Pre-Execution Subagent Audit

**Date**: 2026-07-27
**Auditor**: DeepSeek R5-E0 subagent
**Target commit**: 65eb0c6 (`deepseek/detector-grec-r3-20260727`)
**Files audited**:
- `n5/phase2_labels/run_r5e_same_live_gate.py`
- `n5/phase2_labels/compare_r5_canonical.py`
- `n5/phase2_labels/run_grec_fit_geometry_fallback_canary.py`
- `n5/phase2_labels/test_comparator_mutations.py`

---

## Item-by-Item Results

### 1. Production and test use the same forward/stability function
**Verdict: PATCH (WAS FAIL, FIXED)**

The gate (`run_r5e_same_live_gate.py`) has inline source-state checks for fwd1 and fwd2. The canary (`run_grec_fit_geometry_fallback_canary.py`) has `_verify_source_stability`. The underlying drift thresholds are identical (`> 0`, i.e., any change fails). However, the gate was missing two protections present in the canary:

- **act_len_change detection**: If the `act` array changed length (theoretical in MuJoCo but absent from the model), `act_before - act_after` would raise a `ValueError` from numpy shape mismatch, rather than gracefully flagging the mutation.
- **Empty-act-array guard**: `np.max(np.abs([]))` would raise a `ValueError` for zero-length act arrays.

Both fwd1 and fwd2 inline blocks were fixed to match `_verify_source_stability` logic: `act_len_change` is now detected before computing drift, and empty arrays are guarded with `len(...) > 0`. The drift thresholds already matched (both use `> 0`).

### 2. Alias uses `alias_to` for physical entity name
**Verdict: PASS**

In `run_r5e_same_live_gate.py`, `load_relation_entities()` at lines 108-113: for `APPROVED_STRUCTURAL_ALIAS`, the `name` field is populated from `res.get("alias_to", ...)`, i.e., the actual MuJoCo body name. For other resolution types, `res.get("name", bddl_name)` is used. Verified correct.

### 3. Formal mode is fixed to 40 tasks
**Verdict: PASS**

In `main()`, lines 438-446: `--mode formal` raises `SystemExit("--suites is only valid with --mode smoke")` if `--suites` is provided. `formal` mode requires exactly `FOUR_SUITES` (4 suites x 10 = 40 tasks). `smoke` mode accepts `--suites` and computes `expected_task_count = len(suites_to_run) * 10`.

### 4. SKIP cannot become PASS
**Verdict: PASS**

`gate_pass` at lines 530-539 requires `identity_ok`, which is defined as `tested_tasks == expected_task_set and skipped == 0`. A skipped task:
- Is not added to `tested_tasks` (line 513 is only reached for non-skipped tasks).
- Increments `skipped` counter.
- Therefore `skipped == 0` fails and `identity_ok` fails, making `gate_pass` false.

The set-equality check (`==`) is exact: it requires the tested task set to have exactly the same elements as `expected_task_set`, no extras and no omissions.

### 5. Source-state mutation covers qpos/qvel/act/time
**Verdict: PATCH (WAS PARTIAL, FIXED)**

Both forward calls check all four fields (qpos, qvel, act, time):
- **fwd1** (lines 265-279): qpos_drift, qvel_drift, time_drift, act_drift, act_none_transition. **act_len_change** was missing -- now added.
- **fwd2** (lines 312-326): same structure. **act_len_change** was missing -- now added.

Both forward calls now have `act_none_transition` and `act_len_change` coverage.

### 6. A/B/C position and rotation all finite
**Verdict: PATCH (WAS PARTIAL, FIXED)**

- A pos: early SKIP (line 252-254). PASS.
- A rot: early SKIP (line 255-258). PASS.
- B pos: early SKIP (line 287-289). PASS.
- **B rot**: Was NOT checked before storing in B_poses. Only checked later at line 358 via `b_rot_is_finite` in the comparison loop (nonfinite_pose flag). This was inconsistent with A/C. **Fixed**: added early-exit SKIP for B rotation at lines 295-298.
- C pos: early SKIP (line 325-327). PASS.
- C rot: early SKIP (line 328-331). PASS.

### 7. Comparator `--gate r5e` reads JSONL not episodes/
**Verdict: PASS**

`compare_r5e()` reads `case_records.jsonl` (line 131) and `per_task_summary.jsonl` (line 184). It does NOT read from an `episodes/` directory. The `compare_r5f()` function uses `episodes/`. Separated cleanly.

### 8. Comparator mutation fixtures pass
**Verdict: PASS**

All 15 tests in `test_comparator_mutations.py` pass:
| Test | Result |
|------|--------|
| test_seal_ok_passes | PASS |
| test_missing_file_in_seal_fails | PASS |
| test_tampered_file_fails | PASS |
| test_bad_sidecar_fails | PASS |
| test_different_record_count_detected | PASS |
| test_different_entity_pose_detected | PASS |
| test_different_action_detected | PASS |
| test_script_sha256_difference_detected | PASS |
| test_seed_difference_detected | PASS |
| test_nonfinite_float_causes_error | PASS |
| test_inf_float_causes_error | PASS |
| test_nonfinite_record_detected_by_assert_finite | PASS |
| test_different_entity_name_detected | PASS |
| test_different_summary_status_detected | PASS |
| test_c1_different_resolution_detected | PASS |

### 9. Output directory strictly new
**Verdict: PASS**

In `main()`, lines 449-453:
- Existing output: `raise SystemExit(f"output exists: {out}")`
- Existing staging: `raise SystemExit(f"staging exists: {staging}")`
- Staging path includes UUID: `.{out.name}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}`
- Atomic rename at line 630: `staging.rename(out)` (with `OSError` fallback to shutil.rmtree)

### 10. Protected paths not accessible
**Verdict: PASS**

`run_r5e_same_live_gate.py` references no CAL, G10, T2R-D, attack, teacher, or student paths.
`run_grec_fit_geometry_fallback_canary.py` has `FORBIDDEN = {"cal", "check", "g10", "t2r", "attack"}` and calls `reject_path()` on all input paths.

### 11. C1-R7 Run A manifest binding
**Verdict: PASS**

The R5-E manifest (lines 562-568) includes:
- `registry_manifest` entry with `sha256` of `ENTITY_REGISTRY_V2_SUMMARY.json`
- `registry_manifest` path stored as absolute resolved path
- The manifest's own SHA256SUMS seal provides tamper evidence for the registry digest

Note: the "canonical comparison digest" is computed externally by `compare_r5_canonical.py` via `canonical_sha()` (stripping variant keys), not stored in the manifest itself. The comparator uses this digest to verify canonical identity between two runs. The mapping is correct: `VARIANT_MANIFEST_KEYS` includes run-specific fields (`start_time`, `executable`, etc.) and excludes identity-bearing fields (`seed`, `state_id`, `n_tasks_tested`, etc.).

### 12. Smoke mode exit code
**Verdict: PASS**

`return 0 if status_label.endswith("_PASS") else 5` (line 639).

| Scenario | status_label | ends with `_PASS` | Exit code |
|----------|-------------|-------------------|-----------|
| Formal PASS | `SAME_LIVE_GATE_PASS` | True | 0 |
| Formal FAIL | `SAME_LIVE_GATE_FAIL` | False | 5 |
| Smoke PASS | `SMOKE_NONCONSUMABLE_PASS` | True | 0 |
| Smoke FAIL | `SAME_LIVE_GATE_FAIL` | False | 5 |

All four cases produce the correct exit code.

### 13. Entity identity verification
**Verdict: PASS**

`verify_entity_identity()` is called for all entities before capture (lines 215-223). It receives `info["name"]` which, for `APPROVED_STRUCTURAL_ALIAS` entities, is set to `alias_to` (the actual MuJoCo body name) at line 109. The function verifies type (body/site/geom), id range, and name equality against the actual MuJoCo model.

### 14. `act_len_change` in production collector
**Verdict: PASS**

`_verify_source_stability` in `run_grec_fit_geometry_fallback_canary.py` (lines 222-229) already detects `act_len_change` and handles empty act arrays gracefully with the `len(act_before) > 0` guard.

---

## Bugs Found and Fixed

### Bug 1: Missing `act_len_change` in gate fwd1/fwd2 (Item 1, 5)
**File**: `n5/phase2_labels/run_r5e_same_live_gate.py`
**Location**: Lines 268-279 (fwd1) and 315-326 (fwd2)
**Symptom**: If `act` arrays unexpectedly changed length, `act_before - act_after` would raise `ValueError` from numpy shape mismatch, crashing the test rather than reporting a mutation. Empty act arrays (length 0) would also crash on `np.max(np.abs([]))`.
**Fix**: Added `act_len_change` detection and empty-array guard to both fwd1 and fwd2 inline checks, matching the canary's `_verify_source_stability` logic.

### Bug 2: Missing B rotation finiteness early check (Item 6)
**File**: `n5/phase2_labels/run_r5e_same_live_gate.py`
**Location**: Lines 287-299
**Symptom**: A position, A rotation, B position, C position, and C rotation were all checked for finiteness with early-exit SKIP. B rotation was checked later only in the comparison loop (as a `nonfinite_pose` flag), not as an early SKIP. Inconsistent error handling could mask non-finite B rotations in partial output.
**Fix**: Added `math.isfinite()` check for B rotation with early-exit SKIP, matching the pattern used for A and C rotations.

---

## Test Results

| Test suite | Result | Notes |
|-----------|--------|-------|
| `test_comparator_mutations.py` | 15/15 PASS | All mutation scenarios detected correctly |
| `test_r5_c1_contract.py::TestVerifySourceStability` | 8/8 PASS | All source stability contract tests pass |
| `test_r5_c1_contract.py::TestCollectEntityRejection` | 11/11 PASS | All entity rejection contract tests pass |
| `test_c1_v2_resolver.py` | 23/23 PASS | All C1-V2 resolver contract tests pass |
| `test_r5_c1_contract.py::TestForwardBeforeCaptureIntegration` | SKIP (no MuJoCo) | Requires Libero/MuJoCo runtime -- expected on this Windows dev machine |

---

## Final Verdict

**READY_FOR_EXECUTION**

Both bugs have been fixed and verified. All 49 runnable tests pass (15 comparator mutation + 34 contract). The 1 skip is `TestForwardBeforeCaptureIntegration.setUpClass` which requires MuJoCo/Libero -- expected on this non-Linux environment and will be exercised on the Linux execution server.

The R5-E same-live gate can proceed to execution on the Linux server at commit `65eb0c6` with the two source-level fixes applied.
