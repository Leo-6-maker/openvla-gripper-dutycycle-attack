# L12 E4C.2 — RC1a Remap + Teacher-P Eligibility Construction

**Stage**: E4C.2
**Date**: 2026-06-15
**Branch**: `exp/l12-critical-close-window-selector-20260615`
**Prereg Commit**: (to be assigned by git)

---

## Scope

Convert the 402 frozen E4C.1 schema-passing trace candidates into an auditable
clean-label eligibility pool. No training, no GPU, no new data capture.

## Authorized Input

```
Input inventory: tables/e4c_audit/l12_e4c_data_inventory_v2.csv
Input SHA256: (E4C.1 commit 9232c4a frozen)
Schema-passing traces: 402 (corrected criteria)
Schema-failing traces: 3
Input manifest: tables/e4c_audit/l12_e4c2_input_manifest.csv
Input manifest SHA256: B11AC1E8DF74F63FD764985FF335A7D6648114AF66045FC34ACA6956E6B85863
```

### Corrected schema-passing criteria (NOT the buggy `schema_pass` column)

```python
has_required_header_fields == True
row_count > 0
file_size > 0
len(sha256) == 64
```

The `schema_pass` column in E4C.1 inventory stores the SHA256 string (truthy)
for passing traces, not a boolean. E4C.2 constructs eligibility from raw columns.

## Remapper Version

```
REMAPPER_VERSION = "rc1a_corrected_v2_e1_5"
```

Located in: `scripts/stageb/remap_v4_trace_for_l12.py`

### Open-convention invariant (frozen)

```
raw_gripper > 0.5  → env = -1  → OPEN  (decoded_open_bool = 1)
raw_gripper < 0.5  → env = +1  → CLOSE (decoded_open_bool = 0)
raw_gripper == 0.5 → boundary, excluded
```

Check: `decoded_open_bool == 1` iff `clean_gripper_env < -0.5`

### Field validity check (row-level, all rows)

For each of these fields, check every row:
```
obj_x, obj_y, obj_z
eef_x, eef_y, eef_z
clean_gripper_env
decoded_open_bool
gripper_qpos_before
```

Record per trace: missing count, parse-fail count, non-finite count,
invalid-domain count, first invalid row index. No 50-row sampling.

### Gap state reset

When gripper is invalid/neutral (abs(env) <= 0.5), internal state
(prev_clean_close, close_streak) is reset to zero. close_onset after
an invalid gap is flagged as `close_onset_after_invalid_gap`.

## Teacher-P Algorithm (frozen)

File: `src/gripper_attack/phase_detector.py`

Only grasp privilege is available (NO placement/target coordinates).

### Thresholds (frozen)

| Parameter | Value | Unit |
|-----------|-------|------|
| EEF_TO_OBJ_NEAR_THRESHOLD | 0.08 | meters |
| OBJECT_LIFT_MIN_DELTA | 0.005 | meters |
| OBJECT_LIFT_LOOKAHEAD | 15 | steps |
| SUSTAINED_MOTION_FRAMES | 2 | consecutive frames |

### Teacher-P criteria (all must hold at candidate close `t`)

1. `close_onset == 1` AND `clean_close == 1`
2. `decoded_open_bool == 0` (gripper not already open)
3. Grasp privilege valid at `t` through `t + lookahead`:
   - eef_x, eef_y, eef_z, obj_x, obj_y, obj_z, eef_to_obj_distance
   - All non-empty, non-NaN for every step in [t, t+lookahead)
4. `eef_to_obj_distance < 0.08` at step `t`
5. Sustained vertical lift: cumulative `obj_z[t+i] - obj_z[t] >= 0.005m`
   for >= 2 consecutive frames with EEF within 0.08m of object

### Teacher-P abstain conditions

- Grasp privilege check fails (any required field missing/NaN)
- No close candidate satisfies all 5 criteria
- Returns `-1` (abstain)

### Teacher-R (rule baseline, also frozen)

First `close_onset AND clean_close` with `gripper_qpos_before < 0.01`.

## CLOSE Candidate Definition (frozen)

File: `src/gripper_attack/critical_close_selector.py`

A step `t` is a CLOSE-event candidate when ANY of:
- `raw_open_to_close_crossing`: raw[t-1] > 0.5 AND raw[t] <= 0.5 (validity-gated)
- `close_onset == 1`
- `close_streak == 1`

### Per-candidate features (frozen, all 16 from E4B.3)

| Feature | Type |
|---------|------|
| total_score | discrete_score |
| raw_crossing_bonus | candidate_definition |
| close_streak_bonus | candidate_definition |
| close_onset_qpos_bonus | qpos_conditioned_discrete |
| eef_deceleration_bonus | discrete_score |
| qpos_ready_bonus | qpos_conditioned_discrete |
| eef_speed_now | continuous_dynamic |
| eef_speed_prev | continuous_dynamic |
| eef_deceleration_delta | continuous_dynamic |
| close_streak | candidate_definition |
| raw_crossing | candidate_definition |
| close_onset | candidate_definition |
| qpos | continuous_dynamic |
| time_since_prev_close | temporal_context |
| time_since_last_open | temporal_context |
| candidate_index | temporal_context |

## Eligibility Classification (frozen)

Each trace gets exactly one category:

```
ELIGIBLE_MULTI_CANDIDATE   — Teacher-P available, >= 2 CLOSE candidates
ELIGIBLE_SINGLE_CANDIDATE  — Teacher-P available, exactly 1 CLOSE candidate
TEACHER_P_UNAVAILABLE      — grasp privilege invalid OR no close passes Teacher-P
TEACHER_P_AMBIGUOUS        — Teacher-P abstain due to multiple tied candidates
NO_CLOSE_CANDIDATE         — 0 CLOSE candidates from rule-based enumeration
FIELD_VALIDITY_FAIL        — required field has missing/NaN/non-finite values
OPEN_CONVENTION_FAIL       — decoded_open_bool inconsistent with env gripper
RC1A_REMAP_FAIL            — remapper produced 0 valid rows or invariant failure
PROVENANCE_FAIL            — file SHA mismatch vs E4C.1 inventory
OTHER_ABSTAIN              — catch-all for unexpected issues
```

## Prohibited for E4C.2

- Adjusting thresholds based on aggregate results
- Using attack outcomes, VIS success, random-sensitive as labels
- Teacher-P unavailable → negative
- Bronze labels as final truth
- Placement privilege (target coordinates absent in all 402 traces)

## Threshold Non-Tuning Pledge

No parameter value will be changed based on E4C.2 aggregate counts.
All thresholds frozen at the values stated above, which were frozen
in `src/gripper_attack/phase_detector.py` before E4C.2 authorization.

## Output Artifacts (to be committed separately)

```
tables/e4c_audit/l12_e4c2_trace_status.csv
tables/e4c_audit/l12_e4c2_close_candidates.csv
tables/e4c_audit/l12_e4c2_teacher_p_coverage.csv
tables/e4c_audit/l12_e4c2_task_summary.csv
tables/e4c_audit/l12_e4c2_failure_taxonomy.csv
tables/e4c_audit/l12_e4c2_output_hashes.csv
tables/e4c_audit/l12_e4c2_run_log.txt
reports/L12_E4C2_ELIGIBILITY_REPORT.md
```

## Stop Rule

```
TRAINING_STARTED: NO
```

After E4C.2 results are committed, D1b preregistration uses the eligible pool
to freeze train/val/test splits, leakage audit, model config, checkpoint rule,
and baseline protocol — before any training is authorized.
