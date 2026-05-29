# Parser/Teacher V2 Hardcoding & Leakage Audit

**Date**: 2026-05-29
**Scope**: `src/`, training scripts, config files

## 1. Hardcoded Window Audit

### state5/state7 Special Windows
- [ ] `state5` → window (75-84) references in detector path: **PASS (0 found)**
- [ ] `state7` → window (78-87) references in detector path: **PASS (0 found)**
- [ ] Any task_id → window step-index hardcode: **PASS (0 found)**
- [ ] Any numeric window range (35-45, 75-84, 78-87) in new code: **PASS (0 found)**

Result: No hardcoded state-specific windows in any detector or parser code.

## 2. Privileged State Leakage Audit

### Teacher-only fields correctly isolated
- `object_pose_json`, `target_pose_json`, `object_to_target_distance`, `object_height_delta_from_start` — only in `src/utils/libero_privileged_state.py` (teacher module)
- FORBIDDEN_INPUT_SUBSTRINGS in `src/utils/proprio_causal_student.py:60-80` properly blocks these from student input
- `tests/v4/test_teacher_privileged_no_student_leakage.py` — 10 tests, all passing

### Deployment feature schema
- Object-100 training (tmp_train_obj100.py, tmp_train_visual.py): 13 proprio fields, no normalized_step, no object/target pose — **PASS**
- Config (`configs/proprio_no_step_object100_b_window_full.yaml`): defines 13 allowed proprio inputs, forbidden list — **PASS**

## 3. Attack/Manual/Random Outcome Isolation

- `src/utils/autowindow_protocols.py:20`: `uses_attack_outcome: False` — **PASS**
- `src/utils/autowindow_validation.py:58`: validates `uses_attack_outcome is False` — **PASS**
- `src/utils/condition_protocols.py`: defines attack protocols (oracle, random, VIS) — these are ATTACK-SIDE only, not used for teacher labels — **PASS**
- Attack adapter (`src/gripper_attack/attack_adapter.py`): `random_start` is attack parameter, not label contamination — **PASS**
- No teacher/label code references `attack_outcome`, `manual_outcome`, `oracle_outcome`, `random_outcome`, `VIS_OUTCOME` — **PASS**

## 4. Mechanism Classification Audit

- `classify_mechanism()` in `src/utils/libero_privileged_state.py`: regex-based, no hardcoded task_id→mechanism map — **PASS**
- All 40 LIBERO tasks classified correctly (static audit passed) — **PASS**
- Object-100 v1=v2 regression passed (0 lost, 0 gained) — **PASS**

## 5. Issues Found

### ISSUE-1: `normalized_step` in NUMERIC_FEATURES (LOW RISK)
- **File**: `src/utils/proprio_causal_student.py:46`
- **Problem**: `NUMERIC_FEATURES` includes `"normalized_step"` which is explicitly forbidden for deployment
- **Impact**: This is the Milestone 2C utility module. Current Object-100 training scripts (`tmp_train_obj100.py`, `tmp_train_visual.py`) use their OWN feature lists WITHOUT `normalized_step`. So current training is NOT affected.
- **Risk**: If someone reuses this utility module for future training, `normalized_step` would leak in.
- **Fix**: Remove `"normalized_step"` from `NUMERIC_FEATURES` or add it to `FORBIDDEN_INPUT_SUBSTRINGS`.
- **Verdict**: Fix before universal detector training.

### ISSUE-2: `recent_close_streak` / `recent_open_streak` / `recent_gripper_flip_count` in NUMERIC_FEATURES (INFO)
- **File**: `src/utils/proprio_causal_student.py:43-45`
- **Problem**: These derived features are not in the Object-100 training ALLOWED list. They represent gripper history features.
- **Impact**: Not used in current training. May or may not be useful.
- **Verdict**: Document and decide before universal training.

## 6. Summary

| Category | Status | Issues |
|----------|--------|--------|
| Hardcoded windows | PASS | 0 |
| Privileged state leakage | PASS | 0 |
| Attack/manual outcome leakage | PASS | 0 |
| Mechanism hardcoding | PASS | 0 |
| normalized_step leakage | LOW RISK | 1 (not in current training) |
| Derived feature schema drift | INFO | 1 (not in current training) |

## Decision
- Current Object-100 training is CLEAN.
- Fix ISSUE-1 before universal detector training.
- ISSUE-2 is informational — decide during combined schema design.
