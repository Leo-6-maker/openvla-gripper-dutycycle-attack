# V2 Layer 1/2 Code Audit

**Date:** 2026-06-18
**Branch:** exp/l2-sc5-student-v2-20260618
**Base commit:** e7c3355

## A. Frozen D5 Detector Behavior

### A1. Candidate mechanism
D5 produces candidates ONLY at CLOSE onset transitions (d5_frozen_feature_adapter_v1.py:398-416):
1. `raw_crossing` — raw_gripper transitions from >0.5 to <=0.5
2. `close_onset` — first step where env_gripper > 0.5 and close_streak was 0
3. `close_streak == 1` — first step of new close streak

All three are close-transition gates. No candidate fires for open or idle states.

### A2. Cannot produce continuous output during stable_carry
During sustained closed-gripper carry, `close_streak` increments past 1, so `close_onset` (needs streak=0) and `close_streak==1` never re-fire. `raw_crossing` requires open-to-close transition, which doesn't occur during closure. Result: exactly ONE candidate per close event.

### A3. First-trigger lock and abstain
- `abstain` field gates candidate (d5_frozen_online_detector_v1.py:197-202)
- `self.has_emitted` ensures at most one emission per episode
- After emit_step is set, all subsequent candidates silently swallowed

### A4. Why D5 cannot be modified into SC5 detector
D5 is architecturally a close-ONSET detector. Its 16-element feature vector is dominated by onset-specific features (close_onset, close_streak, close_onset_qpos_bonus, close_streak_bonus, raw_crossing). An SC5 detector fires during sustained closed-gripper carriage — a regime D5 explicitly skips. Converting requires: rewriting candidate gate, adding duration/oscillation features, retraining MLP, breaking all SHA256 bindings. That is a NEW detector, not a modification.

**Conclusion: D5 MUST remain as baseline only.**

---

## B. Existing 2C Student Behavior

### B1. Model architecture
`ProprioCausalMLP` (proprio_causal_student.py:245-267): pure feedforward MLP with shared 2-layer 64-dim backbone + 4 output heads (phase, hazard, release_safe, confidence). NO recurrence or temporal convolution. Each timestep receives one feature row independently. The config YAML specifies `type: causal_tcn, history_len: 16` but no TCN class is implemented.

### B2. Inputs — normalized_step IS present
`NUMERIC_FEATURES` (line 29-47): 17 features including `normalized_step`. 2 categorical features: `mechanism_type`, `parse_confidence`. Forbidden input substrings block object pose, target pose, teacher windows, identity cols, visual features.

### B3. Phase/hazard/release-safe definitions
7 phases: `approach`, `grasp_close`, `lift`, `carry`, `pre_release_hazard`, `release_safe`, `other`. Hazard and release_safe from teacher labels `teacher_hazard`, `teacher_release_safe` as binary flags. Hazard name: `window_full_hazard`, variant: `B_window_full`.

### B4. Row-level evaluation, no first-trigger lock
Replay evaluates at row-level (causal_replay_episode:140-230). For each step, it applies causal mask (history[:t]) and independently predicts. No "if triggered, lock" flag. NO first-trigger lock. NO K10 window completeness check. Config specifies `trigger_durations: [5,10]` and `cooldown: 10` but code does NOT implement them.

### B5. Training split
`split_mode="task_id"` (train_proprio_causal_student.py:49). Splits by task_id within each suite (70/15/15). No cross-task contamination.

---

## C. Existing Teacher

### C1. Current phase vocabulary
9 phases: approach, grasp_close, stable_grasp, first_lift, stable_carry, pre_place_unsupported, release_safe, recovery_or_regrasp, abstain_unsupported. FAILURE_CRITICAL_PHASES = {stable_carry, pre_place_unsupported}.

### C2. find_teacher_anchor() STILL prefers pre_place
v2_privileged_teacher.py:265-270: first filters for pre_place_unsupported among failure-critical phases, returns highest-confidence one (reason: 'pre_place_unsupported_preferred'). Only if none exist falls back to stable_carry midpoint (272-277). **This function MUST NOT be called for SC5. New find_sc5_anchor_v2() required.**

### C3. Teacher config NOT hash-frozen
TeacherConfig.version is informational string ('v2_teacher_fixed_semantics'), not cryptographic. Nothing prevents silent parameter drift.

### C4. Fail-closed on missing fields
Fully enforced via `_check_required_fields` (63-70). Returns None → abstain_unsupported with confidence 0.0. No default-zero fill.

### C5. Phase 3 SC5 rule origin
SC5 = earliest stable_carry_start + 5, K=10. Derived from Phase 3 command-hold pilot (3 dev + 2 held-out states showing consistent SC5 failure). NOT from find_teacher_anchor().

---

## D. Data and Splits

### D1. Legacy dataset
Old 400-episode / 87,474-row data from milestone_2e2_object100_privileged_artifact_rich_20260527. All traces have complete privileged fields (object_pose, target_pose). Source: step_records.jsonl with teacher_privileged_state_available=True.

### D2. Split method
`assign_splits` supports task_id and episode_key modes. task_id mode ensures train/val/test task_id sets are disjoint. Config specifies episode_key; training script defaults to task_id (stricter boundary).

### D3. Forbidden features
Leakage test confirms: normalized_step, task_id, state_id, run_id, episode_key, object_pose, target_pose, teacher_window, attack_outcome, manual_outcome excluded from training inputs.

### D4. Butter held-out
s8, s9, s11 must be strict held-out. s5 CLEAN_FAIL excluded. s3: no-valid-corridor / abstain test only.

---

## E. Scientific Mismatch Summary

| Layer | Current State | Required for SC5 |
|-------|---------------|------------------|
| D5 | CLOSE onset only | Baseline only, cannot be SC5 |
| Student MLP | Includes normalized_step | Must remove |
| Student MLP | Row-level, no trigger lock | Episode-level, first-trigger |
| Student MLP | No K10 check | Full K10 corridor check |
| Teacher anchor | Prefers pre_place | SC5 = sc_start + 5 |
| Teacher config | Not hash-frozen | Must freeze |
| Dataset labels | Old hazard/release | New SC5 corridor labels |

---

## F. Reuse Plan

### FILES_TO_KEEP_FROZEN
- src/gripper_attack/d5_frozen_*.py (all SHA-bound)
- src/gripper_attack/attack_adapter.py
- scripts/stageb/run_l3_d5_vis_temporal.py
- scripts/stageb/audit_l3_d5_vis_temporal_v3.py
- artifacts/v2_phase3_anchor_manifest.json
- configs/d5_v1_production_bundle.json

### FILES_TO_REUSE_WITH_WRAPPER
- src/utils/proprio_causal_student.py (ProprioCausalMLP, split logic, normalization)
- tests/v4/test_proprio_student_*.py
- src/gripper_attack/v2_privileged_teacher.py (add find_sc5_anchor_v2)

### FILES_TO_ADD
- src/gripper_attack/sc5_streaming_features_v2.py
- src/gripper_attack/sc5_trigger_runtime_v2.py
- scripts/stageb/build_sc5_student_dataset_v2.py
- scripts/stageb/train_sc5_student_v2.py
- scripts/stageb/run_sc5_student_replay_v2.py
- scripts/stageb/audit_sc5_student_replay_v2.py

---

## G. Classification

```
CURRENT_LAYER1: D5 close-onset candidate-based (discrete, SHA-bound)
CURRENT_LAYER2: ProprioCausalMLP with normalized_step, row-level, no trigger lock
STUDENT_PHASE_VOCABULARY: 7-class (approach..other), not aligned with Teacher 9-class
SC5_CORRIDOR_CHECK: NOT IMPLEMENTED in any existing replay
FIRST_TRIGGER_LOCK: NOT IMPLEMENTED
SC5_RULE: Phase 3 derived (sc_start+5), NOT from find_teacher_anchor()
```
