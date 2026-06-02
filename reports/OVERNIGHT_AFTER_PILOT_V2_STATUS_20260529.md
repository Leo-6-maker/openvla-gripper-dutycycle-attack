# Overnight Status — After Pilot V2

**Generated**: 2026-05-30 02:55 CST
**Server**: 10.60.133.4
**Final states**: task_dependent_oracle_sensitivity, current_proxy_family_ineffective, sustained_proxy_design_ready

---

## 1. Pilot V2 Results (24/24 rollouts)

| Condition | tomato_sauce | milk | Total | vs Clean |
|-----------|-------------|------|-------|----------|
| clean | 3/3 | 3/3 | 6/6 | baseline |
| oracle_open | 0/3 | 2/3 | 2/6 | -4 |
| random_control | 2/3* | 3/3 | 5/6 | -1* |
| gripper_inversion_proxy | 3/3 | 3/3 | 6/6 | 0 |

\* random tomato s0: 0 triggers, 0 attacks — natural policy failure, not attack-induced.

## 2. Oracle Expansion (12/12 rollouts)

| Task | Clean | Oracle | Classification |
|------|-------|--------|----------------|
| salad_dressing | 3/3 | 3/3 | true_oracle_robust |
| ketchup | 3/3 | 3/3 | true_oracle_robust |

## 3. Complete Object Oracle Sensitivity

| Task | Oracle SR | avg_qpos_delta | avg_burst | Sensitivity |
|------|-----------|---------------|-----------|-------------|
| tomato_sauce | 0/3 | -0.0035 | 125 | **HIGH** |
| milk | 2/3 | -0.0044 | 52 | MEDIUM |
| salad_dressing | 3/3 | -0.0143 | 74 | ROBUST |
| ketchup | 3/3 | -0.0105 | 35 | ROBUST |

**Critical finding**: Oracle physically opens the gripper on ALL tasks (qpos goes from ~+0.030 to ~+0.017). Salad_dressing has the LARGEST qpos response (-0.0143) but still succeeds. Tomato has the SMALLEST qpos response (-0.0035) but fails completely. This proves the failure mechanism is about disrupting a task-specific critical contact phase, not about brute-force gripper opening.

## 4. Proxy Diagnosis

| Metric | Oracle | Proxy |
|--------|--------|-------|
| Attack burst | 28-160 steps | 9-19 steps |
| qpos response | -0.003 to -0.014 | ~0.000 |
| Feedback loop | YES | NO |
| Official SR impact | 4/6 fail | 0/6 fail |

**Root cause**: attack_remaining=5 for both. Oracle gets sustained attack via feedback (gripper opens → detector re-triggers → attack_remaining resets). Proxy lacks feedback because qpos doesn't respond to brief inversion. Without code change, proxy cannot be fixed.

## 5. Sustained Proxy Design

Design spec ready at `reports/SUSTAINED_PROXY_ATTACK_DESIGN_SPEC.md`. Key changes needed:
1. Decouple attack_burst_steps from detector_trigger_duration
2. Add sustained-open hold during burst
3. Optional qpos-feedback extension

**No code changes applied tonight.**

## 6. GPU/Xid Status

| GPU | Status |
|-----|--------|
| 0 | Quarantined (other user, Xid13 history) |
| 1-7 | All idle, no fresh Xid |
| Last Xid | 2026-05-29 14:03 (GPU0 only) |
| Disk | 35%, OK |

## 7. Quarantine Log

| Item | Reason |
|------|--------|
| random_control tomato s0/s1/s2 (original pilot_v2) | Duplicate worker collision |
| ketchup s1 (detector-clean, original root) | Partial output from killed loop |
| salad_dressing duplicate rerun | task_id=2 mislabeled as ketchup |

## 8. Generated Artifacts

### Reports (10)
- PILOT_V2_SUCCESSFIX_FINAL_STATUS.md
- PILOT_V2_PRELAUNCH_BLOCKING_AUDIT.md
- GIT_PROVENANCE_BLOB_EQUIVALENCE_AUDIT.md
- PILOT_V2_PROXY_WEAKNESS_DIAGNOSIS.md
- PROXY_BURST_DURATION_CODE_AUDIT.md
- SUSTAINED_PROXY_ATTACK_DESIGN_SPEC.md
- TASK_DEPENDENT_GRIPPER_SENSITIVITY_ANALYSIS.md
- OBJECT_ORACLE_EXPANSION_STATUS.md
- DETCLEAN_DUPLICATE_LOOP_STOP_SNAPSHOT.md
- OVERNIGHT_AFTER_PILOT_V2_STATUS_20260529.md (this file)

### Tables (25+)
- Pilot V2: 13 CSVs in pilot_v2 output root/tables/
- Oracle expansion: 7 CSVs in expansion output root/tables/
- Detector-clean: merged_summary, quarantine, task_id_mapping
- Git provenance: blob_equivalence

## 9. Recommended Next Steps

### Tomorrow priority:
1. **Manual audit**: tomato_sauce oracle failures (verify drop/slip mechanism)
2. **Code review**: sustained proxy design spec → implement attack_burst_steps
3. **Object-wide sensitivity table**: optionally test remaining Object tasks with oracle
4. **CQ evaluation**: run on all pilot v2 + expansion episodes

### Do NOT:
- Run proxy sweep without code changes
- Claim VIS attack success
- Claim universal detector
- Use pilot v1 as evidence
- Modify code without review

### Final wording:
"Detector-selected windows are not uniformly vulnerable across Object tasks. Oracle-open interventions show strong causal sensitivity for tomato_sauce and moderate sensitivity for milk, while salad_dressing and ketchup remain robust despite physical gripper opening. The current gripper_inversion_proxy is ineffective because it lacks sustained physical response. Sustained proxy design is specified and ready for implementation after code review."
