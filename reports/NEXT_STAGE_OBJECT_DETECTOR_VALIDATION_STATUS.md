# Object Detector Validation Status — Stage Summary

**Date**: 2026-05-29
**Branch**: `eval/official-libero-clean-20260525`

## Final State

### Stage 0: Worker Sanity ✓
- GPU0 (quarantined Xid13), GPU7 (idle, visual extraction queue)
- GPUs 1-6: Goal-100 + L10-A + L10-B running, all stable
- No fresh Xid. Disk 35% (601G/1.8T), inodes 1%.
- **L10 overlap**: FALSE ALARM. `--task_start`/`--task_count` correctly respected.
  - L10-A (GPU4,5): tasks 0-4, task_id=0 (LIVING_ROOM_SCENE2), 7 eps done
  - L10-B (GPU1,3): tasks 5-9, task_id=5 (STUDY_SCENE1) complete, task_id=6 in progress

### Stage 1: Replay Audit ✓
- **Tables generated**: 4 CSVs + per-task results in report
- **Best model**: ProprioNoStep (composite score 0.6727)
- **Results**:

| Model | Coverage | FE Rate | Miss | Latency | FP Fail |
|-------|----------|---------|------|---------|---------|
| **ProprioNoStep** | **0.9909** | 63.6% | 0 | 0.2 | 0 |
| VisualNoStep | 0.8909 | 54.5% | 0 | 0.2 | 0 |
| VisualProprioNoStep | 0.9000 | 63.6% | 0 | 2.2 | 0 |

- **False-early tolerance**: VisualNoStep drops to 2 far-wrong FE at ±5 tol (best precision). ProprioNoStep has 5 far-wrong FE (most sensitive).
- **Decision**: ProprioNoStep = primary for coverage. VisualNoStep = supporting phase-confirmation signal. Fusion needed for FE reduction.

### Stage 2: Fusion Calibration (Running)
- **VAL results** (partial, before crash):

| Strategy | Coverage | FE Rate |
|----------|----------|---------|
| Max_Prop_Vis | 0.8923 | 38.5% |
| VisualOnly | 0.8846 | 30.8% |
| Mean_Prop_Vis | 0.8769 | 30.8% |
| Weighted_0.7P_0.3V | 0.7769 | **15.4%** ← lowest FE |
| Hysteresis_Prop_Vis | 0.8231 | 30.8% |

- **Fixed script re-running** — full test results pending
- **Key signal**: Weighted 0.7P+0.3V reduces FE from 64%→15% but loses 22% coverage. Hysteresis is middle ground.

### Stage 3: Shadow Validation ✓
- Script: `scripts/run_detector_shadow_rollout.py` (prepared)
- Plan: `reports/OBJECT100_ONLINE_SHADOW_VALIDATION_PLAN.md`
- 2 tasks × 5 states = 10 shadow episodes
- **Blocked**: awaiting Stage 2 fusion finalization + user approval

### Stage 4: Attack Pilot Plan ✓
- Plan: `reports/OBJECT_DETECTOR_MATCHED_ATTACK_PILOT_PLAN.md`
- 2 tasks × 5 states × 4 conditions = 40 rollouts
- 4 gates: Oracle, VIS-vs-Random, CQ, Detector
- **Blocked**: awaiting shadow validation + user explicit approval

### Stage 5: CQ Metrics ✓
- Report: `reports/CQ_METRICS_ATTACK_PILOT_READINESS.md`
- Status: PARTIALLY READY
- `infer_failure_phase()` covers 5 failure modes from `grasp.py`
- CQFR/CQSR aggregation: trivial (~20 lines), not yet implemented
- Premature release / drop detection: nice-to-have, not blocking

### Stage 6: Goal/L10 Workers (Running)
| Worker | Progress | Current Task | GPU |
|--------|----------|-------------|-----|
| Goal-100 | 31/100 | task 4/10 (wine_bottle done, moving to next) | 2,6 |
| L10-A | 7 eps | task_id=0 (LIVING_ROOM_SCENE2) | 4,5 |
| L10-B | 10+2 eps | task_id=5 done, task_id=6 in progress | 1,3 |

- Do not aggregate until workers complete
- GPU7: idle, available for Goal/L10 visual extraction after rollouts

### normalized_step Issue
- `proprio_causal_student.py:46` includes `normalized_step` in legacy NUMERIC_FEATURES
- **NOT** used in current deployment training (both tmp_train scripts use own ALLOWED lists)
- Config explicitly lists `normalized_step` in `forbidden_inputs`
- 7/7 audit tests pass
- **Risk**: LOW (legacy only). Fix before universal training by removing from NUMERIC_FEATURES.

### Tests Run
- `tests/v4/test_normalized_step_deployment_audit.py`: 7/7 passed
- Parser v2 static audit: 40/40 correct
- Parser v2 cross-validation: Object-100 0 lost, 0 gained
- Previous: 25/25 parser/privileged tests passed (from Milestone 2D)

## Blockers

| Stage | Status | Blocker |
|-------|--------|---------|
| Attack pilot | BLOCKED | Shadow validation not launched |
| Shadow validation | READY | Awaiting user approval |
| Fusion calibration | RUNNING | Fix in progress |
| Universal detector | BLOCKED | Goal/L10 labels stale, not refreshed |
| Goal/L10 aggregation | BLOCKED | Workers still running (~4-6h ETA) |

## Boundary Statement
This stage validates Object-100 detector behavior and prepares online shadow / matched attack pilot infrastructure. No attacks have been run. No VIS/random/oracle/manual outcomes have been used to train or tune the detector. Privileged object/target state is teacher-label-only and does not enter deployed student features. The universal cross-suite detector remains blocked until Goal-100 and L10-100 parser-v2 reruns complete and are fully audited.
