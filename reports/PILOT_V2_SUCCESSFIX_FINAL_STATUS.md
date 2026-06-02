# Pilot V2 Final Status — Object Detector Matched Attack Pilot

**Generated**: 2026-05-30 02:10 CST
**Server**: 10.60.133.4
**Status**: pilot_v2_completed_strong_oracle_only

---

## 1. Executive Summary

Pilot v2 completed 24/24 rollouts across 2 tasks × 3 states × 4 conditions. The detector-selected windows are **causally sensitive to forced gripper opening** (oracle_open causes 4/6 failures, including 3/3 on tomato_sauce). However, the current `gripper_inversion_proxy` does not reduce official success (6/6). Random_control causes 1/6 failure but that episode had 0 detector triggers (natural policy failure, not attack-induced).

**Bottom line**: Detector finds gripper-sensitive windows. Oracle proves causality. Proxy needs calibration.

---

## 2. Experimental Design

| Parameter | Value |
|-----------|-------|
| Tasks | tomato_sauce (task_id=5), milk (task_id=7) |
| States | s0, s1, s2 per task |
| Conditions | clean, oracle_open, random_control, gripper_inversion_proxy |
| Detector | ProprioNoStep TCN (38,602 params, 16-step history, 13 features) |
| Detector SHA | `4b3f3d479d6bbb92b2bd15cffec0be587bf221dc81663aaff93e44afdd9c7b1f` |
| Attack type | Command-layer gripper inversion proxy (NOT VIS PGD) |
| Total rollouts | 24 |

---

## 3. Provenance

| Field | Value |
|-------|-------|
| Server HEAD | `c62214fabe3d9991029a3a450a9a1f0f4de75f14` |
| Remote audited freeze | `087044390498f271fddfe009d95e6701fc9450fd` |
| Blob equivalence | 6/6 pilot-critical files identical |
| Success predicate | `success_official = done_any` (LIBERO official) |
| Fresh Xid | None (last Xid 2026-05-29 14:03, GPU0 only) |
| Working tree | Clean |

---

## 4. Official SR Results

| Condition | Milk | Tomato | Total | Delta vs Clean |
|-----------|------|--------|-------|----------------|
| clean | 3/3 | 3/3 | **6/6** | baseline |
| oracle_open | 2/3 | 0/3 | **2/6** | **-4** |
| random_control | 3/3 | 2/3 | **5/6** | -1* |
| gripper_inversion_proxy | 3/3 | 3/3 | **6/6** | 0 |

\* random_control tomato s0: 0 detector triggers, 0 attacks — natural policy failure, not attack-induced.

### Matched State Detail

| Task | State | Clean | Oracle | Random | Proxy |
|------|-------|-------|--------|--------|-------|
| milk | s0 | True | True | True | True |
| milk | s1 | True | True | True | True |
| milk | s2 | True | **False** | True | True |
| tomato | s0 | True | **False** | **False*** | True |
| tomato | s1 | True | **False** | True | True |
| tomato | s2 | True | **False** | True | True |

\* tomato s0 random: natural failure (0 triggers)

---

## 5. Trigger / Attack Audit

| Condition | Total Triggers | Attack Steps | orig≠att Steps |
|-----------|---------------|-------------|-----------------|
| clean | 178 | 0 | 0 |
| oracle_open | 506 | 514 | 294 |
| random_control | 131 | 132 | 68 |
| gripper_inversion_proxy | 67 | 68 | 68 |
| **Total** | **882** | **714** | **430** |

### Key observations:

- **Clean**: 178 triggers, 0 attacks — attack guard works correctly.
- **Oracle**: 506 triggers (highest), 514 attacks — heavy sustained attack. Failed episodes hit max_steps=290.
- **Random**: 131 triggers, 132 attacks — moderate attack activity. Milk 3/3 survives.
- **Proxy**: 67 triggers (lowest), 68 attacks — brief attack bursts. Every attack step changed action. 6/6 survives.

---

## 6. Gripper Response

| Condition | att_gripper_mean | Direction |
|-----------|-----------------|-----------|
| oracle_open | +1.00 | Full open command |
| proxy | -0.50 to -0.99 | Inverted sign (variable) |
| random | -0.20 to 0.00 | Random, near-neutral |

Oracle consistently pushes gripper to extreme open. Proxy inverts the gripper sign but the magnitude and timing may be insufficient to break grasp stability.

---

## 7. Interpretation

### Strong evidence:
1. **Detector windows are gripper-sensitive**: oracle_open causes 4/6 SR drop. Tomato 3/3 completely broken.
2. **Task-dependent sensitivity**: Milk survives oracle 2/3, tomato 0/3. Milk grasps may be mechanically more robust.
3. **Attack guard works**: Clean 6/6 has 0 attack_applied.

### Weak/negative evidence:
1. **Proxy does not reduce official SR**: 6/6 success. Attack bursts are short (9-18 steps) and gripper inversion magnitude may be insufficient.
2. **Random_control near-baseline**: 5/6 success (1 natural failure). Random perturbation ineffective.

### What this means:
The detector identifies windows where forced gripper opening breaks the task. The current command-layer proxy inverts gripper sign but does not cause SR drop — likely because it doesn't open the gripper wide enough or long enough to break contact.

---

## 8. Quarantine Log

| Item | Reason |
|------|--------|
| random_control tomato s0/s1/s2 (original) | Duplicate worker collision — contaminated |
| ketchup s1 (detector-clean) | Partial output from killed duplicate loop |
| salad_dressing duplicate rerun | task_id=2 mislabeled as ketchup |

---

## 9. Recommended Next Steps

1. **Manual audit**: tomato oracle failures (s0,s1,s2) — verify object drop/contact loss
2. **Proxy calibration**: Increase attack burst duration, adjust inversion magnitude, or sweep threshold
3. **Expand Object tasks**: Run oracle on more Object tasks to confirm detector sensitivity generalizes
4. **CQ evaluation**: Run contact-quality metrics on proxy/random episodes to detect sub-SR degradation
5. **True VIS PGD**: Only after command-layer evidence is solid, and only with explicit approval

---

## 10. Output Artifacts

| Artifact | Path |
|----------|------|
| Pilot v2 output root | `/data/liuyu/outputs/milestone_2f_object_detector_matched_pilot_v2_successfix_20260529` |
| Aggregated manifest | `.../tables/official_clean_artifact_rich_manifest.csv` |
| State selection | `tables/pilot_v2_verified_state_selection.csv` |
| Detector-clean merged | `tables/pilot_v2_detector_clean_merged_summary.csv` |
| Git provenance | `reports/GIT_PROVENANCE_BLOB_EQUIVALENCE_AUDIT.md` |
| Pre-launch audit | `reports/PILOT_V2_PRELAUNCH_BLOCKING_AUDIT.md` |
| Overnight status | `reports/PILOT_V2_SUCCESSFIX_FINAL_STATUS.md` |

---

**Final status**: `pilot_v2_completed_strong_oracle_only`
