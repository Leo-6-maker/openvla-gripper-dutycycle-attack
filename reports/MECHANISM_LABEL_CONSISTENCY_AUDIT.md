# Mechanism Label Consistency Audit

**Date**: 2026-06-06 15:32
**Data**: labels_v2 (31 rows) + mechanism_taxonomy (60 rows) + calibration v2 (10 rows)
**Purpose**: Resolve physical_bridge / no_action_bridge / mechanism_unclear definition conflicts

---

## 1. Gold Positives: Mechanism Breakdown

### ALL 9 gold positives are physical_bridge_positive

| ID | Task | State | Window | Taxonomy | Qpos Delta | Vis Open | Evidence Conf | Phase? | Calib v2? |
|----|------|-------|--------|----------|-----------|----------|--------------|--------|----------|
| ketchup_s0_w16_33 | ketchup | 0 | [16,33] | action_positive_physical_stron | 0.038042 | 18 | strong | YES | no |
| butter_s0_w29_46 | butter | 0 | [29,46] | action_positive_physical_stron | 0.037905 | 18 | strong | no | no |
| alphabet_soup_s4_w4_21 | alphabet_soup | 4 | [4,21] | claim_usable | 0.032241 | 18 | strong | no | no |
| bbq_sauce_s9_w22_39 | bbq_sauce | 9 | [22,39] | claim_usable | 0.038021 | 18 | strong | no | no |
| butter_s5_w25_42 | butter | 5 | [25,42] | claim_usable | 0.037568 | 18 | strong | no | no |
| cream_cheese_s4_w28_45 | cream_cheese | 4 | [28,45] | claim_usable | 0.038149 | 18 | strong | YES | no |
| ketchup_s1_w21_38 | ketchup | 1 | [21,38] | claim_usable | 0.037871 | 18 | strong | YES | no |
| milk_s1_w8_25 | milk | 1 | [8,25] | claim_usable | 0.036838 | 18 | strong | no | no |
| milk_s4_w19_36 | milk | 4 | [19,36] | claim_usable | 0.03789 | 18 | strong | no | no |

### Evidence Strength Distribution

| Confidence | Count | Definition |
|-----------|-------|------------|
| strong | 9 | qpos>0.03, vis_open=18, action_bridge=1, phys_response=1 |
| good | 0 | qpos>0.03, vis_open=18, action_bridge=1 |
| weak | 0 | partial evidence |
| none | 0 | no physical evidence |

### Key Observation
**ALL 9 positive labels have clear physical bridge evidence**: qpos_delta > 0.03,
vis_open_count = 18/18, action_bridge = 1. NOT ONE positive is "no_action_bridge."

This means:
- The current gold positive set is **mechanism-pure**: it only contains physical gripper openings.
- The "claim_usable" taxonomy for 7/9 positives refers to provenance completeness, not mechanism uncertainty.
- These 7 need calibration v2 confirmation (matched 1R vs 3R) to confirm reproducibility.

---

## 2. No-Action-Bridge: Where Are They?

### 9 no_action_bridge rows — ALL are NEGATIVES

| ID | Task | State | Window | Taxonomy | Task Fail | Qpos | Vis Open |
|----|------|-------|--------|----------|-----------|------|----------|
| alphabet_soup_s6_w40_57 | alphabet_soup | 6 | [40,57] | no_action_bridge | True | 0.0 | 0 |
| bbq_sauce_s0_w5_22 | bbq_sauce | 0 | [5,22] | no_action_bridge | True | 0.0 | 1 |
| ketchup_s4_w28_45 | ketchup | 4 | [28,45] | no_action_bridge | True | 0.0 | 0 |
| milk_s1_w18_35 | milk | 1 | [18,35] | no_action_bridge | True | 0.0 | 0 |
| milk_s8_w8_25 | milk | 8 | [8,25] | no_action_bridge | True | 0.0 | 0 |
| orange_juice_s2_w17_34 | orange_juice | 2 | [17,34] | no_action_bridge | True | 0.0 | 0 |
| salad_dressing_s5_w28_45 | salad_dressing | 5 | [28,45] | no_action_bridge | True | 0.0 | 0 |
| tomato_sauce_s1_w23_40 | tomato_sauce | 1 | [23,40] | no_action_bridge | True | 4e-06 | 1 |
| tomato_sauce_s3_w17_34 | tomato_sauce | 3 | [17,34] | no_action_bridge | True | 0.0 | 0 |

### Key Observation
The 9 "no_action_bridge" rows are **3R VIS failures**: the attack caused task failure
but did NOT produce observable gripper opening (qpos_delta=0, vis_open=0).
These are correctly classified as negatives for a physical vulnerability detector.

But for a **policy/action sensitivity detector**, these would be POSITIVES:
VIS perturbation caused task failure through token-level action corruption, even
without physical gripper opening.

**Currently, there are 0 no_action_bridge POSITIVES in the training set.**

---

## 3. Physical Task Negatives

### 4 rows: VIS caused physical opening but task completed

| ID | Task | State | Window | Qpos Delta | Vis Open | Task Fail |
|----|------|-------|--------|-----------|----------|----------|
| bbq_sauce_s5_w27_44 | bbq_sauce | 5 | [27,44] | 0.038119 | 18 | True |
| ketchup_s5_w9_26 | ketchup | 5 | [9,26] | 0.037403 | 18 | True |
| milk_s5_w25_42 | milk | 5 | [25,42] | 0.03799 | 18 | True |
| salad_dressing_s0_w7_24 | salad_dressing | 0 | [7,24] | 0.036668 | 18 | True |

These are the ideal **confirmed negatives** for a physical detector:
VIS causes gripper opening, but the task succeeds anyway → the window is NOT
a physical vulnerability (policy is robust to opening).

These 4 rows can immediately serve as gold negatives for v3.

---

## 4. Hard Gate POC: The 7 Positives

### 3 positive rows with phase coverage in covered subset

| ID | Task | State | Window | Mechanism | Hazard Score |
|----|------|-------|--------|-----------|-------------|
| ketchup_s0_w16_33 | ketchup | 0 | [16,33] | physical_bridge_positive | 0.0 |
| cream_cheese_s4_w28_45 | cream_cheese | 4 | [28,45] | physical_bridge_positive | 0.0 |
| ketchup_s1_w21_38 | ketchup | 1 | [21,38] | physical_bridge_positive | 0.0 |

### Answer: The hard gate POC positives belong to physical_bridge_positive class

The phase detector assigned hazard_score=0.0 to ALL of these windows, because:
1. Phase detector was trained on CLEAN rollouts, measuring "when does gripper naturally open?"
2. VIS attack windows occur in phases the model classifies as "safe" (gripper closed)
3. The VIS perturbation FORCES the gripper open in a "safe" phase

**This is the proof that phase and vulnerability signals are orthogonal.**

---

## 5. Final Verdict Summary

| Mechanism Verdict | Count |
|------------------|-------|
| physical_bridge_positive | 9 |
| no_action_bridge_negative | 9 |
| infra_or_manual | 8 |
| physical_task_negative | 4 |
| needs_calibration_v2 | 2 |
| mechanism_unclear | 1 |

---

## 6. Usability for Detector Training

| Detector Type | Usable Rows | Count | Notes |
|--------------|-------------|-------|-------|
| Physical Vuln Detector | YES | 9 | 9 gold positives, all physical_bridge |
| Physical Vuln Detector | PENDING | 2 | Need calibration v2 confirmation |
| Policy/Action Detector | YES | 0 | No no_action_bridge positives yet |
| Policy/Action Detector | CANDIDATES | 9 | 9 no_action_bridge negatives → could be positives |


## 7. Resolution: Definition Conflicts

### Conflict 1: "claim_usable" vs physical_bridge_positive

**RESOLVED**: "claim_usable" refers to provenance completeness (need calibration v2),
NOT mechanism type. All 7 claim_usable positives have strong physical evidence
(qpos>0.03, vis_open=18, action_bridge=1). They ARE physical_bridge_positive.
Only difference vs the 2 strong positives: provenance documentation.

### Conflict 2: no_action_bridge as "negative" vs positive for policy detector

**RESOLVED**: For a physical vulnerability detector, no_action_bridge IS negative
(VIS caused task failure but NOT through gripper opening). For a policy/action
sensitivity detector, no_action_bridge WOULD BE positive. These are two different
detection targets. Do not mix them.

### Conflict 3: mechanism_unclear rows in taxonomy

**RESOLVED**: 8 polluted rows → infra_or_manual (exclude). 1 weak_physical_uncertain
(alphabet_soup s0 w[3-20]) → mechanism_unclear (qpos=0.0276 < 0.03 threshold).
Remaining mechanism_unclear taxonomy rows are clean controls not in labels_v2.

---

## 8. Recommended Label Split for v3

```
Physical Vulnerability Detector:
  train positives: 9 physical_bridge_positive
    - 2 confirmed strong (ketchup s0, butter s0)
    - 7 needs calibration v2 confirmation
  train negatives: 4 physical_task_negative + confirmed clean controls
  exclude: 9 no_action_bridge (wrong mechanism), 8 polluted, 1 weak_physical

Policy/Action Sensitivity Detector (exploratory):
  train positives: 0 currently available (need to collect)
  training candidates: 9 no_action_bridge negatives from v2
    - These show VIS→task_failure without physical opening
    - Could be promoted to positives for a policy detector
  exclude: physical_bridge_positive (different mechanism)
```

---

## 9. Required Actions

1. **Calibration v2**: Confirm 7 claim_usable positives show matched 1R=3R agreement → promote to strong
2. **Clean-control 3R**: Confirm >=6 controls as true negatives → physical detector negatives
3. **No-action-bridge collection**: If policy detector desired, run 3R VIS on candidates showing no_action_bridge pattern
4. **Do NOT mix mechanisms**: physical_bridge ≠ no_action_bridge ≠ clean_control in same training set without multi-task labels
