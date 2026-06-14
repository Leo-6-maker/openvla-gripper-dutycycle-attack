# L12 DeepSeek Detector Training — D1 Preregistration

**Date**: 2026-06-15
**Stage**: D1 (preregistration only)
**Branch**: `exp/l12-critical-close-window-selector-20260615`
**HEAD**: `9232c4a6695f3ebe101e6581fddbef4cb23ebc9d`
**Parent**: `1458239`
**Worktree**: clean

---

## RESULT_CLASS: DATA_NOT_YET_ELIGIBLE

**TRAINING_STARTED: NO**

---

## Data Source Audit

Five potential data sources were audited against the frozen D1 protocol requirements:

| # | Source | Traces | RC1a Remap | Teacher-P | Labels | Attack-Outcome Labels | Verdict |
|---|--------|--------|------------|-----------|--------|----------------------|---------|
| 1 | E4C inventory (402 schema-passing) | 402 | NO | NO | NO | NO | NOT ELIGIBLE |
| 2 | E2 development set (12 remapped) | 12 | YES | YES | Partial | NO | NOT ELIGIBLE (dev-only, too small, no split) |
| 3 | Detector v0.3 training (K5/K5b/K5c) | 22 pairs | UNKNOWN | NO | YES | YES (cmd_specific, random_sensitive) | PROHIBITED |
| 4 | Detector v3 gold+vis1r ablation | ~72 pairs | UNKNOWN | NO | YES | YES (gold_3r, silver_positive_1r) | PROHIBITED |
| 5 | s20m0 train/val/test split | ~123 parents | UNKNOWN | NO | YES | YES (RAND labels) | PROHIBITED |

### Detailed findings

**Source 1 — E4C Inventory (402 traces)**
- Schema-validated: YES (required header fields, non-zero rows, valid SHA)
- RC1a remapped: NO — remap was never executed on any of the 402 traces
- Teacher-P evaluated: NO — `teacher_privileged_critical_close_anchor()` never called
- CLOSE-candidate enumeration: NO — `rule_based_close_predictor()` never run
- Invariant status: UNKNOWN
- Field validity status: UNKNOWN
- Teacher-P available count: UNKNOWN
- Multi-candidate count: UNKNOWN
- Blocking gap: E4C.2 (per-trace RC1a remap + Teacher-P evaluation) was explicitly DEFERRED by reviewer

**Source 2 — E2 Development Set (12 traces)**
- RC1a remapped: YES (`rc1a_corrected_v2_e1_5`)
- grasp_privilege_valid: 12/12 True
- placement_privilege_valid: 0/12 (expected — no target coordinate fields)
- Teacher-P anchors computed: YES (E2 audit)
- Tasks covered: 7/10 (alphabet_soup, bbq_sauce, butter, chocolate_pudding, cream_cheese, ketchup; missing: milk, orange_juice, salad_dressing, tomato_sauce)
- Already consumed in E4B development analysis: YES
- Formal train/val/test split: NO
- Formal per-candidate labels with (positive=Teacher-P, negative=other candidates): NO
- Blocking gaps: (a) 12 traces is a development set, not a training corpus; (b) already seen during E4B feature analysis; (c) no held-out split possible from 7-task coverage; (d) 3 tasks missing entirely

**Source 3 — Detector v0.3 Training (22 pairs)**
- Label type: `cmd_specific` (= command-sensitive attack response), `random_sensitive`
- These are VIS attack outcome labels — prohibited for clean-only Layer2 detector
- Also: RC1a provenance of underlying traces unverified against current remapper

**Source 4 — Detector v3 Gold+Vis1r Ablation**
- Label tiers: `gold_3r`, `silver_positive_1r` — derived from VIS attack response
- Prohibited for clean-only Layer2 detector

**Source 5 — s20m0 Train/Val/Test Split**
- Labels: S20F-J RAND labels — random-sensitive attack outcome labels
- Prohibited for clean-only Layer2 detector

---

## Minimum Requirements NOT MET

The following D1 hard gates cannot be satisfied with any existing data:

| Gate | Status | Blocking Gap |
|------|--------|-------------|
| Per-trace RC1a remap on training candidates | FAIL | Only 12/402 E4C traces have been remapped |
| Teacher-P available/abstain per trace | FAIL | Only 12 traces evaluated |
| CLOSE-candidate enumeration per trace | FAIL | Only 12 traces have candidate enumeration |
| Zero attack-outcome labels | FAIL | All existing labeled sets use prohibited labels |
| Trace-level grouped split (train/val/test) | FAIL | No split defined for eligible data |
| Same trace/SHA/group_id not across splits | FAIL | No split to audit |
| Teacher-P unavailable → abstain (not negative) | FAIL | No labeling protocol applied to eligible data |
| Sufficient per-task coverage for grouped generalization | FAIL | E2 set covers 7/10 tasks with 1-2 traces each |

---

## Path to Eligibility (E4C.2 — DEFERRED)

To make D1 eligible, the following must be executed (requires reviewer authorization):

1. Run RC1a remap on all 402 E4C schema-passing traces
2. Run Teacher-P evaluation on all remapped traces
3. Run rule-based CLOSE-candidate enumeration on all traces
4. Filter to Teacher-P-available traces with >=2 candidates
5. Audit invariants and field validity per trace
6. Create grouped train/val/test split (by task_key+state_id)
7. Run leakage audit (SHA, trace_id, group_id, task-state across splits)
8. Freeze manifest, config, and all artifacts with SHA256
9. Freeze baseline protocol on the same splits

---

## Files Created

| File | SHA256 |
|------|--------|
| `reports/L12_DEEPSEEK_DETECTOR_TRAINING_PREREG.md` | `7A40A58F1BC91CDEA9283DF06EBA757F88E8901E307AEDF2636CF3DC673B5DA3` |
| `tables/deepseek_detector/data_eligibility_audit.csv` | `EF3B3BDFE422FF64B5BE889A45850A162F3639BDFB104AF1EA9C43902094F1BE` |
| `tables/deepseek_detector/prereg_run_log.txt` | `8043932CA7FDE16B5C00B963113C0173E39F21ACA64B6D5C349C52B38E47B6A4` |

---

## Checklist

- [ ] Training manifest frozen: NO (no eligible data)
- [ ] Config frozen: NO (blocked on data)
- [ ] Split defined: NO (blocked on data)
- [ ] Leakage audit passed: NO (blocked on data)
- [ ] Baseline protocol frozen: NO (blocked on data)
- [ ] Training started: NO

---

## Reviewer Authorization Required

E4C.2 (RC1a remap + Teacher-P coverage on 402 historical traces) was explicitly DEFERRED.
D1 cannot proceed until E4C.2 is authorized and completed.

**NEXT_ACTION**: Await reviewer decision on whether to authorize E4C.2 unblock, or to provide an alternative eligible training dataset.
