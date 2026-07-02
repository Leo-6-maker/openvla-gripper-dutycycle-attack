# Existing Evidence Freeze Report — 2026-07-02 (REVISION 3)

> This report supersedes `reports/server_audit/SERVER_EXPERIMENT_STATUS_AUDIT_20260702.md` for gate, classification, and evidence status.
> Previous aggregate files `object_emitted_parent_set_141.csv` and `object_no_emission_21.csv` are superseded by per-episode CSVs and cross-condition reconciliation.

## Executive Summary

Object condition totals verified. Cross-condition key matching confirmed (162 keys, 0 mismatches). CLEAN2000 census committed (2000 rows). **Critical finding: all Object artifacts used `upstream_tf_jpeg` preprocessing, not `official_pil_lanczos`** — protocol description must be corrected. **All 1350 teacher labels are placeholder defaults** (anchor=0, confidence=0.5, window=[0-10]) — CLEAN2000 cannot be used for detector timing training.

**Gate: HOLD_AUDIT_INCOMPLETE**

---

## Server Quiescence

SERVER_EXECUTION_QUIESCENT = PASS. PROCESS_STOPPED = PASS. LAST_ARTIFACT_ATOMICITY = UNKNOWN.

---

## CLEAN300: ARCHIVED AND SUPERSEDED

---

## Object Frozen Evidence

**Status: FROZEN_REPORTED_RESULT_WITH_PROVENANCE_SEAL_PENDING**

### Verified
- OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS (all 6 conditions match)
- OBJECT_CROSS_CONDITION_KEY_MATCH = PASS (162 keys, TRUE=EARLY=ORACLE, 0 mismatches)
- TRUE_T10_EMISSION_SPLIT = PASS (141 emitted + 21 no-emission, per-episode rows)
- 930-row master ledger with 29 fields including full 64-char SHA256, artifact paths, episode keys

### Protocol Deviation (CRITICAL)
All episodes record `preprocess_backend = upstream_tf_jpeg` with `preprocess_uses_jpeg=True`.
Claimed frozen protocol says `official_pil_lanczos`. This is the PR #43 draft protocol.
Epsilon, PGD steps, K, token, route, fallback, arm gate NOT recorded in episode_summary.json.

### EARLY_SHIFT Correction
All 141 EARLY_SHIFT episodes have `attack_applied=True`. The 42 "no-emission" episodes are "attack applied but detector silent" — previous "natural difficulty" framing was incorrect. FR=30.5% (43/141), lower than TRUE_T10 emitted-only 100%, consistent with timing specificity but early-shift is not "harmless."

---

## CLEAN2000 Authority Census

**Status: CLEAN2000_FROZEN — CENSUS ONLY. NOT READY FOR DETECTOR TIMING TRAINING.**

### Classification (Mutually Exclusive, Sum = 2000)

| Category | Count |
|---|---|
| PRIMARY_ELIGIBLE (task_success=True, teacher_eligible=True) | 1043 |
| CLEAN_FAILURE_SAFETY (task_success=False, teacher_eligible=True) | 307 |
| SUPPLEMENTARY_EVENT (teacher_eligible=False — task type out of scope) | 650 |
| **TOTAL** | **2000** |

### Teacher Label Audit (CRITICAL)

**All 1350 "valid" teacher labels are placeholder defaults:**

| Field | Value | Notes |
|---|---|---|
| teacher_anchor_step | 0 (100% of valid labels) | Single unique value across all 1350 |
| teacher_confidence | 0.5 (100% of valid labels) | Single unique value; matches default threshold |
| teacher_window | [0-10] (100%) | Identical window for every episode |
| n_unique_anchors | 1 | All 1350 share anchor=0 |
| n_unique_confidences | 1 | All 1350 share confidence=0.5 |

These are NOT real teacher timing labels. They are placeholder defaults — likely filled when the teacher model could not produce genuine anchor-step predictions. The index has 2000 entries with valid JSON schema, but the 1350 "valid" entries contain no timing signal.

**CLEAN2000 can be used for:**
- Success/failure census ✓
- Mechanism scope (which tasks are pick-place) ✓
- Per-suite/per-task statistics ✓

**CLEAN2000 CANNOT be used for:**
- Detector timing training (no real anchor steps)
- Confidence-weighted loss (no real confidence)
- Per-episode mechanism analysis (no real teacher_event_id)
- Window-based feature extraction (window=[0-10] is constant)

### Supplementary Episodes (650)

These have `teacher_label_valid=False`, `teacher_anchor_step=-1`, `teacher_confidence=0.0`, `teacher_invalid_reason="teacher_ineligible"`. The schema supports this abstention pattern: the record exists (`label_present_in_index=True`), the schema is valid for abstention records, and the reason is explicit. These are **valid explicit abstentions** for mechanism-ineligible tasks, not label gaps.

### Per-Suite

| Suite | Total | Teacher Eligible | PRIMARY | CLEAN_FAILURE | SUPPLEMENTARY |
|---|---|---|---|---|---|
| libero_spatial | 500 | 500 | 411 | 89 | 0 |
| libero_object | 500 | 500 | 367 | 133 | 0 |
| libero_goal | 500 | 300 | 234 | 66 | 200 |
| libero_10 | 500 | 50 | 31 | 19 | 450 |

---

## Task Registry

The INDEX_DRAFT.jsonl contains canonical task names for all 40 tasks. Per-task primary counts and teacher eligibility match between INDEX, PRIMARY, and TEACHER_LABEL_INDEX files. Task identity is consistent across all three files. See `tables/server_freeze/clean2000_suite_task_summary.csv` for per-task breakdown with INDEX-authoritative names.

---

## Historical Canary Classification

| Experiment | ep_summary Count | Success | Failure | scientifically_valid_rows |
|---|---|---|---|---|
| TMA (Object) | 171 | 34 | 137 | UNKNOWN (exceeds planned 162) |
| TMA_RT (Object) | 170 | 128 | 42 | UNKNOWN (exceeds planned 162) |
| UMA (Object) | 55 | 55 | 0 | UNKNOWN (CLEAN-only, incomplete) |
| SHUFFLED (Object) | 28 | 28 | 0 | UNKNOWN (CLEAN-only, incomplete) |
| UMA (SOTA) | 0 | N/A | N/A | 0 (FAILED_ENGINEERING_ATTEMPT) |
| SHUFFLED (SOTA) | 0 | N/A | N/A | 0 (FAILED_ENGINEERING_ATTEMPT) |

Counts exceeding planned=162 (TMA=171, TMA_RT=170) indicate retry/duplicate/canary artifacts.

---

## Runtime Code Freeze

**RUNTIME_CODE_FINGERPRINTED = PARTIAL. RUNTIME_CODE_FROZEN = NO.**

Corrected blob registry:

| File | base_blob | current_blob | diff_sha256 |
|---|---|---|---|
| attack_adapter.py | b05737c9 | efb42788 | 0e994bd6... |
| run_v2_vis_sc5_mlp_bridge.py | 4deb0019 | 634b0edf | 482c47d1... |
| v4_run_eval_openvla.py | 6c06d60b | e886a84f | 1db05b03... |
| run_sc5_cross_suite_clean.py | 6cf4ac4f | 6f4d2a0d | 0f5383a8... |

4 patch files saved. Previous registry had base/current swapped — corrected.

---

## Backup

BACKUP_NOT_EXECUTED. vla server /data/liuyu has 1.1T free. CLEAN2000 is ~574 MB.

---

## Gate Summary

```
SERVER_EXECUTION_QUIESCENT: PASS
CLEAN300_ARCHIVED: PASS
OBJECT_CONDITION_TOTALS: PASS
TRUE_T10_EMISSION_SPLIT: PASS
OBJECT_CROSS_CONDITION_KEY_MATCH: PASS
OBJECT_PROTOCOL_PROVENANCE: HOLD (preprocessing deviation upstream_tf_jpeg vs official_pil_lanczos)
CLEAN2000_AGGREGATE_CENSUS: PASS
CLEAN2000_LABEL_TIMING: HOLD (all 1350 labels are placeholder defaults)
RUNTIME_CODE_FREEZE: HOLD (blob direction corrected, patch files saved)
BACKUP: HOLD (not executed)
```

**HOLD_AUDIT_INCOMPLETE**

Sub-blocks: HOLD_OBJECT_PROVENANCE_RECONCILIATION, HOLD_CLEAN2000_RECONCILIATION, HOLD_RUNTIME_CODE_SNAPSHOT_MISSING, HOLD_BACKUP_NOT_SECURED.

---

CLEAN300 IS ARCHIVED AND SUPERSEDED.
DTY CLEAN2000 IS THE AUTHORITATIVE CROSS-SUITE CORPUS.
NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
EXPERIMENT EXECUTION REMAINS NOT AUTHORIZED.
