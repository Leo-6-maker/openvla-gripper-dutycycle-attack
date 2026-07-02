# Existing Evidence Freeze Report — 2026-07-02 (REVISION 4)

## Executive Summary

Object condition totals independently verified. Cross-condition selection-mask equality confirmed (141 selected, 21 excluded). **RAND_T10 correction: attack_applied=True for 141 emitted episodes (all succeed).** Actual preprocessing confirmed as `upstream_tf_jpeg` (not `official_pil_lanczos`). Epsilon/PGD/K/token remain UNKNOWN. Teacher labels are constant default-like values — CLEAN2000 is CENSUS_ONLY.

**Gate: HOLD_AUDIT_INCOMPLETE**

---

## Server Quiescence

SERVER_EXECUTION_QUIESCENT = PASS. PROCESS_STOPPED = PASS.

---

## CLEAN300: ARCHIVED AND SUPERSEDED

---

## Object Frozen Evidence

**Status: FROZEN_EMPIRICAL_RESULTS_UNDER_LEGACY_PROTOCOL_DRIFT**

See `reports/server_audit/OBJECT_FROZEN_EVIDENCE_RECONCILIATION_20260702.md` (Revision 4) for full detail.

### Verified
- OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS (all 6 conditions, corrected RAND_T10 accounting)
- TRUE_SELECTION_MASK_RECONCILIATION = PASS (141 selected, 21 excluded, selection masks equal)
- TRUE_T10_EMISSION_SPLIT = PASS (141 emitted, 21 no-emission)

### Attack Accounting (corrected)

| Condition | Total | Attack Applied | AA+mlp | AA+no_mlp | NoAA+mlp | NoAA+no_mlp | Success |
|---|---|---|---|---|---|---|---|
| CLEAN | 162 | 0 | 0 | 0 | 141 | 21 | 162 |
| RAND_T10 | 162 | 141 | 141 | 0 | 0 | 21 | 162 |
| RANDOM_TIME_V3 | 162 | 162 | 126 | 36 | 0 | 0 | 119 |
| EARLY_SHIFT_T10 | 141 | 141 | 99 | 42 | 0 | 0 | 98 |
| TRUE_T10 | 162 | 141 | 141 | 0 | 0 | 21 | 21 |
| COMMAND_OPEN_ORACLE_T10 | 141 | 141 | 141 | 0 | 0 | 0 | 0 |

### Protocol Status

| Parameter | Value |
|---|---|
| preprocessing | `upstream_tf_jpeg` (CONFIRMED from episode_summary.json) |
| jpeg_roundtrip | True (CONFIRMED) |
| epsilon | UNKNOWN |
| PGD steps | UNKNOWN |
| K | UNKNOWN |
| target_token | UNKNOWN |
| route | UNKNOWN |
| fallback | UNKNOWN |
| arm_gate | UNKNOWN |

Claimed frozen protocol (`official_pil_lanczos`, `epsilon=6/255`) does NOT match actual artifacts. The `upstream_tf_jpeg` preprocessing does NOT imply epsilon=2/255 — epsilon is independently unknown.

---

## CLEAN2000 Authority Census

**Status: CENSUS_ONLY. TIMING_TRAINING = FORBIDDEN.**

### Classification (2000 rows)

| Category | Count |
|---|---|
| PRIMARY_ELIGIBLE | 1043 |
| CLEAN_FAILURE_SAFETY | 307 |
| SUPPLEMENTARY_EVENT | 650 |

### Teacher Label Audit

| Metric | Value |
|---|---|
| Label index entries | 2000 (100% coverage) |
| teacher_label_valid=True | 1350 |
| teacher_label_valid=False | 650 |
| anchor_step = 0 (all valid) | 1350 (100% of valid) |
| anchor_step = -1 (all invalid) | 650 (100% of invalid) |
| confidence = 0.5 (all valid) | 1350 (100% of valid) |
| confidence = 0.0 (all invalid) | 650 (100% of invalid) |
| n_unique_anchors (valid) | 1 |
| n_unique_confidences (valid) | 1 |

All 1350 teacher_label_valid=True entries have identical anchor=0, confidence=0.5. These are **CONSTANT_DEFAULT_LIKE_TIMING_FIELDS** — no timing variation exists. Whether these are placeholder defaults, absolute-step-0 coordinates, or cropped-trace local coordinates cannot be determined without the generator script and source privileged records.

**CLEAN2000_TIMING_SIGNAL = NOT_USABLE_AS_CURRENTLY_STORED**

See `tables/server_freeze/clean2000_teacher_label_audit.csv` (2000 rows) and `tables/server_freeze/clean2000_episode_census.csv` (2000 rows).

### Abstention Semantics

The 650 SUPPLEMENTARY_EVENT entries have:
- `label_present_in_index = True` (record exists)
- `teacher_label_valid = False` (no positive anchor)
- `teacher_anchor_step = -1` (structured abstention marker)
- `teacher_confidence = 0.0`
- `teacher_invalid_reason = "teacher_ineligible"` (explicit reason)

These are valid structured abstentions for mechanism-ineligible tasks. To fully resolve the abstention schema, four independent fields would be needed: `label_record_present`, `record_schema_valid`, `positive_anchor_valid`, `explicit_abstention_valid`. Current JSONL uses two fields (`teacher_label_valid`, `teacher_invalid_reason`) which conflate these semantics.

### Usable For

| Use Case | Status |
|---|---|
| Suite/task census | YES |
| Clean success/failure | YES |
| Mechanism scope (pick-place vs other) | YES |
| Eligible/ineligible task taxonomy | YES |
| Timing detector training | NO (CONSTANT_DEFAULT_LIKE_TIMING_FIELDS) |
| Pooled detector training | FORBIDDEN |
| LOSO detector training | FORBIDDEN |

---

## Historical Canary Classification

All paper_usable=NO. SOTA UMA/SHUFFLED = FAILED_ENGINEERING_ATTEMPT (0 artifacts). Object-level TMA/UMA/SHUFFLED = EXPLORATORY_CANARY (scientifically_valid_rows=UNKNOWN where count > planned).

---

## Runtime Code Freeze

RUNTIME_CODE_FINGERPRINTED = PARTIAL. RUNTIME_CODE_FROZEN = NO.
4 dirty patch files saved. Blob direction corrected.
`tables/server_freeze/runtime_code_sha_registry.csv` has corrected base vs current blob SHAs.

---

## Backup

BACKUP_NOT_EXECUTED. vla:/data/liuyu has 1.1T free.

---

## Gate Summary

```
SERVER_EXECUTION_QUIESCENT: PASS
CLEAN300_ARCHIVED: PASS

OBJECT_RESULT_TOTALS: PASS
TRUE_SELECTION_MASK_RECONCILIATION: PASS
RAND_CONTROL_ATTACK_ACCOUNTING: CORRECTED (141/141 attacked, all succeed)
OBJECT_ACTUAL_PREPROCESSING: PASS (confirmed upstream_tf_jpeg)
OBJECT_FULL_PROTOCOL_PROVENANCE: HOLD (epsilon/PGD/K/token/route/fallback/arm_gate UNKNOWN)

CLEAN2000_CENSUS: PASS
CLEAN2000_TIMING_LABEL_PROVENANCE: HOLD (constant default-like, generator provenance unknown)

RUNTIME_CODE_FREEZE: HOLD
BACKUP: HOLD
```

**HOLD_AUDIT_INCOMPLETE**

---

CLEAN300 IS ARCHIVED AND SUPERSEDED.
DTY CLEAN2000 IS THE AUTHORITATIVE CROSS-SUITE CORPUS.
CLEAN2000 TIMING LABELS ARE CONSTANT DEFAULT-LIKE — TIMING TRAINING FORBIDDEN.
RAND_T10: 141/141 ATTACKED EPISODES ALL SUCCEED — PERTURBATION DIRECTION MATTERS.
NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
EXPERIMENT EXECUTION REMAINS NOT AUTHORIZED.
