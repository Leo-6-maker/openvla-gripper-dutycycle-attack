# Existing Evidence Freeze Report — 2026-07-02 (AMENDED 2026-07-02T20:00)

## Executive Summary

**Supersedes**: `reports/server_audit/SERVER_EXPERIMENT_STATUS_AUDIT_20260702.md` (which is now marked SUPERSEDED_BY this report for gate, classification, and evidence status).

Object condition totals verified from episode_summary.json. Per-episode master ledger committed (930 rows, 6 conditions). CLEAN2000 census committed (2000 rows, 3 mutually-exclusive categories). Runtime code fingerprinted (SHA256 + diff SHA + base blob SHA + file_size + mtime). Server quiescent. CLEAN300 archived.

**Gate: HOLD_AUDIT_INCOMPLETE** with sub-blocks: OBJECT_PROTOCOL_PROVENANCE, CLEAN2000_LABEL_SEMANTICS, RUNTIME_CODE_FREEZE, BACKUP.

---

## Server Quiescence

| Check | Verdict |
|---|---|
| OpenVLA processes | 0 — PASS |
| Bridge PIDs (6) | All DEAD — PASS |
| Outputs directory | Stable at 4096 bytes (empty) — PASS |
| Evidence writes (-15 min) | 0 files — PASS |
| Dual snapshot (T0=18:57, T1=19:01) | STABLE — PASS |
| Watcher (other project) | watch_understandingonly_b28.sh — HARMLESS |

**SERVER_EXECUTION_QUIESCENT = PASS**
**PROCESS_STOPPED = PASS**
**LAST_ARTIFACT_ATOMICITY = UNKNOWN** (cannot verify bridge completion at kill time)

---

## CLEAN300: ARCHIVED AND SUPERSEDED

CLEAN300_STATUS = ARCHIVED_SUPERSEDED_DATA. SCIENTIFIC_AUTHORITY = NONE.

---

## Object Frozen Evidence

**Status: FROZEN_REPORTED_RESULT_WITH_PROVENANCE_SEAL_PENDING**

All 6 condition totals match. 930-row master ledger committed. 141 emitted + 21 no-emission sets committed as per-episode CSVs.

- OBJECT_CONDITION_TOTAL_REAGGREGATION = PASS
- OBJECT_EPISODE_SET_RECONCILIATION = PASS
- OBJECT_PROTOCOL_PROVENANCE = HOLD (config/manifest/checkpoint SHA chain not yet sealed)

---

## CLEAN2000 Authority Census

**Status: CLEAN2000_FROZEN_WITH_LABEL_GAPS** (corrected framing below)

### Classification (Mutually Exclusive, Sum = 2000)

| Category | Count | % | Description |
|---|---|---|---|
| PRIMARY_ELIGIBLE | 1043 | 52.15% | task_success=True, teacher_eligible=True, mechanism_eligible=True |
| CLEAN_FAILURE_SAFETY | 307 | 15.35% | task_success=False, teacher_eligible=True (analyzable failure) |
| SUPPLEMENTARY_EVENT | 650 | 32.50% | teacher_eligible=False — task type out of scope for single-event pick-place mechanism |

### Label vs Mechanism Semantics (CORRECTED)

The 650 SUPPLEMENTARY_EVENT episodes have:
- **label_present_in_index: True** (all 2000 episodes have a teacher label index entry)
- **teacher_label_valid: False** — this is an EXPLICIT ABSTENTION, not a label gap
- **teacher_anchor_step: -1, teacher_confidence: 0.0** — structured abstention markers
- **teacher_invalid_reason: "" (empty)** — not "invalid", just inapplicable

These 650 episodes are mechanism-ineligible by task type (multi-object, multi-stage, articulated, push). The teacher correctly abstains rather than producing a forced label. This is NOT a "label gap" — it's "valid explicit abstention for mechanism-ineligible tasks."

The actual label gaps (episodes where a label SHOULD exist but does NOT) = 0.

### Per-Suite Breakdown

| Suite | Total | PRIMARY | CLEAN_FAILURE | SUPPLEMENTARY | Teacher Eligible | Labels Present |
|---|---|---|---|---|---|---|
| libero_spatial | 500 | 411 | 89 | 0 | 500 (100%) | 500 (100%) |
| libero_object | 500 | 367 | 133 | 0 | 500 (100%) | 500 (100%) |
| libero_goal | 500 | 234 | 66 | 200 | 300 (60%) | 500 (100%) |
| libero_10 | 500 | 31 | 19 | 450 | 50 (10%) | 500 (100%) |
| **TOTAL** | **2000** | **1043** | **307** | **650** | **1350 (67.5%)** | **2000 (100%)** |

### Supplementary Breakdown by Task Type

| Abstain Reason | Count |
|---|---|
| multi_object_two_items | 200 |
| articulated_task_no_pick_place | 100 |
| multi_object_two_different_targets | 100 |
| multi_stage_open_drawer_then_place | 50 |
| multi_stage_place_then_close | 50 |
| multi_stage_place_then_close_drawer | 50 |
| push_task_not_pick_place | 50 |
| multi_stage_turn_on_then_place | 50 |

---

## Historical Canary Classification (CORRECTED)

| Experiment | ep_summary Count | Success | Failure | Classification |
|---|---|---|---|---|
| TMA (Object) | 171 | 34 | 137 | EXPLORATORY_CANARY |
| TMA_RT (Object) | 170 | 128 | 42 | EXPLORATORY_CANARY |
| UMA (Object) | 55 | 55 | 0 | EXPLORATORY_CANARY (CLEAN-only) |
| SHUFFLED (Object) | 28 | 28 | 0 | EXPLORATORY_CANARY (CLEAN-only) |
| EARLY_SHIFT (Object) | 27 | 0 | 27 | EXPLORATORY_CANARY |
| RAND_LINF (Object) | 100 | 100 | 0 | EXPLORATORY_CANARY (CLEAN-only) |
| RANDOM_TIME (Object) | 162 | 146 | 16 | EXPLORATORY_CANARY |
| RANDOM_TIME_INVALID (Object) | 95 | 90 | 5 | EXPLORATORY_CANARY (INVALIDATED) |
| UMA (SOTA) | 0 | N/A | N/A | FAILED_ENGINEERING_ATTEMPT |
| SHUFFLED (SOTA) | 0 | N/A | N/A | FAILED_ENGINEERING_ATTEMPT |

All counts from episode_summary.json scan, not directory enumeration.

---

## Runtime Code Freeze

**RUNTIME_CODE_FINGERPRINTED = PARTIAL. RUNTIME_CODE_FROZEN = NO.**

4 dirty tracked files fingerprinted with: current_sha256, file_size, mtime, git_base_blob_sha, base_file_sha256, diff_sha256. See `tables/server_freeze/runtime_code_sha_registry.csv`.

What is MISSING for full freeze:
- Saved diff patch files (recoverable from diff_sha)
- Untracked file inventory with SHA
- 32 manifest SHAs listed individually (not just counted)
- Condition registry SHA (currently marked not_hashed)

---

## Backup Status

**BACKUP_NOT_EXECUTED. OBJECT_EVIDENCE_SIZE_NOT_MEASURED.**

vla server /data/liuyu has 1.1T free. CLEAN2000 is ~574 MB. Object evidence size needs `du -sb` measurement.

---

## Final Gate

```
SERVER_EXECUTION_QUIESCENT: PASS
CLEAN300_ARCHIVED: PASS
OBJECT_CONDITION_TOTALS: PASS
OBJECT_EPISODE_SET_RECONCILIATION: PASS
OBJECT_PROTOCOL_PROVENANCE: HOLD
CLEAN2000_AGGREGATE_CENSUS: PASS
CLEAN2000_ROW_LEVEL_AND_LABEL_SEMANTICS: RESOLVED (labels present, abstention is explicit)
RUNTIME_CODE_FREEZE: HOLD (fingerprinted but not frozen)
BACKUP: HOLD (not executed)
```

**HOLD_AUDIT_INCOMPLETE**

Sub-blocks: HOLD_OBJECT_PROVENANCE_RECONCILIATION, HOLD_RUNTIME_CODE_SNAPSHOT_MISSING, HOLD_BACKUP_NOT_SECURED.

Next step remains read-only revision of PR #47. No experiments may be launched.

---

CLEAN300 IS ARCHIVED AND SUPERSEDED.
DTY CLEAN2000 IS THE AUTHORITATIVE CROSS-SUITE CORPUS.
NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
EXPERIMENT EXECUTION REMAINS NOT AUTHORIZED.
