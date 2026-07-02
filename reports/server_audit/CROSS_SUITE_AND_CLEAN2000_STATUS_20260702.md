# Cross-Suite and CLEAN2000 Status — 2026-07-02

> **SUPERSEDED_BY: `reports/server_freeze/EXISTING_EVIDENCE_FREEZE_REPORT_20260702.md` (Revision 4)**
>
> This file's estimated statistics, "conditionally ready" status, teacher label claims, and detector training readiness are superseded. See the freeze report for:
> - 2000-row census with exact mutually-exclusive classifications
> - Teacher label audit (all 1350 timing labels are constant default-like values)
> - CLEAN2000 status downgraded to CENSUS_ONLY (not usable for timing detector training)
> - CLEAN300 archived and superseded
>
> Retained below: server paths, file inventory, and SHA256SUMS verification (still valid).

---

## CLEAN300 Status — ARCHIVED AND SUPERSEDED

CLEAN300_STATUS = ARCHIVED_SUPERSEDED_DATA. SCIENTIFIC_AUTHORITY = NONE.

---

## CLEAN2000 File Inventory (verified 2026-07-02)

| File | Size | Lines | SHA Verified |
|---|---|---|---|
| CLEAN2000_INDEX_DRAFT.jsonl | 3.5 MB | 2000 | YES (via SHA256SUMS.txt) |
| CLEAN2000_ATTEMPT_LEDGER.jsonl | 509 KB | 2000 | YES |
| CLEAN2000_PRIMARY.jsonl | 340 KB | 1043 | YES |
| CLEAN2000_TEACHER_LABEL_INDEX.jsonl | 929 KB | 2000 | YES |
| CLEAN2000_TEACHER_CROSS_VALIDATION.json | 24 KB | — | YES |
| CLEAN2000_FEATURES_25D_ALL_STEPS.csv | 287 MB | — | YES |
| CLEAN2000_FEATURES_25D_VALID_ONLY.csv | 287 MB | — | YES |
| CLEAN2000_FEATURE_GOLDEN_PARITY.json | 1 KB | — | YES |
| SHA256SUMS.txt | 1.4 KB | — | YES |
| FREEZE_ENVELOPE.json | 4.6 KB | — | YES |

---

## Current CLEAN2000 Status (from Revision 4 freeze report)

- **Classification**: 2000 rows, 3 mutually-exclusive categories (1043 PRIMARY + 307 CLEAN_FAILURE + 650 SUPPLEMENTARY)
- **Teacher labels**: All 1350 "valid" labels are constant default-like (anchor=0, confidence=0.5, window=[0-10])
- **Timing training**: NOT AUTHORIZED (no usable timing signal)
- **Census use**: AVAILABLE (success/failure, mechanism scope, per-suite statistics)
- **Actual label gaps**: 0 (all 2000 have index entries; 650 are explicit abstentions)

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
