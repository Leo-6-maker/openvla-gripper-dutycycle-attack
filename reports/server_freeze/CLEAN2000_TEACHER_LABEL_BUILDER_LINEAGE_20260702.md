# CLEAN2000 Teacher Label Builder Lineage 2026-07-02

Status: CONSTANT_CANONICAL_TIMING_LABELS_WITH_REAL_SOURCE_AVAILABLE.

Canonical CLEAN2000 teacher labels use constant/default-like positive timing fields (`anchor=0`, `confidence=0.5`, `window=[0,10]`, empty event id) for eligible positives. Existing real teacher source labels were found for 2000/2000 episodes, but 0/2000 match the canonical constant fields exactly.

Read files:

- `CLEAN2000_BUILD_MANIFEST.json`
- `FREEZE_ENVELOPE.json`
- `CLEAN2000_SOURCE_INVENTORY.json`
- `CLEAN2000_TEACHER_CROSS_VALIDATION.json`
- `CLEAN2000_VALIDATION_REPORT.json`
- `CLEAN2000_TEACHER_LABEL_INDEX.jsonl`

Interpretation:

- CLEAN2000 is authoritative as a cross-suite census corpus.
- The canonical timing labels are not authorized for detector timing training.
- Existing real teacher sources can be audited, but this task did not generate new labels and did not train a detector.

Artifact: `tables/server_freeze/clean2000_teacher_source_availability.csv`.
