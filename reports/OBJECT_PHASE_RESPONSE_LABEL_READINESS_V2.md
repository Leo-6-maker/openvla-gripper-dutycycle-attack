# Object Phase Response Label Readiness V2

**Labels CSV**: `tables/object_phase_response_labels_v2.csv`
**Conflict CSV**: `tables/object_phase_response_labels_v2_conflicts.csv`
**Rows**: 31
**Train rows**: 22
**Verdict**: **PASS**

## Blocking Issues

- None.

## Label Status Counts

| Status | Count |
|---|---:|
| ignore | 9 |
| negative | 13 |
| positive | 9 |

## Source Counts

| Source | Count |
|---|---:|
| batch1 | 3 |
| batch3 | 18 |
| batch3b | 10 |

## Role Counts

| Role | Count |
|---|---:|
| complete_denominator | 1 |
| manual_merge_validated | 1 |
| raw_audit_group | 1 |
| standard | 28 |

## Boundaries

- Only positive/negative rows are train-eligible.
- manual_review, ignore, polluted, random-failed, denominator-failed, infra-failed, Xid/OOM, missing-trace, provenance-failed, schema-incomplete, and ambiguous rows must not enter train.
- This builder does not train detector v2.
