# V2 SC5 Canonical Corpus Audit Report

**Date**: 2026-06-18
**Branch**: exp/l2-sc5-canonical-corpus-deepseek-20260618
**Base Census**: 45aad452 (repo_provenance=PASS)
**Builder SHA**: 7a3e9f8

## Corpus Overview

| Metric | Value |
|--------|-------|
| Total episodes | 299 |
| Train episodes | 295 |
| Strict held-out | 4 (Butter s8 x2, s9 x2) |
| Total rows | 46,384 |
| Source roots scanned | 6 |
| SC5-valid episodes | 161 |
| No-corridor negatives | 111 (pick_and_place) |
| OOD (drawer+stove) | 30 |

## Data Gates

### G-C0 Inventory Reconciliation — PARTIAL

| Category | Count |
|----------|-------|
| Census step_records total | 3,013 |
| Builder JSONL scanned | 662 |
| Builder clean-success found | 323 |
| After dedup (unique) | 299 |
| Entered canonical corpus | 299 |

**Gap**: Builder scanned 662/3013 (22%) of census step_records. The 6 source roots are the primary privileged artifact directories. Remaining 2,351 records include: 876 schema-missing, 1,244 clean-fail/unknown, 90 attack, 65 unknown suite, 50 incompatible, and additional candidates from directories not in builder roots. Full reconciliation requires scanning all census-identified candidate directories.

**Status**: G-C0_NOT_FULLY_RECONCILED — corpus covers key privileged artifact dirs

### G-C1 Provenance — PASS

- All 299 episodes have `clean_status=CLEAN` (validated during census)
- No attack/intervention episodes in corpus
- Attack contamination: 0

### G-C2 Schema — PASS

- All 25D fields present via direct/vector_extracted/causally_derived sources
- Missing-to-zero: 0 (fail-closed: invalid steps produce valid=False)
- Gripper semantic conflicts: 0 (raw<=0.5=CLOSE matches env>0=CLOSE)
- action_dy duplication: 0 (fixed in bdccf36)

### G-C3 Teacher — PASS

- Train-only calibration: 176 valid train paths
- Teacher config SHA: 54f50e4300ab7119
- guard=5, K=10 (frozen)
- Phase order validated: grasp_close->stable_grasp->first_lift->stable_carry->release_safe

### G-C4 Object/Event Identity — PARTIAL

- Event segmenter module exists (sc5_event_segmenter_v2.py) but NOT integrated into builder
- Multi-stage episodes flagged but not segmented at event level
- 17 "other" mechanism episodes are SC5-valid (LIVING_ROOM_SCENE* tasks — actually pick-and-place)
- "drawer" classification incorrectly includes "cabinet" tasks (put_X_on_top_of_cabinet = pick_and_place)

**Action**: Integrate event segmenter; fix mechanism classifier (remove "cabinet" from drawer keywords)

### G-C5 Dedup/Split — PASS

- 24 duplicate groups identified and resolved
- 272 initial-state groups, 0 cross-split violations
- Dedup priority: clean provenance > schema > privileges > manifest
- Parent events: not yet tracked (needs event segmenter integration)

### G-C6 Strict Held-out — PASS (with caveat)

- Butter s8, s9, s11: 0 in train/val/calibration
- s11: not in data (confirmed — not captured in Object100)
- s8: 2 held-out episodes (from milestone_2e2 and milestone_2e5_l10100)
- s9: 2 held-out episodes (same sources)
- **s5 ISSUE**: Butter s5 appears in train with SC5-valid=True. Census classifies s5 as CLEAN_FAIL. Per spec, s5 should be AUDIT_ONLY. This needs manual review.

### G-C7 Corpus Diversity

| Category | Count |
|----------|-------|
| Source roots | 6 |
| Unique tasks | 38 |
| SC5-valid pick_and_place | 141 |
| No-corridor pick_and_place | 111 |
| SC5-valid other (LIVING_ROOM) | 17 |
| SC5-valid stove (moka) | 3 |
| OOD drawer | 19 |
| OOD stove | 11 |
| Held-out (Butter) | 4 |

Corridor length: min=1, max=402, median=51
Anchor step: min=54, max=439, median=83

## Known Issues

1. **Butter s5 in train** — census classifies as CLEAN_FAIL, should be AUDIT_ONLY not train
2. **Builder only scans 6 dirs** — needs expansion to cover all 449 PRIMARY census candidates
3. **Event segmenter not integrated** — module exists but builder uses whole-episode SC5 only
4. **Mechanism classifier** — "cabinet" keyword incorrectly triggers "drawer" classification
5. **119 calibration paths excluded** — due to empty-string privileged fields in wait steps

## Current Status

```
SC5_CANONICAL_CORPUS_PARTIAL_PASS

Gates passing: G-C1, G-C2, G-C3, G-C5, G-C6
Gates partial: G-C0 (incomplete scan), G-C4 (segmenter pending)
Gates failing: none blocking

Remaining:
- Expand builder to all 449 PRIMARY census candidates
- Integrate event segmenter for multi-stage tasks
- Fix Butter s5 classification
- Fix mechanism classifier
- Full inventory reconciliation with 3013 census records
```

## Corpus Class Breakdown

| Class | Count | Description |
|-------|-------|-------------|
| PRIMARY_SC5_POSITIVE | 141 | Clean pick-and-place with valid SC5 corridor |
| NO_CORRIDOR_NEGATIVE | 111 | Clean pick-and-place, no valid K10 start |
| OOD_ABSTAIN | 30 | Drawer/stove mechanisms |
| CONDITIONAL_OTHER | 17 | LIVING_ROOM multi-object (SC5-valid but needs segment) |
