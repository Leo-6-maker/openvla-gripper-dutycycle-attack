# C16 Closeout: SC5 Canonical Corpus Final Audit

**Date**: 2026-06-18
**Branch**: exp/l2-sc5-canonical-corpus-deepseek-20260618

## G-C0: Inventory Reconciliation — PASS

| Category | Count |
|----------|-------|
| Census total | 3,013 |
| Eligible (Tier A+B+C) | 688 |
| Excluded by tier | 2,325 |
| Clean after provenance | 354 |
| Attack contamination caught | 332 |
| Policy excluded (s3/s5) | 2 |
| Unique after dedup | 314 |
| INCLUDED | 142 |
| NONCONTIGUOUS_POLICY_TIMELINE | 20 |
| OOD_MULTI_STAGE_ABSTAIN | 104 |
| TIER_B_OBJECT_AMBIGUOUS | 37 |
| TIER_B_TARGET_AMBIGUOUS | 11 |
| **Total reconciled** | **314 = 142+20+104+37+11 ✓** |

## G-C1: Provenance — PASS

- Attack contamination: 332 detected, 0 in corpus
- Clean provenance validated per-episode via SC5SchemaAdapterV2
- All 142 included episodes verified clean

## G-C2: Schema — PASS

- Missing-to-zero: 0 (fail-closed, NaN for missing)
- Gripper semantic conflicts: 0
- action_dy duplication: 0
- 25D feature source audit recorded per step
- Field source types: direct, vector_extracted, causally_derived only

## G-C3: Teacher — PASS

- Train-only calibration on Tier A + field-validated Tier B
- Tier C (93 train episodes) excluded
- Full Teacher config frozen (22 fields, dataclasses.asdict)
- Calibration source file-content SHA256s recorded
- Guard=5, K=10 (frozen)
- SHA: per-build, file-content based

## G-C4: Object/Event Identity — PARTIAL

- Event segmenter implemented with 5-gate enforcement
- All 104 Tier C parents → OOD_MULTI_STAGE_ABSTAIN (conservative)
- Tier B: 47 field-validated, 37 object-ambiguous, 11 target-ambiguous
- TIER_B_SEMANTIC_BINDING_NOT_PROVEN (documented limitation)

## G-C5: Dedup/Split — PASS

- 33 duplicate groups, 64 episodes resolved
- 185 initial-state groups, 0 cross-split violations
- Train/val/held-out: 0 violations all pairs
- Split isolation audit JSON written per build

## G-C6: Strict Held-out — PASS

- Butter s8/s9/s11: 0 in train, 0 in calibration, 0 in normalization
- s5: AUDIT_ONLY (excluded from corpus)
- s3: SUPPLEMENTARY_ABSTAIN (excluded from corpus)
- Held-out rows: 310 (2 episodes), correctly isolated

## G-C7: Corpus Diversity — REPORTED

- Source roots: 6 milestone directories from frozen census
- Unique tasks: 38
- Corpus: 20,438 rows, 142 episodes, 113 SC5-valid
- Split: 112 train, 28 val (from 142; pre-build grouping 262/50)
- Train groups: 138, Val groups: 47

## Layer 1/2 Replay — PROVISIONAL 6/6 PASS

| Metric | Mean (3 seeds) | Target |
|--------|---------------|--------|
| Coverage | 0.873 | ≥0.80 ✅ |
| False-early | 0.025 | ≤0.10 ✅ |
| Post-release | 0.000 | ≤0.05 ✅ |
| Median abs error | 2.7 | ≤8 ✅ |
| K10 containment | 0.974 | ≥0.85 ✅ |
| No-corridor abstain | 0.954 | ≥0.90 ✅ |

## Layer 3 E2E — DETECTOR-TRIGGERED VIS POC + ATTRIBUTION PASS

| State | CLEAN (Phase 3) | VIS_SC5 | RAND_T10 |
|-------|:---:|:---:|:---:|
| Butter s0 | ✅ success | ❌ FAILED (8/10 OPEN) | ✅ success (0/10 OPEN) |
| Butter s2 | ✅ success | ❌ FAILED (9/10 OPEN) | ✅ success (0/10 OPEN) |

- VIS: targeted PGD attack causes task failure
- RAND: random L∞ perturbation at same ε does not
- 0 invalid feature steps, 0 privileged/manual input
- Checkpoint SHA: 66ec2d... , Dataset SHA: f942f4...
- VIS attack attribution established

## Status

```
C16_CLOSEOUT_PARTIAL_FREEZE
SC5_CANONICAL_CORPUS_BLOCKED_RECONCILIATION → PARTIAL_PASS
REMAINING: same-trajectory SC5 alignment audit (non-blocking)
```
