# LOTO 10-Fold Phase A Progress Report — 2026-06-28

**Trainer commit**: `3ff548f` (V3 — manifest-bound, protocol-enforced)
**Protocol**: `LOTO_10FOLD_PROTOCOL_FREEZE_V2.json`
**Builder**: `build_sc5_loto_fold_v3.py` (per-fold teacher calibration + label regeneration)

## Phase A Status

| Fold | Test Task | Val Task | Seeds | Best VL (min) | Status |
|------|-----------|----------|-------|---------------|--------|
| 00 | 8 chocolate_pudding | 6 butter | 3/3 | 1.09 | FROZEN (V2 corrected) |
| 01 | 9 orange_juice | 7 milk | 3/3 | 0.90 | FROZEN |
| 02 | 0 alphabet_soup | 8 chocolate_pudding | 3/3 | 1.87 | TRAINED |
| 03 | 1 cream_cheese | 9 orange_juice | 3/3 | 0.78 | TRAINED |
| 04 | 2 salad_dressing | 0 alphabet_soup | 3/3 | — | TRAINING |
| 05 | 3 bbq_sauce | 1 cream_cheese | 3/3 | — | TRAINING |
| 06 | 4 ketchup | 2 salad_dressing | 3/3 | — | TRAINING |
| 07 | 5 tomato_sauce | 3 bbq_sauce | 0/3 | — | PENDING |
| 08 | 6 butter | 4 ketchup | 0/3 | — | PENDING |
| 09 | 7 milk | 5 tomato_sauce | 0/3 | — | PENDING |

## Functional Gates (Train + Val replay)

| Fold | Train Emit | Val Emit | cp_p99 | Gate |
|------|-----------|----------|--------|------|
| 00 | 315-320/400 | Butter 44-45/50 | 0.91-0.94 | PASS |
| 01 | 295-306/400 | Milk 37/50 | 0.89-0.96 | PASS |
| 02-09 | — | — | — | PENDING |

## Held-out Status

- **Fold 00**: Chocolate V2 evaluated (coverage 0.933, FPR 0.70-0.95). Amendment-controlled.
- **Fold 01-09**: 0/9 held-out labels generated. 0/9 held-out evaluated.
- **Phase B**: HOLD until 27 checkpoints frozen.

## Cross-Suite

- **93 clean candidates** (31 spatial, 31 goal, 31 L10) inventoried
- All classified **Class B (CAUSAL_ONLY)** — missing target_x/y/z fields (pre-0280c85 bridge)
- 93/93 unique telemetry hashes, 0 duplicates
- Frozen: `sc5_cross_suite_clean_corpus_v1/phase8_causal_only/CLASSIFICATION_FREEZE.json`
- Target enrichment: unresolved pending offline canary

## Protocol Compliance

- Per-fold teacher calibration: ✓ (train-only, 400 privileged records)
- Per-fold label regeneration: ✓ (V2PrivilegedTeacher from fold-specific config)
- Per-fold train-only normalization: ✓
- Test task excluded from teacher labels: ✓ (all folds)
- Heldout labels not generated: ✓ (all folds Phase A)
- Runtime strict load: ✓ (all checkpoints)
- state_dict parity: 0.0 (all checkpoints)
- Manifest SHA binding: ✓ (trainer V3)
- Protocol SHA enforcement: ✓ (trainer V3)
- No threshold/modification: ✓

## Remaining

- Folds 04-06 training completion
- Folds 07-09 build + train
- Fold 02-09 functional replay
- Fold 02-09 Phase A freeze
- 27-checkpoint global freeze → Phase B
