# C2f Ablation — Gates and Interpretation

## Modality Ablations

| Name | 25D | Language | RGB | Context |
|---|---|---|---|---|
| A: 25D only | yes | no | no | yes |
| B: 25D + language | yes | yes | no | yes |
| C: 25D + RGB | yes | no | yes | yes |
| D: Full C2f | yes | yes | yes | yes |

## Success Gates (all must pass)

### Gate 1: Input chain
- RGB parity: steps == PNG frames, no blank images
- features_25d: len=25, no NaN/Inf
- task_language: non-empty for all episodes
- Context 108D: present

### Gate 2: Label nondegenerate
- primary_attackable positive rate > 0% (not all-zero)
- hazard positive rate > 0% (not all-zero)
- unsupported_or_abstain < 90% of all labels
- event_role distribution contains at least 2 non-trivial categories

### Gate 3: Ablation signal
- D_full (RGB+language) L10 primary-event recall > A_25d_only
- L10 recall > C2e3 baseline (45.6%) OR meaningful improvement over C2e3
- Overall FP <= 30%
- O/G/S recall does not collapse below C2e3 baseline

### Gate 4: No leakage
- Train/val/test split by episode, not window
- No episode appears in multiple splits
- No D7B2 outcome used in training

## Decision Tree

```
merge → hygiene PASS?
  NO  → FIX_COLLECTION, re-collect affected shards
  YES → audit PASS?
    NO  → LABEL_SIGNAL_WEAK, manual review
    YES → stats materialization → CLIP materialization → ablation training

ablation results:
  D > A on L10 primary recall?
    YES → C2F_SIGNAL_CONFIRMED_RGB_LANGUAGE_HELPS_L10
    NO  → C2F_V0_NO_CLEAR_VISUAL_LANGUAGE_GAIN
```

## Branch B: Label Signal Weak

If primary/hazard labels are all-zero or nearly all-zero after merge:

```text
C2F_CLEAN2000_COLLECTION = PASS_INPUT_CHAIN_BUT_LABEL_SIGNAL_WEAK
```

Paper text:
> C2f establishes the observation-rich data path but exposes that robust
> clean-only visual-language teacher labeling for L10 multi-object tasks
> remains unresolved.

This is not a failure. It's a boundary conclusion supporting D7 as the
main result and C2f as the next step.
