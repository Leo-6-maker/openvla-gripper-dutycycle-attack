# Stage-B RC1a 14cfabe — Multi-Head Detector Readout (72 pairs)

**Date**: 2026-06-08
**Data anchor**: d4a3827
**Code commit**: 14cfabe
**Pool**: 72 pairs (45 master + 6 smoke + 21 expansion)

## Readout configuration

- GroupKFold by task_state_seed (5 folds)
- LogisticRegression, class_weight=balanced
- Feature groups: TaskOnly, CleanNoTask*, Clean+Task*
- Metrics: AUROC, AUPRC, P@K, Enrichment@K, per-task AUROC

## Head A: cmd_specific (N=56, pos=20, neg=36)

| Feature Group | AUROC | AUPRC | P@5 | Enrich@5 |
|--------------|-------|-------|-----|----------|
| TaskOnly | **0.761** | 0.698 | 1.00 | 2.8x |
| CleanNoTaskNoTiming | 0.567 | 0.426 | 0.40 | 1.1x |
| CleanNoTaskWithTiming | 0.643 | 0.512 | 0.60 | 1.7x |
| Clean+Task | 0.639 | 0.573 | 0.60 | 1.7x |
| Clean+Task+Timing | 0.665 | 0.582 | 0.60 | 1.7x |

**Verdict**: TASK-BIASED. TaskOnly dominates. Clean proprio features (AUROC=0.567) barely above chance.
Per-task AUROC: bbq_sauce=0.10, salad_dressing=0.00. tomato_sauce has 7/20 cmd positives (35%).
**Not yet a reliable clean-feature online detector.**

## Head B: vis_specific_phys strict (N=39, pos=3, neg=36)

| Feature Group | AUROC | P@3 | Enrich@3 |
|--------------|-------|-----|----------|
| TaskOnly | 0.787 | 0.33 | 4.3x |
| CleanNoTaskNoTiming | 0.380 | 0.00 | 0.0x |
| CleanNoTaskWithTiming | 0.731 | 0.33 | 4.3x |

**Verdict**: UNDER-POWERED. Only 3 strict phys positives (after excluding shared_qpos, rand_phys_confound, cmd_specific overlaps).
Cannot draw reliable conclusions. Need more strict phys positives before evaluating visual features.

## Head C: abstain_any (N=45, pos=9, neg=36)

| Feature Group | AUROC | AUPRC | P@5 | Enrich@5 |
|--------------|-------|-------|-----|----------|
| TaskOnly | 0.727 | 0.357 | 0.20 | 1.0x |
| **CleanNoTaskWithTiming** | **0.917** | **0.641** | **0.80** | **4.0x** |
| Clean+Task+Timing | 0.914 | 0.625 | 0.60 | 3.0x |
| CleanNoTaskNoTiming | 0.667 | 0.288 | 0.20 | 1.0x |

**Verdict**: STRONGEST CLEAN-FEATURE SIGNAL. CleanNoTaskWithTiming (AUROC=0.917, P@5=0.80) significantly
outperforms TaskOnly (AUROC=0.727, P@5=0.20). Per-task AUROC: alphabet_soup=0.923, cream_cheese=0.800.
Timing features (window_center, relative timing) are critical — without them AUROC drops to 0.667.

**This is the best current clean-feature predictor. Random-sensitive / confounded windows can be identified from clean proprioceptive + timing features without task identity.**

## Key conclusions

1. **cmd_specific** is dominated by task bias. Clean proprio features add little beyond "which task am I in?" Tomato_sauce has 35% of cmd positives.

2. **vis_specific_phys strict** is underpowered (3 positives). Cannot assess clean or visual features for this head.

3. **abstain_any** is the strongest and most trustworthy clean-feature signal. It can identify random-sensitive windows without relying on task identity. This validates the multi-head selector design: learn to abstain first, then predict vulnerability.

## Next step

Visual sidecar (DINOv2) to test whether frozen visual features can:
- Reduce task bias in cmd_specific prediction
- Help identify phys_strict windows (once more are collected)
- Maintain or improve abstain_any performance

**Do NOT**: claim detector solved, expand phys without visual evidence, merge abstain into negative.
