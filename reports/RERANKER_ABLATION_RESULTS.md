# Proprio + Visual Re-ranker Ablation Results

**Date**: 2026-05-31 | **Branch**: `exp/proprio-visual-reranker-ablation-20260530`

## Executive Summary

**Proprio + Visual re-ranker does NOT improve selectivity with current visual models.** Visual scores at ProprioNoStep trigger windows (contact phase) are uniformly near-zero and non-discriminative. The re-ranker concept is sound, but current visual models have the wrong temporal profile.

## Method

1. Run ProprioNoStep on all 50 Full10 clean episodes to find trigger windows
2. At each Proprio trigger window (contact-timed), evaluate visual model scores
3. Simulate: "only attack if Proprio triggered AND visual score > threshold"
4. Sweep thresholds and measure selectivity

## Results

### Proprio Trigger Windows

- 47/50 episodes have ProprioNoStep triggers
- Trigger steps: 102-259 (contact/transport/placement phase)
- 10 high, 10 robust, 18 medium, 9 low sensitivity windows

### Visual Scores at Proprio Windows

| Model | Score Range | Median | Discriminative? |
|-------|------------|--------|-----------------|
| VisualNoStep_v2 | 0.0000 - 0.010 | ~0.0000 | NO |
| VisualProprioNoStep_v2 | 0.0000 - 0.091 | ~0.0012 | NO |
| task_only | — | — | N/A |

### Selectivity Simulation

| Strategy | High Recall | Robust Specificity | Gap |
|----------|------------|-------------------|-----|
| Proprio-only (attack all) | 1.00 | 0.00 | — |
| Proprio + Visual re-rank (best th) | 1.00 | 0.00 | 0.00 |
| Proprio + VisualProprio re-rank (best th) | 0.10 | 0.90 | ~0 |

**Visual re-ranker achieves zero selectivity improvement.** At best threshold, it either attacks everything (like Proprio-only) or attacks nothing — no meaningful discrimination.

### Per-Task Re-ranker Effect (median threshold)

| Task | Class | Windows Kept | Oracle sus30 Fail |
|------|-------|-------------|-------------------|
| cream_cheese | high | 2/5 | 5/5 |
| tomato_sauce | high | 1/5 | 5/5 |
| ketchup | robust | 3/5 | 0/5 |
| salad_dressing | robust | 2/5 | 0/5 |

Re-ranker drops both vulnerable AND robust windows indiscriminately.

## Root Cause

Visual model scores have the wrong **temporal profile**:
1. Scores are highest at episode start (step 0-10) — "scene novelty"
2. Scores decay monotonically as episode progresses
3. By the time Proprio fires (step 100-200), scores have decayed to near-zero
4. Visual models encode "I see a new scene" not "this is a vulnerable contact moment"

This explains why standalone visual fires pre-contact (step 4): that's when scores are highest. It also explains why re-ranker fails: when Proprio provides the correct timing, visual scores are already zero.

## What We Learned

1. **Re-ranker concept is sound**: Proprio provides contact timing; visual must assess vulnerability AT those moments.
2. **Current visual models fail**: They encode scene novelty/appearance, not contact dynamics. Their temporal profile is wrong.
3. **Training fix needed**: To make re-ranker work, visual must be trained differently:
   - Use ONLY contact-phase frames (step 100+), not all frames
   - Use frame differences / motion features instead of static appearance
   - Or use temporal contrastive learning to detect "change at contact"

## Decision

**Re-ranker with current models: NOT RECOMMENDED for online pilot.**

ProprioNoStep remains the sole production detector. The re-ranker concept requires a fundamentally different visual training approach to produce contact-timed scores.

## Data

- Candidate windows: `/data/liuyu/outputs/milestone_2l_reranker_ablation_20260530/tables/reranker_candidate_windows.csv`
- Replay data: `/data/liuyu/outputs/milestone_2k_visual_detector_v2_training_20260530/tables/full10_replay_results.csv`
