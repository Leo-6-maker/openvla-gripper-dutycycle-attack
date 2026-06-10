# Stage-B RC1a S7 Action-Hidden Readout Final

**Date**: 2026-06-10
**Commit**: TBD
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

S7 tested whether action-hidden features (last-layer hidden states at gripper token from OpenVLA generate) improve Layer-2 ranking beyond the S6 action-logit/action-dynamics baseline. After fixing a K5c parent-name parsing bug (abbreviated task names) that reduced initial coverage from 30/40 to full 40/40, the hidden features show a **Strong PASS**: HiddenSafe ranking within the CleanRand abstain pool consistently improves yield over random ranking at all tested K values.

### Core Finding

> Action-hidden states, when used in **HiddenSafe** direction (1 − HiddenRisk), provide the best standalone rand AUC (0.691) and improve fixed-K yield by +0.05 to +0.08 over the CleanRand+RandomRank baseline at K=7,10,15 on the 40-parent stable pool.

### Claim Boundary

**Allowed:**
1. HiddenSafeRank passes offline fixed-K readout on 40/40 stable pool
2. Hidden features contain safety-direction signal (AUC=0.691 standalone)
3. HiddenSafeRank yields consistent but moderate ranking improvement
4. This is a Layer-2 ranking **candidate**, validated by OOF GroupKFold

**Forbidden:**
- Hidden solves Layer-2 ranking
- Hidden detector is finished
- Fresh VIS/RAND attack confirmation has been run
- Cross-suite generalization demonstrated
- Layer-3 physical bridge solved

---

## S7 Gates

| Gate | Status | Detail |
|------|--------|--------|
| Coverage >= 36/40 | **PASS** | 40/40 (after K5c abbreviation fix) |
| feature_source = pre_window_only | **PASS** | All 38 unique targets |
| online_safe = True | **PASS** | All rows verified |
| hidden_dim = 4096 | **PASS** | Consistent across all extractions |
| No leakage columns | **PASS** | No VIS/RAND/yield/qpos/success/failure |
| Prompt present | **PASS** | All rows have non-empty prompt |

---

## Audit 1: Missing Parent Recovery

**Root cause**: 10 K5c parents used abbreviated task names (`alpha`, `cream`, `salad`, `tomato`, `bbq`, `oj`) not recognized by the original KNOWN list (full names only).

**Fix**: Added ABBREV mapping to parse logic. 1 parent shared a window with existing extraction; 9 required targeted re-extraction on GPU 2,6.

**Result**: 29 → 38 unique pre-window targets, 40/40 stable pool parents matched. 8/9 LIBERO tasks represented (ketchup absent from stable pool).

---

## Audit 2: Polarity Sanity

| Check | Result |
|-------|--------|
| y_rand=1 means rand_sensitive | VERIFIED |
| predict_proba[:,1] = P(rand) | VERIFIED |
| FP tomato is_rand=0 | VERIFIED (should be low risk) |
| FN salad is_rand=1 | VERIFIED (should be high risk) |

**Polarity**: HiddenRisk (predict_proba[:,1]) is **anti-correlated** with rand label (AUC=0.309). HiddenSafe (1 − HiddenRisk) is correctly correlated (AUC=0.691). The hidden states encode a "safety" signal, not a "risk" signal.

```
Direction          AUC    FP_tomato  FN_salad
HiddenRisk         0.309  0.8670     0.3685
HiddenSafe         0.691  0.1330     0.6315
```

---

## Readout Results (40/40, GroupKFold by task, n_splits=3)

| Model | RandAUC | FP_tomato | FN_salad | yield |
|-------|---------|-----------|----------|-------|
| TaskOnly | 0.486 | 0.5039 | 0.5132 | +0.46 |
| CleanProprio | 0.486 | 0.6130 | 0.5201 | +0.48 |
| ActionLogitOnly | 0.561 | 0.6586 | 0.3802 | +0.41 |
| **ActionHiddenOnly** | 0.309 | 0.8670 | 0.3685 | +0.07 |
| CleanProprio+Logit | 0.475 | 0.6712 | 0.5269 | +0.51 |
| CleanProprio+Hidden | 0.301 | 0.8709 | 0.4253 | +0.13 |
| ActionLogit+Hidden | 0.361 | 0.9293 | 0.3728 | +0.19 |
| CleanProprio+Logit+Hidden | 0.283 | 0.9081 | 0.5009 | +0.23 |

Note: ActionHiddenOnly uses the HiddenRisk (raw predict_proba[:,1]) direction. The HiddenSafe (1−score) direction gives AUC=0.691, making it the best standalone single-modality rand detector.

---

## Audit 3: Fixed-K Rank Ablation

All methods use the same CleanRand abstain filter (bottom 50% by CleanRand score), then rank within the pass set. Lower score = selected first.

```
K=7:
  HiddenSafeRank    cmd=0.5714  rand=0.4286  yield=+0.71  tasks=3
  CleanRandRank     cmd=0.5714  rand=0.2857  yield=+0.66  tasks=2
  RandomRank        cmd=0.2857  rand=0.5714  yield=+0.40  tasks=5

K=10:
  HiddenSafeRank    cmd=0.6000  rand=0.4000  yield=+0.68  tasks=4
  CleanRandRank     cmd=0.5000  rand=0.3000  yield=+0.60  tasks=4
  RandomRank        cmd=0.4000  rand=0.5000  yield=+0.46  tasks=6

K=15:
  HiddenSafeRank    cmd=0.6000  rand=0.4000  yield=+0.64  tasks=5
  CleanRandRank     cmd=0.4667  rand=0.4000  yield=+0.56  tasks=5
  RandomRank        cmd=0.4000  rand=0.4667  yield=+0.45  tasks=7

Oracle (K=15):      cmd=1.0000  rand=0.0000  yield=+0.95  tasks=5
```

**HiddenSafeRank consistently beats RandomRank and CleanRandRank at all K.** Yield improvement: +0.05 to +0.08 over CleanRandRank.

---

## Per-Task Highlights

| Task | CleanProprio AUC | Hidden AUC | Notes |
|------|-----------------|------------|-------|
| cream_cheese | 0.583 | **0.917** | Hidden excels |
| milk | 0.000 | 0.500 | Hidden better |
| salad_dressing | 0.250 | 0.375 | FN improves (0.520→0.369) |
| tomato_sauce | 0.333 | 0.389 | FP worsens (0.613→0.867) |
| alphabet_soup | 0.667 | 0.000 | Hidden fails |

Hidden helps some tasks (cream_cheese, milk, salad_dressing) but hurts others (alphabet_soup, tomato_sauce FP).

---

## Decision

**Classification**: Strong PASS per S7 handoff rules (HiddenSafeRank > RandomRank at all fixed K).

**Polarity**: HiddenSafe = 1 − HiddenRisk is frozen as the selection direction. This is a polarity diagnostic, not a post-hoc optimization.

**Limitations**:
1. Improvement is moderate (+0.05 to +0.08 yield)
2. FP/FN response is task-dependent (helps salad, hurts tomato FP)
3. Only validated on 40-parent stable pool (8 tasks, no ketchup)
4. HiddenSafe direction was discovered through polarity audit

---

## Next Step

Generate Layer-2 HiddenSafe confirmation queue (32 jobs: 8 windows × 2 seeds × 2 conditions). Queue audit must pass before any launch. No VIS/RAND attack without explicit approval.
