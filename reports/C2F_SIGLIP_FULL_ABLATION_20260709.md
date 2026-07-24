# C2f Full SigLIP A/B/C/D Ablation — 2026-07-09

**Supersedes**: `EXPERIMENT_STATUS_20260709_1315_TASK01_SIGLIP_REVIEW.md` (training results)
**GitHub commits**: `abb67a1` (merge+dataload) through latest
**Dataset**: OpenVLA-SigLIP 2000ep full NPZ (`f13f37d4`, 363,513 windows, 0 NaN)

## 1. Ablation Design

| Variant | Features | GPU | Runtime |
|---|---|---|---|
| **D (Full)** | 25D temporal + 2176d SigLIP visual + 4096d Llama-emb language + 108d context | 0 | 543s |
| **A (Baseline)** | 25D temporal + zero visual + zero language + 108d context | 1 | 551s |
| **C (Visual only)** | 25D temporal + 2176d SigLIP visual + zero language + 108d context | 2 | 732s |
| **B (Language only)** | 25D temporal + zero visual + 4096d Llama-emb language + 108d context | 3 | 511s |

All: batch=64, epochs=10, lr=1e-3, hidden=128, proj=128, seed=42. Same train/val/test split.

## 2. Test Final (Default Gate: tau_emit=0.33, tau_suppress=0.67)

| Variant | Recall | FP | F1 | L10_rec | L10_fp | Goal_rec | Obj_rec | Spat_rec | macro_rec | macro_fp |
|---|---|---|---|---|---|---|---|---|---|---|
| **D (Full)** | **97.7%** | **4.2%** | **0.954** | 99.4% | 1.2% | 98.5% | 77.7% | 98.7% | 93.6% | 8.3% |
| C (Visual) | 96.9% | 3.9% | 0.952 | 99.2% | 1.1% | 97.8% | 68.6% | 98.4% | 91.0% | 8.0% |
| A (Baseline) | 93.3% | 4.6% | 0.927 | 95.4% | 1.5% | 87.9% | 75.2% | 96.2% | 88.7% | 9.0% |
| B (Language) | 92.1% | 5.0% | 0.918 | 95.5% | 2.6% | 84.5% | 58.1% | 97.2% | 83.8% | 9.5% |

## 3. Best F1 Threshold (optimized per-variant)

| Variant | tau_emit | tau_suppress | Recall | FP | F1 | L10_rec | L10_fp |
|---|---|---|---|---|---|---|---|
| **D (Full)** | 0.25 | 0.50 | **97.2%** | **4.7%** | **0.958** | 99.5% | 1.4% |
| C (Visual) | 0.25 | 0.50 | 97.0% | 4.5% | 0.958 | 98.8% | 0.9% |
| A (Baseline) | 0.30 | 0.50 | 96.2% | 7.7% | 0.936 | 97.6% | 2.0% |
| B (Language) | 0.40 | 0.50 | 93.5% | 7.1% | 0.925 | 98.0% | 4.9% |

## 4. Best C2F Gate (constrained: low L10 FP, high L10 recall)

| Variant | tau_emit | Recall | FP | L10_rec | L10_fp |
|---|---|---|---|---|---|
| **D (Full)** | 0.25 | 98.7% | 6.8% | 99.9% | 2.0% |
| C (Visual) | 0.25 | 98.2% | 6.8% | 99.7% | 1.2% |
| A (Baseline) | 0.30 | 96.3% | 7.8% | 97.7% | 2.0% |
| B (Language) | 0.25 | 94.6% | 8.6% | 98.5% | 5.8% |

## 5. Visual-Language Gain Analysis

### D (Full) vs A (Baseline)
| Metric | A | D | Delta |
|---|---|---|---|
| Overall Recall | 93.3% | 97.7% | **+4.4pp** |
| Overall FP | 4.6% | 4.2% | -0.4pp |
| F1 | 0.927 | 0.954 | **+0.027** |
| L10 Recall | 95.4% | 99.4% | **+4.0pp** |
| Goal Recall | 87.9% | 98.5% | **+10.6pp** |
| Object Recall | 75.2% | 77.7% | +2.5pp |
| Spatial Recall | 96.2% | 98.7% | +2.5pp |
| Macro Recall | 88.7% | 93.6% | **+4.9pp** |
| BestF1 FP | 7.7% | 4.7% | **-3.0pp** |

### Visual contribution (C vs A)
- Visual alone adds **+3.6pp recall** (93.3→96.9) with **FP drop** (4.6→3.9)
- Visual is the **dominant modality**, matching 99% of D's F1 score

### Language contribution (D vs C) 
- Language adds **+0.8pp recall** (96.9→97.7) at cost of +0.3pp FP
- Language alone (B vs A): **-1.2pp recall** — language without visual is worse than baseline
- Language main benefit: Object recall +9.1pp (68.6→77.7) and Goal +0.7pp

### Language-only (B vs A)
- Language alone degrades overall recall (-1.2pp) and Object recall (-17.1pp)
- Language features without visual **confuse the detector** rather than help
- The 4096-dim Llama embedding may introduce noise without visual grounding

## 6. Spatial FP Problem

Spatial FP remains high across all variants (21-29% at default gate). This is:
- **Not a visual/language issue**: persists in A (24.1%)
- **A label issue**: Spatial primary rate = 74.8% of all windows → detector over-emits
- Best_f1 threshold reduces to ~12-21% but still elevated

## 7. Object Recall Analysis

| Variant | Object Test Recall | Object Val Recall |
|---|---|---|
| D (Full) | 77.7% | 86.5% |
| C (Visual) | 68.6% | 85.3% |
| A (Baseline) | 75.2% | 74.5% |
| B (Language) | 58.1% | 59.9% |

- Object test recall limited by **window-level primary rate = 13.7%** (label sparsity)
- Val recall higher than test due to split composition
- Visual boosts Object val recall from 74.5% to 85.3%
- Language alone hurts Object (58.1%)

## 8. Comparison with 200-ep Smoke

| Metric | 200ep D | Full D |
|---|---|---|
| Overall Recall | 98.7% | 97.7% |
| Overall FP | 8.9% | 4.2% |
| Object Recall (val) | 96.6% | 86.5% |
| L10 Recall | 99.4% | 99.4% |
| BestF1 FP | 4.8% | 4.7% |

Full 2000ep results are more conservative — Object recall dropped from 96.6% to 86.5% (val) because the full test split actually contains Object episodes (200ep smoke had Object in val only).

## 9. Val Final (for completeness)

| Variant | Recall | FP | F1 | Obj_rec | Obj_fp |
|---|---|---|---|---|---|
| D (Full) | 97.9% | 5.6% | 0.956 | 86.5% | 3.8% |
| C (Visual) | 97.7% | 5.6% | 0.955 | 85.3% | 3.0% |
| A (Baseline) | 92.8% | 5.9% | 0.928 | 74.5% | 5.1% |
| B (Language) | 92.1% | 6.1% | 0.923 | 59.9% | 3.6% |

## 10. Conclusions

1. **OpenVLA-SigLIP visual features provide significant gain**: +4.4pp recall, FP stable
2. **Visual is the dominant modality**: C matches 99.6% of D's F1
3. **Language adds marginal benefit** (+0.8pp recall) but **language alone is harmful** (-1.2pp vs A)
4. **Spatial FP is a label/detector issue**, not a feature issue
5. **Object recall bottleneck is label sparsity** (13.7% window primary), not features
6. **Full 2000ep results are robust**: FP drops from 8.9% (200ep) to 4.2% (2000ep)

## 11. Gate Status

```
C2F_SIGLIP_FULL_MATERIALIZATION      = PASS (2000ep, 363,513 win, 0 NaN)
C2F_SIGLIP_FULL_ABLATION             = PASS (A/B/C/D complete)
C2F_VISUAL_LANGUAGE_GAIN             = CONFIRMED (+4.4pp recall, visual dominant)
C2F_SPATIAL_FP                       = KNOWN_ISSUE (label-driven, not feature-driven)
C2F_OBJECT_RECALL_BOTTLENECK         = LABEL_SPARSITY (13.7% window primary)
C2F_ONLINE_CANARY                    = NOT_STARTED
```

## 12. Next Steps

```
P1: Online C2f canary (TRUE_T10_C2f vs RAND_T10_C2f on shared D7 parents)
P1: McNemar test on canary results
P2: v1.2 TeacherLabeler audit (relaxed matching FP check)
P2: Object label density improvement strategy
```

## 13. Boundaries

- D7 Table1 FROZEN — C2f does not modify
- Student input excludes privileged state
- Visual backbone = OpenVLA-SigLIP (NOT CLIP)
- Language embedding = Llama token embedding mean-pool (NOT full LLM forward)
- All training on merged Clean2000 v1.1 caveat labels
- Object window primary rate = 13.7% (label sparsity caveat)
