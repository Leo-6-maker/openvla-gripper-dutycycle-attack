# CLEAN2000 Multi-Suite Detector Experiment Spec V1

**Status**: CODE_PREPARATION_ONLY | NO_FORMAL_TRAINING
**Branch**: `feature/multisuite-clean2000-detector-prep-v1`
**Base**: `ace18762`

## 1. Objective

Compare detector generalization across four LIBERO suites using a frozen Object-only baseline, a balanced pooled multi-suite detector, and Leave-One-Suite-Out (LOSO) detectors. All use identical model architecture, feature contract, normalization semantics, and runtime state machine as the current SC5 Object detector.

## 2. Detector Variants

| ID | Name | Training Data | Claim Type |
|----|------|--------------|------------|
| A | Object-only Frozen | Object/SC5 only (frozen checkpoint) | Zero-shot cross-suite |
| B | Balanced Pooled | All 4 suites, suite-balanced sampling | Supervised interpolation |
| C1-C4 | LOSO × 4 | 3 suites train, 1 held-out test | Cross-suite generalization |
| D | Fixed Normalized-Time | None (heuristic) | Simple baseline |
| E | Rule-based Proprio | None (heuristic) | Simple baseline |
| F | Logistic/Linear | Same as B | Linear baseline |
| G | Privileged Teacher | Oracle labels | Upper bound |

## 3. Architecture Contract

All variants A-C must maintain byte-identical:
- 25D feature order (extracted from current SC5 detector code)
- Feature normalization mean/std
- MLP architecture (layers, hidden dims, activations)
- State machine (corridor/phase/release heads, tau thresholds)
- Guard step count, K window
- One-shot / reset semantics

## 4. Data Pipeline

```
CLEAN2000 source corpus
  → build_clean2000_index.py
  → validate_clean2000_corpus.py
  → build_teacher_label_index.py
  → build_primary_and_safety_sets.py
  → build_detector_splits.py
  → validate_detector_splits.py
  → extract_frozen_feature_contract.py
  → train_detector.py / evaluate_detector.py
```

## 5. Forbidden Actions

- NO formal training until data freeze complete
- NO reading TRUE_T10 or CLEAN1500 aggregate results
- NO GPU tasks
- NO modifying Object-only checkpoint
- NO modifying current VIS detector or branch
- NO merging to main

## 6. Output State

```
MULTISUITE_DETECTOR_CODE_PREP_PASS
FORMAL_DATA_FREEZE_PENDING
FORMAL_TRAINING_NOT_STARTED
CURRENT_VIS_DETECTOR_UNCHANGED
```
