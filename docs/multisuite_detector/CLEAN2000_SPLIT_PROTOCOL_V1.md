# CLEAN2000 Split Protocol V1

## Core Principle

All windows from the same episode/parent MUST belong to the same split. No window-level randomization across splits.

## Split Types

### 1. In-Domain Episode-Grouped Split

```
Purpose: Engineering sanity, interpolation baseline
Method: Random split at episode level, stratified by suite
Train/Val/Test: 60/20/20 per suite
Constraint: Same episode not in multiple splits
Claim: Interpolation only, NOT generalization
```

### 2. Task-Grouped Split

```
Purpose: New-task generalization within known suites
Method: Split at task_id level
Train: 70% of tasks per suite
Val: 15% of tasks per suite
Test: 15% of tasks per suite
Constraint: All states/episodes/windows of a task in same split
```

### 3. Leave-One-Suite-Out (LOSO)

```
Purpose: Cross-suite generalization (primary evidence)
Folds: 4

Fold 1: Train {Object, Spatial, Goal} → Test LIBERO-10
Fold 2: Train {Object, Spatial, LIBERO-10} → Test Goal
Fold 3: Train {Object, Goal, LIBERO-10} → Test Spatial
Fold 4: Train {Spatial, Goal, LIBERO-10} → Test Object

Critical constraints:
- Test suite NEVER used for normalization statistics
- Test suite NEVER used for threshold selection
- Test suite NEVER used for checkpoint selection
- Test suite NEVER used for early stopping
- Test suite NEVER used for class weight estimation
- Normalization mean/std computed from 3 training suites ONLY
- Validation split drawn from training suites ONLY
```

## Leakage Prevention

The following must be REJECTED (fail-closed):
- Same episode_key in train and test
- Same parent_key in train and test
- Any test-suite row in normalization computation
- Any test-suite row in class weight computation
- Window from test episode in training set
- LOSO test suite appearing in any training-time statistic

## Reproducibility

All splits use deterministic seed (42 for primary, 123/456 for alternates).
Split manifest includes SHA256 of the split definition for provenance.
