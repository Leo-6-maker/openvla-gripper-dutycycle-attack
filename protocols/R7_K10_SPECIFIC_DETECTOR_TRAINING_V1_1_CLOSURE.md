# R7 K10-Specific Detector Training V1.1 — Exact-Fold OOF Closure

Status: **FROZEN / AUTHORIZED FOR TRAIN600 OOF CLOSURE ONLY**  
Date: 2026-07-19  
Parent: `protocols/R7_K10_SPECIFIC_DETECTOR_TRAINING_V1.md`

## Purpose

Close the formal evidence defects in the submitted R7.3 OOF run without changing the scientific hypothesis, models, optimization recipe, labels, seed, scheduler or protected-data boundary.

This is a protocol-conformance rerun, not model tuning.

## Authorized candidates

Exactly:

```text
R7-S-LINEAR-25D
R7-A-GRU-25D
```

No architecture, input, loss, optimizer, scheduler, threshold-grid or seed change is permitted.

## Frozen execution

```text
population             = Fold-0 train identities only: 600
OOF validation folds   = 5 × exactly 120 identities
OOF training folds     = 5 × exactly 480 identities
fold overlap           = 0
OOF validation union   = exactly the 600 train identities
stratification         = suite + episode-level K10 feasibility
seed                   = 20260717
epochs                 = 10 exact
precision              = FP32
optimizer              = AdamW, lr 1e-3, wd 1e-5
batching               = 8 episode-normalized losses
normalization          = each 480-identity training fold only
early stopping         = disabled
thresholds             = 0.05, 0.10, ..., 0.95
scheduler              = frozen V5OneShotScheduler
```

## Mandatory OOF gates

A threshold is eligible only if every condition is true:

```text
K10 feasible-hit recall       >= 0.80
emit precision                >= 0.80
no-corridor abstention        >= 0.90
outside-rankable emit count   = 0
Teacher release/regrasp emits = 0
one-shot compliance           = 1.00
```

`outside_rankable emit` and `Teacher release/regrasp emit` must be computed from actual episode records. Constants or inferred zeros are forbidden.

Select the highest eligible threshold. If none is eligible, write `HOLD_OOF`; no fallback threshold is permitted.

## Forbidden reads and actions

```text
Fold-0 validation payload/features/labels = forbidden
FIT-DEV/CAL/CHECK                         = forbidden
CS200                                     = forbidden
final 600-identity model                  = forbidden when HOLD_OOF
visual or policy-intent model             = forbidden
new seed or hyperparameter                = forbidden
simulator / command-OPEN / VIS / RAND     = forbidden
```

Reading the sealed Fold-0 split manifest solely to identify the authorized train600 population is allowed. Loading any of the 200 validation episodes is forbidden.

## Required sealed evidence

For every fold:

```text
train identity list and count = 480
OOF identity list and count   = 120
normalization mean/std
checkpoint
10-epoch history
OOF step predictions
source binding
```

For every candidate:

```text
exact five-fold manifest
600-identity OOF prediction closure
600 × 19 episode-threshold ledger
19-row metric table with all six gates
HOLD/PASS decision
validation_payload_reads = 0
final_model_trained = false when HOLD_OOF
full source binding and manifest
SHA256SUMS + SHA256SUMS.sha256
```

An independent read-only auditor must recompute all 19 metric rows and gate decisions from the stored predictions and frozen K10/Physics targets, verify exact 480/120 fold closure, verify source seals, and exit nonzero on any mismatch.

## Stop boundary

After both candidate roots and independent audit bundles are sealed, stop.

```text
R7_R3_1_OOF_CLOSURE = AUTHORIZED
R7_R4_EXACT_PREFIX  = HOLD
R7_R5_ATTACK_CANARY = HOLD
VISUAL_TRAINING     = HOLD
```
