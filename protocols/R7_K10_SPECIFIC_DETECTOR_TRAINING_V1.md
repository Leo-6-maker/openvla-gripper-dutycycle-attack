# R7 K10-Specific Detector Training V1

Status: **FROZEN / AUTHORIZED FOR FIT FOLD-0 ONLY**  
Date: 2026-07-19  
Parent evidence: R7.1 K10 labeler V1.2.1 and R7.2.2 closure replay

## 1. Scientific question

Can a clean-only causal proprioceptive detector learn the frozen dense K10 opportunity-start target substantially better than the frozen Physics-tier checkpoints and simple causal close/dwell baselines?

The target is a clean gripper-critical opportunity start. It is not a VIS-vulnerability label and cannot by itself support an attack-effectiveness claim.

## 2. Scope and forbidden actions

Authorized:

- FIT identities only;
- Fold-0 train 600 / validation 200 from the sealed fold bundle;
- clean-only S1 Student inputs, Physics V2.1 auxiliary veto labels, and K10 V1.2.1 targets;
- at most two candidates defined below;
- one fixed seed and one fixed training recipe;
- one final validation evaluation per candidate at its train-only selected threshold.

Forbidden:

- FIT-DEV, CAL, CHECK, CS200 or attack/manual-outcome reads;
- simulator execution, command-OPEN, VIS, RAND or any intervention;
- policy-intent or visual candidates;
- changing the K10 Teacher, thresholds or label geometry;
- validation-driven threshold selection, epoch selection, architecture changes or loss changes;
- early stopping, hyperparameter search or rerunning with a new seed after seeing validation;
- overwriting any existing label, checkpoint, prediction or replay root.

## 3. Frozen data contract

Use the repository official V5 loader for Student inputs and candidate gating.

```text
Student input        = causal features_25d only
Student valid mask   = official loader valid_mask
Candidate gate       = official loader candidate_close
Utility target       = K10 V1.2.1 is_feasible_start
Release target       = Physics V2.1 release_imminent / known mask
Regrasp target       = Physics V2.1 regrasp_or_unstable / known mask
```

The K10 label root is target-only. For every identity, require exact step-count and candidate-close parity with the official loader. Any missing identity, field, non-finite value, schema mismatch or parity mismatch is fatal.

Training/validation identities must exactly match the sealed Fold-0 bundle:

```text
train identities      = 600
validation identities = 200
intersection          = 0
union                  = 800 FIT identities
```

Normalization is computed from the active training partition only. Each OOF fold computes its own normalization from its 480 training identities. The final model computes normalization from all 600 Fold-0 training identities.

## 4. Authorized candidates

Exactly two candidates are allowed.

### R7-S-LINEAR-25D

A causal per-step linear model:

```text
utility_logit = Linear(25, 1)
release_logit = Linear(25, 1)
regrasp_logit = Linear(25, 1)
```

No hidden layer, temporal convolution, recurrence or policy-intent input.

### R7-A-GRU-25D

The repository causal proprioceptive GRU branch with the frozen V5 hidden dimension:

```text
GRUCell(25, 128)
Linear(128, 1) utility head
Linear(128, 1) release head
Linear(128, 1) regrasp head
```

No policy-intent branch, visual branch, uncertainty head or future context.

No third candidate is allowed in R7.3 V1.

## 5. Frozen loss

All losses are calculated only on official `valid_mask AND candidate_close` steps.

### Utility loss

Use episode-balanced binary cross entropy on the dense K10 start mask.

For a positive episode containing both positive and negative candidate steps:

```text
sum of positive-step weights = 0.5
sum of negative-step weights = 0.5
```

For a no-corridor episode:

```text
sum of negative-step weights = 1.0
```

Each episode therefore contributes equal total utility mass independent of trajectory length or number of feasible starts.

### Auxiliary veto losses

Use masked binary cross entropy for release and regrasp targets on their known masks. Normalize each auxiliary loss by its valid step count.

Frozen total:

```text
total_loss = utility_loss + 0.3 * release_loss + 0.3 * regrasp_loss
```

No focal-loss exponent, margin loss, listwise loss, label smoothing or post-hoc reweighting is authorized.

## 6. Frozen optimization

```text
seed                  = 20260717
precision             = FP32
optimizer             = AdamW
learning_rate         = 1e-3
weight_decay          = 1e-5
epochs                = 10 exact
batching              = 8 episodes per optimizer step
clip_grad_norm        = 5.0
early stopping        = disabled
validation in training= disabled
```

At every epoch, deterministically shuffle the training identity list using `Random(seed + epoch)`. Accumulate the mean of eight episode-normalized losses before each optimizer step. The final incomplete batch is allowed.

All random libraries used by the implementation must be seeded. Deterministic CPU/GPU settings and environment metadata must be recorded.

## 7. Train-only OOF threshold selection

Build and seal a deterministic five-fold partition of the 600 training identities. Stratify by suite and episode-level K10 feasibility. Each identity appears in exactly one OOF validation fold.

For each candidate:

1. train five models on 480 identities each using the frozen recipe;
2. generate OOF predictions for the held-out 120 identities;
3. concatenate exactly one OOF prediction per training identity;
4. replay the frozen `V5OneShotScheduler` over the predeclared utility thresholds.

Threshold grid:

```text
0.05, 0.10, 0.15, ..., 0.95
```

Scheduler configuration remains frozen:

```text
minimum_candidate_dwell = 10
persistence             = 3 of 5
release threshold       = 0.5
regrasp threshold       = 0.5
one-shot                = enabled
```

An OOF threshold is eligible only when all conditions hold:

```text
K10 feasible-hit recall       >= 0.80
emit precision                >= 0.80
no-corridor abstention        >= 0.90
outside-rankable emit count   = 0
Teacher release/regrasp emits = 0
one-shot compliance           = 1.00
```

Select the highest eligible threshold. If no threshold is eligible, the candidate has no selected threshold and is `HOLD_OOF`; do not invent a fallback threshold.

The threshold sweep is diagnostic. No threshold may be selected from Fold-0 validation.

## 8. Final model and one-time validation

After OOF threshold selection, train one final model on all 600 training identities for exactly ten epochs. Evaluate the 200 validation identities once at the selected OOF threshold.

A full predeclared validation threshold curve may be written for diagnosis, but it cannot change the selected threshold or trigger a rerun.

Required validation outputs:

```text
K10 feasible-hit recall
emit precision
positive-episode coverage
no-corridor abstention
false-early emit count/rate
late/outside-corridor emit count/rate
outside-rankable emit count
Teacher release/regrasp emit count
one-shot compliance
mean/median delay from first feasible start
emission-conditional K10 containment
paired max-inside minus max-outside diagnostics
best-feasible rank and normalized percentile
per-suite and per-task confusion tables
```

## 9. Promotion gate

A candidate passes Fold-0 development only if, at the OOF-selected threshold, validation satisfies:

```text
K10 feasible-hit recall       >= 0.80
emit precision                >= 0.80
no-corridor abstention        >= 0.90
false-early rate              <= 0.05
outside-rankable emit count   = 0
Teacher release/regrasp emits = 0
one-shot compliance           = 1.00
```

It must also exceed the deployable first-valid-dwell-10 baseline by at least 0.30 in both recall and precision.

Representation diagnostics are secondary and cannot rescue a failed scheduler gate.

If both candidates pass, select by this frozen order:

1. higher emit precision;
2. higher feasible-hit recall;
3. higher no-corridor abstention;
4. lower false-early count;
5. prefer the linear candidate if still tied.

If neither passes, R7.3 ends in HOLD. No further proprioceptive tuning on the same validation split is authorized. A visual-modality proposal requires a new audit and protocol.

## 10. Required artifacts

For each OOF fold and final candidate root:

```text
PROTOCOL.json
SOURCE_BINDING.json
MODEL_CONTRACT.json
IDENTITY_MANIFEST.json
NORMALIZATION.json
checkpoint.pt
TRAIN_HISTORY.json
STEP_PREDICTIONS.jsonl
EPISODE_THRESHOLD_LEDGER.jsonl
THRESHOLD_METRICS.csv
REPRESENTATION_DIAGNOSTICS.csv
AUDIT.json
MANIFEST.json
SHA256SUMS
SHA256SUMS.sha256
```

A top-level handoff must include all full digests, exact commands, GPU/CPU environment, wall-clock time, identity counts, OOF threshold decision, one-time validation results and gate verdicts.

An independent read-only auditor must recompute population closure and all ledger-derived metrics from the sealed prediction records. Gate failure must exit nonzero.

## 11. Stop boundary

After the two candidates, OOF predictions, final checkpoints, validation ledgers and independent audit bundles are sealed, stop for review.

```text
R7_R3_FOLD0_TRAINING = AUTHORIZED
R7_R4_EXACT_PREFIX   = HOLD
R7_R5_ATTACK_CANARY  = HOLD
FIT_DEV/CAL/CHECK    = NOT READ
CS200                = NOT READ
```
