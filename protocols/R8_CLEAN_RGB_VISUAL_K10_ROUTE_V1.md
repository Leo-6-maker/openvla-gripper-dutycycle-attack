# R8 Clean-RGB Visual K10 Route V1

Status: **FROZEN DESIGN / R8.0 SOURCE AUDIT AUTHORIZED ONLY**  
Date: 2026-07-19  
Parent evidence: R7.1 K10 V1.2.1, R7.2.2 frozen Physics transfer replay, R7.3.1 terminal proprio-only HOLD

## 1. Scientific question

Can causal visual evidence from an already collected clean rollout identify the frozen K10 clean gripper-critical opportunity start better than the failed 25D proprio-only candidates?

The target remains a clean opportunity-start proxy. It is not a VIS-vulnerability label, command-OPEN effect label, or attack-success ground truth.

## 2. Existing-data boundary

### Official sources

- Official V3 FIT S1 / 25D Student stream, states 0-19 only;
- Physics Teacher V2.1 release/regrasp auxiliary labels;
- K10 opportunity label V1.2.1;
- sealed Fold-0 manifest: train 600 / validation 200.

### Candidate RGB source

The only discovered per-step RGB source is the parallel C2F clean root `clean2000_obs_clean_36712cc`. It has direct PNG frames and metadata/path alignment, but it is not yet bound to the Official V3 rollout campaign.

Identity-name equality is insufficient. C2F RGB may be used only after R8.0 proves that every used frame represents the same local clean trajectory as the Official V3 Student/Teacher/K10 step to which it would be joined.

### Forbidden data

- FIT-DEV, CAL, CHECK, CS200 or any state 20-49 semantic/image read;
- attack, command-OPEN, VIS, RAND or manual outcome data;
- C2F teacher-like fields as Student inputs;
- future frames, privileged masks, object pose, target pose, contact or task-success fields;
- project-data fine-tuning of the visual encoder;
- validation-driven architecture, preprocessing, epoch or threshold changes.

## 3. R8.0 — read-only RGB source binding audit

R8.0 is the only currently authorized action. It may inventory FIT paths and read the shared clean numeric streams required for binding. It may decode a deterministic train600 image sample for byte/shape checks. It may not train or extract model embeddings.

### 3.1 Required identity closure

```text
Official FIT identities     = 800
C2F FIT identities          = 800
canonical-key intersection  = 800
missing / extra / duplicate = 0
Fold-0 train / validation   = 600 / 200
```

No identity may be silently dropped or replaced.

### 3.2 Required campaign and camera binding

For every FIT identity, bind and report when available:

- task language/prompt hash;
- OpenVLA model, checkpoint and processor hashes;
- simulator/task/BDDL identity;
- initial reset/state checksum and seed;
- camera name, resolution, intrinsics/extrinsics and renderer configuration;
- collector/source commits;
- episode step count.

A missing field is not automatically a mismatch, but the final auditor must state whether the remaining shared evidence is sufficient for label transfer. Camera configuration must be explicitly bound; otherwise R8 remains HOLD.

### 3.3 Step-local trajectory equivalence

The visual detector is limited to a seven-frame causal receptive field. For every Official V3 rankable candidate step `t`, and every K10 start, audit the local prefix `[t-6, ..., t]` after left clipping at step 0.

Where shared fields exist, require:

```text
action vector max abs error       <= 1e-7
gripper command max abs error     <= 1e-7
qpos / gripper width max error    <= 1e-6
EEF pose/velocity max error       <= 1e-6
step index and episode length     exact
```

A local visual sample is usable only when all required shared numeric fields and all seven frame pointers are present and aligned. Because the folds are frozen, the route passes only if all 800 identities have complete usable coverage for every rankable step and every K10 start. No partial-identity training set is authorized.

If the two campaigns do not expose enough shared numeric evidence to establish local equivalence, return `HOLD_UNBINDABLE`; do not infer equivalence from identity names or initial states alone.

### 3.4 Frame integrity

For every used frame:

- unique canonical `(identity, step)` mapping;
- file exists and decodes as RGB;
- fixed dimensions within the bound camera contract;
- content SHA-256 recorded;
- no duplicate step pointer;
- no future-frame lookup;
- every K10 start has complete `t-6:t` context.

### 3.5 Leakage firewall

The materialized binding manifest may contain only:

```text
canonical identity
step
RGB relative path and content SHA-256
camera/preprocess contract pointer
source/campaign binding pointer
```

C2F fields such as `teacher_hazard`, `teacher_phase`, `teacher_primary_attackable`, privileged state, success or attackability must never enter the Student root.

### 3.6 R8.0 pass gate

```text
identity closure                     = 800/800
rankable-step local parity           = 100%
K10-start seven-frame coverage       = 100%
frame decode/hash closure            = 100%
camera contract                      = PASS
Teacher/privileged leakage           = 0
protected semantic/image reads       = 0
source mutation                      = 0
```

Failure of any item keeps visual training HOLD.

## 4. R8.1 — frozen visual input materialization (conditional)

R8.1 is not authorized until R8.0 passes.

Use the exact frozen OpenVLA image tower already bound to the policy checkpoint. Extract image-only, pre-language spatial features; do not consume policy logits, action tokens, language embeddings or Teacher fields.

### 4.1 Frozen encoder and preprocessing

- encoder weights, processor and config bound by full hashes;
- encoder in evaluation mode, gradients disabled;
- deterministic resize/crop/normalization matching the bound processor;
- no augmentation;
- current frame only at extraction time;
- output stored as FP16, downstream accumulation/training in FP32.

### 4.2 Spatial representation

Preserve coarse geometry rather than only a global pooled vector:

```text
frozen patch-token map
-> deterministic adaptive average pool to 4 x 4 cells
-> 16 spatial tokens per step
-> optional frozen global token stored separately
```

The encoder output dimension is discovered from the exact bound tower and then frozen in `VISUAL_ENCODER_CONTRACT.json`; it is not a tunable hyperparameter.

### 4.3 Materialization order

1. Materialize train600 only.
2. Complete five-fold OOF and select a threshold from train600 only.
3. Only if an OOF threshold passes, materialize the frozen validation200 images and run one final evaluation.

The validation image pipeline cannot be inspected or modified after OOF results are observed.

## 5. R8.2 — exactly two visual candidates (conditional)

No third candidate, encoder search, language branch or policy-intent branch is allowed.

### R8-C-VIS7

Visual-only causal detector:

```text
16 spatial tokens -> LayerNorm -> Linear(D,64)
learned one-query spatial attention -> 64D per frame
seven-frame causal TCN: kernel 3, dilations 1 and 2, hidden 128
utility / release / regrasp heads
```

The two causal convolutions have an exact receptive field of seven frames. No recurrent state outside the seven-frame window is allowed.

### R8-D-PROP25-VIS7

Uses the identical frozen visual branch plus:

```text
25D proprio -> GRUCell(25,128)
concat(proprio128, visual128)
Linear(256,128) + Tanh
utility / release / regrasp heads
```

The proprio branch is causal. Visual features, K10 targets and auxiliary Teacher labels remain separate streams.

## 6. Frozen training recipe

```text
Fold-0 train / validation = 600 / 200
OOF folds                 = five exact 480 / 120
seed                      = 20260719
precision                 = FP32
encoder                   = frozen
optimizer                 = AdamW
learning rate             = 1e-3
weight decay              = 1e-5
epochs                    = 10 exact
batching                  = 8 episode-normalized losses
gradient clip             = 5.0
early stopping            = disabled
threshold grid            = 0.05, 0.10, ..., 0.95
scheduler                 = frozen V5 one-shot scheduler
```

Use the R7 dense K10 episode-balanced utility loss plus `0.3 * release BCE + 0.3 * regrasp BCE`. Normalization and any trainable projection statistics are fold-train-only.

## 7. Metrics and promotion gates

### 7.1 Primary OOF and validation gates

At the selected train-only threshold:

```text
K10 feasible-hit recall             >= 0.80
emit precision                      >= 0.80
no-corridor abstention              >= 0.90
emit outside valid/candidate gate   = 0
Teacher release/regrasp emit count  = 0
one-shot compliance                 = 1.00
```

`emit outside valid/candidate gate` is distinct from an emit outside the K10 target. The latter is already counted by emit precision and must not be double-labelled as `outside_rankable`.

No eligible OOF threshold means `HOLD_OOF`, with no final model and no validation image read.

### 7.2 Predeclared visual-specific controls

After candidate/threshold freeze, report the same model under:

- aligned RGB;
- causal lag-20 RGB from the same episode;
- fixed within-task episode-deranged RGB sequence;
- visual-zero ablation for R8-D;
- frozen R7-A proprio-only result as historical baseline.

These controls are diagnostic and may not alter the selected threshold.

For a visual-timing claim, aligned RGB must exceed the better of lag-20 and deranged controls by at least 0.20 in both K10 recall and emit precision. Also report paired max-inside minus max-outside score, best-feasible rank and best-step-in-corridor rate on the same positive episodes.

## 8. Required artifacts

### Source audit root

```text
PROTOCOL.json
SOURCE_ROOTS.json
IDENTITY_BINDING.jsonl
LOCAL_TRAJECTORY_PARITY.jsonl
FRAME_BINDING.jsonl
CAMERA_CONTRACT.json
AUDIT.json
MANIFEST.json
SHA256SUMS
SHA256SUMS.sha256
```

### Visual input root

```text
SOURCE_BINDING.json
VISUAL_ENCODER_CONTRACT.json
PREPROCESSING.json
IDENTITY_MANIFEST.json
FRAME_HASHES.jsonl
per-episode visual token files
AUDIT.json
MANIFEST.json
SHA256SUMS
SHA256SUMS.sha256
```

### Candidate root

Store exact OOF fold identities, normalization, checkpoints, predictions, all 19 threshold ledgers, all metrics and an independent read-only audit bundle. Existing R7/R8 roots are immutable.

## 9. Stop boundary

Current authorization:

```text
R8_0_RGB_SOURCE_BINDING_AUDIT = AUTHORIZED — READ ONLY
R8_1_VISUAL_MATERIALIZATION   = HOLD
R8_2_VISUAL_TRAINING          = HOLD
R8_3_VALIDATION               = HOLD
R7_4_EXACT_PREFIX             = HOLD
R7_5_ATTACK_CANARY            = HOLD
```

Stop after R8.0 produces a sealed source-binding root and independent audit bundle.