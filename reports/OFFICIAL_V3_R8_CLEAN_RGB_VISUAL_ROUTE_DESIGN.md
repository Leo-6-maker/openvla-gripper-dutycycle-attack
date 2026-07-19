# Official V3 R8 Clean-RGB Visual Route Design

Date: 2026-07-19  
Protocol: `protocols/R8_CLEAN_RGB_VISUAL_K10_ROUTE_V1.md`

## Executive decision

Existing clean RGB can potentially be reused, but not by joining on task/state names alone.

The Official V3 clean source has complete FIT policy/proprio/privileged telemetry but no RGB for 800/800 FIT identities. The parallel C2F source has per-step PNGs and 800/800 FIT path alignment, but belongs to a different collector/source campaign. The visual route therefore begins with a trajectory-and-camera binding audit. Visual embedding extraction and training remain HOLD until that audit proves local equivalence.

## Existing assets

| Asset | Coverage | Use in R8 |
|---|---:|---|
| Official V3 S1 25D | 800 FIT | causal proprio and candidate gate |
| Physics V2.1 | 800 FIT | release/regrasp auxiliary supervision only |
| K10 V1.2.1 | 800 FIT; 109 feasible | target only |
| Fold-0 manifest | 600/200 | frozen development split |
| Official V3 RGB | 0/800 | unavailable |
| C2F raw PNG | 2000 identities; 800 FIT path-aligned | candidate source, unbound |
| old `siglip_full_final` outputs | no reliable identity mapping | not used |

C2F metadata contains teacher-like fields. R8 uses a whitelist loader that emits only identity, step, frame path/hash and camera/source pointers. Teacher-like metadata is never copied into the Student input root.

## Why local trajectory binding is necessary

K10 labels describe the Official V3 clean trajectory. A frame from another rollout is a valid input for that label only when it depicts the same local robot/object state and camera at the same step.

A shared state ID establishes an initial condition, not a shared trajectory. Even one different policy action can move the gripper/object and invalidate the visual/K10 pairing. R8 therefore compares shared action and proprio fields over the full seven-step receptive field for every candidate step and K10 start.

The seven-frame architecture deliberately limits the required equivalence scope. It avoids an unlimited recurrent visual history, so a divergence far outside the used local prefix cannot silently contaminate the visual label.

## Visual representation choice

R8 reuses the frozen OpenVLA image tower already available with the policy checkpoint. This choice avoids:

- adding a second pretrained-model selection degree of freedom;
- downloading or fine-tuning a new encoder;
- changing the visual domain from the policy's own representation;
- unnecessary online compute if features can later be shared with the policy forward pass.

Only image-only, pre-language spatial features are used. No task-language embedding, policy token, action logit or hidden policy state is consumed.

A 4x4 pooled spatial token grid is retained rather than a single global vector. The K10 target depends on gripper/object geometry and contact-stage appearance, so spatial information is likely more important than scene-level semantics.

Expected FIT storage after extraction is modest. Depending on the bound image-tower width, sixteen FP16 tokens per frame should require roughly 2-4 GB for the FIT corpus, plus manifests and hashes. The audit reports the exact frame count and size before materialization.

## Candidate design

Exactly two candidates are proposed.

### R8-C-VIS7

A visual-only test of whether clean RGB contains K10 timing information:

```text
4x4 spatial tokens
-> 64D spatial attention pool
-> seven-frame causal TCN
-> utility/release/regrasp
```

### R8-D-PROP25-VIS7

The same visual branch fused with a causal 25D GRU. This is the primary practical candidate because proprio supplies gripper command/state while visual features supply object/gripper geometry.

The visual-only candidate is not an extra architecture search. It is the predeclared modality ablation needed to determine whether the fused result is genuinely visual.

## Why not use a large end-to-end video model

The effective positive population is small: 83 feasible episodes in train600 and 26 in Fold-0 validation. End-to-end visual fine-tuning would add millions of free parameters and encourage task/background memorization.

The frozen-tower plus small temporal head route:

- uses the existing clean data efficiently;
- keeps trainable parameters well below one million;
- supports exact five-fold OOF;
- can be audited step-by-step;
- is cheap enough for controlled ablations;
- preserves a realistic online causal implementation.

## Visual-specific falsification controls

High validation performance alone is insufficient. The model could recognize task identity, static background or object category rather than the gripper-critical time.

R8 predeclares:

1. causal lag-20 frames from the same episode;
2. fixed within-task episode-deranged frames;
3. zero-visual input for the fused model;
4. the frozen R7-A proprio result.

Aligned RGB must outperform the better timing-destroying visual control by at least 20 percentage points in both recall and precision before making a visual-timing claim.

## Stage plan

### R8.0 — source binding

Read-only. Produce identity, trajectory, frame and camera binding ledgers. No model execution.

Possible outcomes:

- `PASS_BOUND`: proceed to deterministic train600 visual extraction;
- `HOLD_UNBINDABLE`: existing C2F RGB cannot be paired with Official K10 labels;
- `HOLD_PARTIAL`: some identities/steps fail; no silent subset training is permitted.

### R8.1 — frozen train600 embeddings

Conditional on R8.0 PASS. Extract image-tower tokens only for train600 and seal them.

### R8.2 — train-only OOF

Train the two fixed candidates on five exact 480/120 folds. Select thresholds only from OOF. If neither candidate passes, stop without reading validation images.

### R8.3 — one-time Fold-0 validation

Only an OOF-passing candidate may materialize and evaluate validation200 visual inputs. Controls are evaluated after model/threshold freeze and cannot trigger a rerun.

### R8.4 — exact-prefix readiness

Even a validation PASS does not authorize attack experiments. It only supplies a frozen Detector candidate for the already planned exact-prefix and command-OPEN/VIS canary audits.

## Claim boundary

A successful R8 model would establish:

> Causal clean RGB contains useful information for localizing the clean K10 gripper-critical opportunity corridor.

It would not establish:

- that the window is physically OPEN-sensitive;
- that a targeted VIS perturbation will trigger gripper OPEN;
- that timing beats random-window VIS;
- that the attack causes contact-quality failure.

Those claims remain reserved for exact-prefix command-OPEN and matched VIS/RAND interventions.

## Current authorization

```text
R7.3.1 proprio route       = terminal HOLD decision
R8.0 RGB source audit      = AUTHORIZED — READ ONLY
R8.1 embedding extraction  = HOLD
R8.2 visual training       = HOLD
R8.3 validation            = HOLD
R7.4 exact-prefix          = HOLD
R7.5 attack canary         = HOLD
```
