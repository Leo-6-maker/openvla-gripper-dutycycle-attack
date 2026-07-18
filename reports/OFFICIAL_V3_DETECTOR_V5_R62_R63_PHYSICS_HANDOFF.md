# Official V3 Detector V5 Physics R6.2/R6.3 Handoff

Date: 2026-07-19  
Code HEAD for this handoff: `d1a4182cd3c22783ce411d522bdec6692f2c0bfd`  
Branch: `codex/official-v3-detector-v5-20260718`  
PR: #87, Draft

## Scope and safety boundary

This handoff covers the CPU R6.2 diagnostics and the conditional R6.3 matched
development smoke. It is FIT-only and clean-only. No CLEAN/S1 mutation,
protected-split semantic read, attack execution, or formal model-selection
authorization is permitted by this evidence.

The R6.3 A/B smoke was started only after the R6.2 engineering conditions were
met. It uses a new sealed 160-identity subset from Fold-0 train identities and
the full natural Fold-0 validation set. It is not a formal trainer run.

## Sealed inputs

| Evidence | Root or file seal |
|---|---|
| Physics Teacher V2.1 | `18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da` |
| Physics Teacher V2.1 independent audit | PASS; bound to the Teacher root seal |
| R6.2 final CPU audit root | `b59eb5d3b7d76f03d496ded5194c6ee21ccbca5503e55bd2a71c9673465df0e9` |
| R6.2 category-balanced subset root | `06e6e520fae6181da181fc6b37bd4c9599a2054c26ccaa9a8dd5cbf27c50148a` |
| R6.2 subset identity-list SHA | `820b6e68b2e8bc79f0073141fdf704a63df52a17aa58f35f18d400a791561df5` |
| Frozen R6.2 loss protocol | `DETECTOR_V5_PHYSICS_LOSS_PROTOCOL_V3.json` |

## R6.2-A: causal target identifiability

The final R6.2 census contains 5,143 complete candidate segments and 1,367
segments with a 10-step decision anchor. Fifty anchorable segments become
tier 2/3 only after the anchor, or 3.6576% of anchorable segments. The anchor
tier differs from the final segment maximum in 504/1,367 cases (36.8691%).

This is not a reason to call the target unusable, but it requires an explicit
causal target. R6.3 therefore trains and diagnoses against the tier available
at the decision anchor, excludes short segments from causal positive targets,
and retains final-segment tier only as retrospective analysis.

Anchor-local episode categories are:

| Category | Count |
|---|---:|
| NO_CANDIDATE | 40 |
| PURE_NEGATIVE | 422 |
| TRUE_MIXED | 338 |

The final-max loader categories were 40 / 392 / 19 / 349 for
NO_CANDIDATE / PURE_NEGATIVE / POSITIVE_ONLY / TRUE_MIXED respectively.

## R6.2-B: category geometry

The anchor-local category census is:

| Set | Identities | Segments with anchor | No candidate | Pure negative | True mixed |
|---|---:|---:|---:|---:|---:|
| Fold-0 train | 600 | 1,027 | 30 | 316 | 254 |
| Fold-0 validation | 200 | 340 | 10 | 106 | 84 |
| Old 80-episode smoke train | 80 | 145 | 4 | 45 | 31 |
| New balanced smoke train | 160 | 297 | 8 | 84 | 68 |

The new subset is deterministic, uses only Fold-0 train identities, contains
exactly 4 identities per each of 40 tasks, and is sealed independently. The
validation set remains the natural 200-identity Fold-0 validation set.

## R6.2-C: shallow learnability bounds

On the same Fold-0 validation set, using anchor-causal segment features:

| Input | True-mixed top-1 | Top-2 | Positive recall at .5 | Pure-negative abstention |
|---|---:|---:|---:|---:|
| 25D proprio | 0.8929 (25/28) | 0.9643 | 0.9412 | 0.8951 |
| 25D + policy intent | 0.8929 (25/28) | 1.0000 | 0.8824 | 0.8951 |
| Privileged Physics components | 1.0000 | 1.0000 | 1.0000 | 0.9506 |

The privileged oracle is learnable. The shallow 25D model has useful signal,
while policy intent adds no top-1 gain in this diagnostic. These results do
not establish an observability limit and do not select a model.

Simple causal baselines were not competitive with the shallow 25D top-1
diagnostic: earliest and first-T10 were 0, causal dwell/longest was 0.4286,
latest causal segment was 0.8929, and policy close-margin peak was 0.25.
These are diagnostic baselines only and do not authorize deployment.

## R6.3 conditional smoke

Two matched development runs were launched in the official A800 environment:

- `V5_A_PHYSICS`: physical GPU 1;
- `V5_B_PHYSICS`: physical GPU 2, with the sealed policy-intent root.

Both use the same sealed 160-identity train subset, natural 200-identity
validation set, seed `20260717`, 10 epochs, FP32, deterministic development
protocol, and no early stopping. Each has an independent output root and
process log. At the time this report was created, both processes were still
running; no R6.3 metric or gate is claimed here. The final result section must
be updated only after process completion and recursive seal verification.

## Current gates

```text
R6.2_CPU_DIAGNOSTICS       = PASS
CAUSAL_TARGET_AUDIT        = PASS WITH CAUSAL-TARGET CAVEAT
CATEGORY_BALANCED_SUBSET   = PASS (ENGINEERING)
PRIVILEGED_ORACLE          = PASS / LEARNABLE
R6.3_MATCHED_SMOKE         = RUNNING
FULL_FOLD0                 = HOLD
FIT_DEV                    = NOT READ
CAL                         = NOT READ
CHECK                       = NOT READ
ATTACK                     = NOT STARTED
SOURCE_ARTIFACT_MUTATION   = 0
```

