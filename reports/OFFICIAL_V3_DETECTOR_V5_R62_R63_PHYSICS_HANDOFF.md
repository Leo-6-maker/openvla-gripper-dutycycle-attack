# Official V3 Detector V5 Physics R6.2/R6.3 Handoff

Date: 2026-07-19  
Code HEAD for this handoff: `c9709bfa5ef7788b39b089db33cf29da68728f74`  
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

## R6.2 correction: loader-causal category closure

The first R6.3 smoke exposed that the R6.2 raw-segment category table was not
the same as the category used by the training loader. The original table
counted raw Physics segments before the loader's exact Student-valid,
known, contiguous, minimum-dwell filtering. The corrected audit root records
both views and uses the loader-causal view for subset selection:

| Set | Raw NO/PURE/TRUE | Loader-causal NO/PURE/TRUE |
|---|---:|---:|
| Fold-0 train (600) | 30 / 316 / 254 | 30 / 487 / 83 |
| Fold-0 validation (200) | 10 / 106 / 84 | 10 / 162 / 28 |
| Old smoke train (80) | 4 / 45 / 31 | 4 / 67 / 9 |
| Corrected subset (160) | 8 / 82 / 70 | 8 / 126 / 26 |

The corrected subset therefore closes the loader-semantic selection rule, but
it cannot make the available Physics categories numerically balanced under the
fixed 4-identities-per-task constraint. This limitation is explicit; the
subset is not described as 50/50 balanced.

Corrected audit and subset seals:

- R6.2 corrected audit root: `64238e2c63673d1d3eabbfe65e3b74952d0ba051f407a87657b1d4d5e0f66a88`;
- corrected subset root: `7b733ea8d750a87448a168f7ba0faf9eef746389301795446c0415ea0e41d2ff`;
- corrected subset identity SHA: `421dcfeb727627cda0dc969ee1a4ea6108152e567b721cb53df44530b63e83f0`.

The first A/B smoke roots are retained as sealed execution references, but
their training-category contract was not closed and their diagnostic metrics
are not used for a scientific decision.

## R6.3 corrected matched smoke

The corrected A/B smoke used the loader-causal subset above, the natural
200-identity Fold-0 validation set, seed `20260717`, 10 epochs, FP32, no early
stopping, and separate physical GPUs (A on GPU1, B on GPU2). Both checkpoint
roots passed `sha256sum -c SHA256SUMS`:

| Candidate | Checkpoint root seal | Smoke diagnostic (non-primary) |
|---|---|---|
| A | `77ac6ac0ce0edc2d8be51477ff671180a6e4b6b26c9102b28a068cafcb352ac9` | 65/84 raw diagnostic top-1; 97/106 raw abstention |
| B | `c44a05ceca9130ff147ab1e1290461319309eadd0bdcef27c05e1c6d3e2a4a97` | 63/84 raw diagnostic top-1; 98/106 raw abstention |

Those smoke diagnostic numerators were produced before the final short-window
diagnostic fix and are explicitly not used below. The sealed checkpoints were
replayed with the corrected strict causal evaluator.

## Strict causal online replay

The replay uses the corrected causal tier, exact contiguous windows, minimum
dwell, candidate and Student-valid gates, one-shot scheduler, and fixed
threshold grid. No threshold reached critical-window recall `>= 0.95`, so both
working points are `HOLD` and selected-threshold metrics are null.

| Candidate | Causal anchor top-1 | Pure-negative abstention | Default replay emits | Outside-rankable emits | Working point |
|---|---:|---:|---:|---:|---|
| A | 24/28 = 0.8571 | 157/162 = 0.9691 | 7 | 1 | HOLD |
| B | 21/28 = 0.7500 | 158/162 = 0.9753 | 4 | 0 | HOLD |

The online evaluation roots also passed recursive checksum verification:

- A evaluation root: `7e42b8a7759c5221ec9541343b0aefba3d25753ed4cb0494bb5417b6bdf37650`;
- B evaluation root: `0fb62a371013673bc0c78c6638e04c807260e5f0796db12048f57e3586738fd6`.

`causal_anchor_top1` is an argmax diagnostic, not scheduler selection
accuracy. Because the fixed recall working point is absent,
`mixed_scheduler_correct_selection` is not reported as a pass. A also has a
default-threshold outside-rankable emission, which fails the smoke safety gate.
Neither result supports a full Fold-0 run.

## Current gates

```text
R6.2_CPU_DIAGNOSTICS       = PASS
CAUSAL_TARGET_AUDIT        = PASS WITH CAUSAL-TARGET CAVEAT
CATEGORY_BALANCED_SUBSET   = PASS (LOADER-CLOSED; CATEGORY-SKEW EXPLICIT)
PRIVILEGED_ORACLE          = PASS / LEARNABLE
R6.3_MATCHED_SMOKE         = HOLD
R6.3_CAUSAL_ONLINE_REPLAY  = PASS SEAL / HOLD GATE
V5_A_PROPRIO_PHYSICS       = HOLD
V5_B_PROPRIO_POLICY        = HOLD
FULL_FOLD0                 = HOLD
FIT_DEV                    = NOT READ
CAL                         = NOT READ
CHECK                       = NOT READ
ATTACK                     = NOT STARTED
SOURCE_ARTIFACT_MUTATION   = 0
```

The best simple causal baseline in R6.2 was `0.8929` top-1. Corrected A is
`0.8571` and corrected B is `0.7500`; neither exceeds that baseline by the
required development margin. No complete Fold-0, multi-seed run, FIT-DEV,
CAL, CHECK, or attack is authorized by this handoff. All old and corrected
roots, logs, checkpoint bundles, prediction bundles, and failure logs remain
preserved.
