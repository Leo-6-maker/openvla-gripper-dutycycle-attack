# R7.3 K10-Specific Detector Training Audit — FORMAL HOLD

Date: 2026-07-19  
PR: #87  
Submitted head reviewed: `e2ccdb7b8d71767e616ba0267205ae319df61eee`

## Verdict

```text
R7_R3_DECISION_LEVEL_HOLD          = SUPPORTED
R7_R3_PROTOCOL_CONFORMANCE         = HOLD
R7_R3_ARTIFACT_RECOMPUTABILITY     = HOLD
R7_R3_INDEPENDENT_AUDITOR          = HOLD
R7_R3_FORMAL_CLOSURE               = HOLD

R7_R3_1_TRAIN600_OOF_CLOSURE       = AUTHORIZED — SAME MODELS / SAME SEED / NO TUNING
FOLD0_VALIDATION_PAYLOAD_READS     = 0 REQUIRED
FURTHER_PROPRIO_TUNING             = NOT AUTHORIZED
VISUAL_MODALITY_TRAINING           = NOT AUTHORIZED
R7_R4_EXACT_PREFIX                 = HOLD
R7_R5_ATTACK_CANARY                = HOLD
```

The submitted evidence supports the operational decision to stop both candidates as `HOLD_OOF`, but it does not yet constitute the frozen R7.3 V1 experiment or a recomputable formal artifact.

## Findings

### P0 — OOF folds violate the frozen 480/120 contract

The frozen protocol requires five models trained on exactly 480 identities and evaluated on exactly 120 OOF identities. The submitted GRU folds are:

```text
476/124
477/123
480/120
483/117
484/116
```

The current round-robin-per-stratum builder does cover all 600 identities exactly once, but it does not enforce five equal-capacity validation folds. The result therefore cannot be described as execution of the frozen 480/120 recipe.

### P0 — two mandatory gates are not computed

`compute_metrics()` hard-codes:

```text
outside_rankable_emit = 0
release_regrasp_emit  = 0
```

and `check_oof_gates()` checks only recall, precision, abstention and one-shot compliance. The protocol requires all six conditions, including zero outside-rankable emissions and zero Teacher release/regrasp emissions.

The reported Linear rows already contain false emissions. For example at tau=0.25 there are 188 emits and 27 K10 hits, so 161 emissions are outside the K10 start set; the report nevertheless omits the outside-rankable gate failure.

This defect does not rescue either candidate: failure of recall/precision/abstention is sufficient to make the full conjunction fail. It does, however, prevent a claim that all six gates were independently evaluated.

### P0 — HOLD auditor is not an independent metric audit

For a `HOLD_OOF` root, the auditor requires only:

```text
HOLD_OOF.txt
OOF_REPORT.json
```

It skips identity manifests, source binding, fold geometry, predictions, threshold ledgers, metric recomputation, normalization provenance and validation-read closure. A PASS from this auditor demonstrates seal consistency plus a self-declared HOLD report; it does not independently establish that 19 thresholds failed the frozen gates.

### P0 — GRU per-threshold evidence is absent from the sealed run

The GRU root was generated before `all_results` was added to the parallel OOF report. The subsequent code fix does not retroactively add the missing 600 × 19 episode-threshold ledger to the sealed `adbe128...` root. Therefore the GRU `HOLD_OOF` decision is not independently recomputable from the submitted artifact.

### P1 — HOLD roots do not satisfy the required artifact contract

On `selected_tau is None`, the trainer writes only the OOF report and HOLD marker before sealing. It does not write the required source binding, exact identity/fold manifests, per-fold normalization, training histories, fold checkpoints, OOF prediction ledger, threshold metrics, protocol record or manifest.

### P1 — CI was not wired to R7.3 at submitted head

At `e2ccdb7...`, the workflow did not compile the R7.3 trainer/auditor or run `tests/test_r7_k10_v3_training.py`. This has been corrected on the PR branch after audit.

### P1 — claim boundary

The defensible current statement is:

> In the submitted train600-only OOF runs, neither the Linear nor GRU candidate produced an eligible working point under the implemented recall/precision/abstention scheduler checks. The proprio-only route remains HOLD.

The current evidence does not justify claiming that the exact frozen 480/120 protocol was completed, nor that every mandatory safety gate was evaluated.

## R7.3.1 narrow closure authorization

A single corrective OOF closure is authorized because no Fold-0 validation payload was evaluated and no threshold was selected. It is not hyperparameter tuning.

Frozen constraints:

```text
candidates       = existing R7-S-LINEAR-25D and R7-A-GRU-25D only
seed             = 20260717
optimizer/loss   = unchanged
training epochs  = 10 exact
OOF folds        = exactly five 480/120 folds
stratification   = suite + episode-level K10 feasibility
validation reads = 0 payload/feature/label rows
final model      = forbidden when HOLD_OOF
new seed         = forbidden
parameter change = forbidden
```

Required corrections:

1. Build and seal an exact-capacity OOF identity manifest: five disjoint 120-identity validation folds, union 600; each complementary training fold exactly 480.
2. Store one OOF prediction per train identity and all 19 scheduler ledgers for both candidates.
3. Compute `outside_rankable_emit` from actual emission locations rather than a constant.
4. Compute Teacher release/regrasp emissions from the frozen Physics V2.1 labels at the emitted step, including validity masks.
5. Evaluate all six OOF gates exactly as frozen; select no fallback threshold.
6. For every fold, seal model checkpoint, normalization, train identities, OOF identities, ten-epoch history and source binding.
7. For each candidate root, seal full OOF metrics, prediction ledger, identity manifest, protocol, source binding, manifest and explicit `validation_payload_reads = 0` evidence.
8. Use an independent read-only auditor that recomputes all 19 metrics and gate decisions from prediction records and Teacher/K10 targets, checks exact 480/120 closure, and rejects missing fields.
9. Preserve all existing R7.3 roots; write new immutable R7.3.1 roots.
10. Stop after both corrected OOF roots and audits are submitted. No validation, final training, visual training, exact-prefix work or attack canary is authorized.

If both corrected candidates remain `HOLD_OOF`, R7.3 closes formally and no further proprio-only development on this split is permitted.
