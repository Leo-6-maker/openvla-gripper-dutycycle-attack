# Official V3 R7.2 Offline Replay Audit — HOLD

Date: 2026-07-19  
PR: #87, Draft  
Reviewed submission: `eb40429f755b30d87ea6e307d6a7e007041dea7b`  
Reviewed evaluator branch: `agent/official-v3-r7-k10-v1.2.1-20260719`

## Verdict

The submitted R7.2 artifact must be preserved, but it is **not a valid replay of the frozen V5-A/V5-B checkpoints** and cannot support the claim that Physics checkpoints fail to transfer to the K10 opportunity target.

```text
R7_R1_FORMAL_ARTIFACT          = PASS
R7_R2_SUBMITTED_ROOT           = PRESERVE / INVALID_FOR_SCIENTIFIC_CLAIMS
R7_R2_MODEL_REPLAY_VALIDITY    = HOLD
R7_R2_METRIC_VALIDITY          = HOLD
R7_R2_BASELINE_COMPLETENESS    = HOLD
R7_R2_1_CORRECTIVE_REPLAY      = AUTHORIZED — READ ONLY
R7_R3_TRAINING                 = HOLD
R7_R4_EXACT_PREFIX             = HOLD
R7_R5_ATTACK_CANARY            = HOLD
```

The qualitative hypothesis that K10-specific training may be needed remains plausible, but it is not established by the submitted replay.

## P0 findings

### P0-1 — threshold-expanded denominator

`compute_metrics()` counts `n_feasible` and `n_no_feasible` over the full list in which every identity is duplicated once per threshold. It then divides each single-threshold numerator by those nine-times-expanded denominators.

For the reported Fold-0 validation population:

```text
unique episodes                 = 200
unique feasible episodes        = 26
thresholds                      = 9
incorrect n_feasible            = 26 × 9 = 234
```

Therefore the submitted values are divided by nine:

```text
V5-A: 3/234 = 0.0128   submitted
      3/26  = 0.1154   denominator-corrected only

V5-B: 1/234 = 0.0043   submitted
      1/26  = 0.0385   denominator-corrected only
```

These corrected fractions are not accepted final results because the model replay itself is also invalid. `no_corridor_abstention` is affected by the same denominator bug, and aggregate `n_episodes` becomes 1,800 rather than 200.

### P0-2 — V5-A is not executed with its trained architecture

Training used `CausalMultimodalVulnerabilityRanker(V5ModelContract(...))`. For V5-A, the official forward path projects the proprio hidden state, applies a one-branch softmax gate, and then applies `Linear + Tanh` fusion.

The submitted evaluator reconstructs a different model:

- sigmoid gating rather than the official one-branch softmax behavior;
- a mixture of raw and projected hidden states;
- no `Tanh` after fusion;
- `strict=False` checkpoint loading.

The resulting utility sequence is not the frozen V5-A checkpoint output.

### P0-3 — V5-B policy intent is discarded

The submitted evaluator detects the intent branch from state-dict key names, but feeds an all-zero 9D tensor:

```text
intent_dummy = zeros([1,T,9])
```

It does not read the sealed policy-intent root, does not use each episode's `clean_policy_intent_9d`, and does not apply the checkpoint's `normalization_mean_9d` / `normalization_std_9d`.

The submitted V5-B result is therefore not a replay of V5-B.

### P0-4 — silent partial weight loading

The evaluator remaps selected keys and calls `load_state_dict(..., strict=False)`. Missing and unexpected parameters are not audited. A formal frozen-checkpoint replay must instantiate the repository model from the checkpoint contract and load with `strict=True`.

### P0-5 — scheduler semantics are replaced

The frozen V5 evaluator uses `V5OneShotScheduler` with:

```text
valid-step gate
candidate-close gate
minimum candidate dwell = 10
3-of-5 persistence
release veto
regrasp veto
one-shot state
```

The submitted evaluator emits at the first step satisfying only:

```text
candidate_close AND utility >= threshold
```

It does not implement Student-valid gating, dwell, persistence, release veto, regrasp veto, or the frozen scheduler state machine. Thus it does not evaluate checkpoint transfer under the deployable V5 decision rule.

### P0-6 — fold and source lineage are not closed

The evaluator hard-codes states `0..4` rather than consuming the sealed fold manifest. It silently skips missing files, defaults missing Student validity to `True`, and does not verify:

- checkpoint bundle seals;
- S1 root seal;
- K10 root seal;
- Physics Teacher root seal;
- policy-intent root seal for V5-B;
- exact 200-identity fold closure;
- exact step-count parity between Student, Teacher, intent, and K10 streams.

### P0-7 — no auditable episode ledger

The in-memory per-threshold episode records are not written to the output root. The sealed root contains only aggregate `replay_report.json`, so hits, false emits, timing, and paired A/B decisions cannot be independently recomputed.

## P1 findings

### P1-1 — required R7.2 metrics are missing

The authorization required:

```text
K10 feasible-hit recall
emit precision
positive-episode coverage
no-corridor abstention
false-early emit
late/outside-corridor emit
release/post-release emit
outside-rankable emit
one-shot compliance
first-feasible-start delay
K10 containment
```

The handoff reports only recall, precision, hits, and emits. It does not close the required timing and safety metrics.

### P1-2 — baseline implementation does not match its description

The script claims first-eligible-T10, causal dwell, and max-so-far baselines, but implements only the first `candidate_close=True` step. This is an early-close baseline, not T10 eligibility or causal dwell.

The existing shallow 25D diagnostic is also omitted rather than reported as lineage-matched or unavailable.

### P1-3 — correlation claim is unsupported

The statement that utility scores do not correlate with K10 starts is stronger than the analysis performed. The evaluator checks only the first threshold crossing. It does not calculate score separation, within-episode K10 ranking, AUROC/AUPRC, or retrospective peak-in-corridor diagnostics. A model may contain weak ranking signal while its frozen scheduler emits too early; these are different conclusions.

### P1-4 — Git integration and CI are not closed

`eb40429...` is not on PR #87's head. Its branch diverges from the integrated PR lineage, and no GitHub Actions runs are associated with the submitted commit. The branch must not be merged or fast-forwarded wholesale.

## R7.2.1 corrective replay contract

Only the following narrow read-only correction is authorized.

### Required repository implementations

The corrected evaluator must import and use the existing official components:

```text
gripper_attack.v5_dataset.load_fit_registry
gripper_attack.v5_dataset.load_v5_episodes
gripper_attack.v5_dataset.load_policy_intent_root
gripper_attack.v5_protocol.V5ModelContract
gripper_attack.v5_protocol.variant_uses_intent
gripper_attack.v5_ranker.CausalMultimodalVulnerabilityRanker
gripper_attack.v5_scheduler.V5OneShotScheduler
gripper_attack.v5_scheduler.V5SchedulerConfig
gripper_attack.b3_training_protocol.load_fit_fold_bundle
gripper_attack.b3_training_protocol.verify_sealed_directory
```

For each checkpoint:

1. instantiate `CausalMultimodalVulnerabilityRanker` from the checkpoint's exact `model_contract`;
2. call `load_state_dict(..., strict=True)`;
3. load the exact sealed Fold-0 validation identities from the fold bundle;
4. use the official 25D Student stream and checkpoint 25D normalization;
5. for V5-B, require the sealed policy-intent root and checkpoint 9D normalization;
6. replay utility, release, and regrasp through the frozen `V5OneShotScheduler` for every diagnostic threshold;
7. do not select or freeze a threshold.

### Population and denominator gates

For every threshold and every candidate:

```text
unique identity count       = 200
identity set                = exact Fold-0 validation set
positive denominator        = unique K10-positive episodes at that threshold
no-corridor denominator     = unique K10-negative episodes at that threshold
threshold duplication       = forbidden in denominators
missing/skipped identities  = 0
step-count mismatch         = 0
```

The expected positive count may be 26 based on the submitted label geometry, but it must be recomputed and bound to the sealed fold and label roots.

### Required outputs

The new sealed root must contain:

```text
MANIFEST.json
SOURCE_BINDING.json
threshold_metrics.csv
episode_threshold_ledger.jsonl
baseline_episode_ledger.jsonl
baseline_metrics.csv
score_diagnostics.csv
AUDIT.json
commands.txt
stdout.log
SHA256SUMS
SHA256SUMS.sha256
```

`SOURCE_BINDING.json` must bind the full digests of:

```text
V5-A checkpoint bundle
V5-B checkpoint bundle
S1 root
Physics Teacher V2.1 root
K10 V1.2.1 label root
Fold root
FIT registry
V5-B policy-intent root
corrected evaluator commit and file SHA
```

### Required metrics

At each threshold, separately for A and B:

```text
unique positive episodes
unique no-corridor episodes
legal K10 hits
feasible-hit recall
emits
emit precision
positive-episode coverage
no-corridor abstention
false-early count/rate
late-or-gap count/rate
release-veto count
regrasp-veto count
release/post-release emit count
outside-rankable emit count
one-shot compliance
hit delay relative to first feasible start
K10 containment
```

Diagnostic-only score analysis:

```text
maximum score inside legal corridor
maximum score outside legal corridor
within-episode best-step-in-corridor rate
step-level AUROC/AUPRC with explicit class imbalance
```

These diagnostics may distinguish representation transfer from scheduler transfer, but are not deployable baselines.

### Baselines

At minimum report:

```text
first candidate-close step                 early reference
first valid candidate step after past dwell=10
frozen causal dwell-so-far rule
existing shallow 25D predictions if exact lineage matches
```

Any future-informed latest/longest/global-best rule must be labeled retrospective only.

## Stop boundary

After producing the corrected sealed replay and independent audit, stop for review.

```text
R7_R3_TRAINING          = NOT AUTHORIZED
R7_R4_EXACT_PREFIX      = NOT AUTHORIZED
R7_R5_ATTACK_CANARY     = NOT AUTHORIZED
FIT_DEV/CAL/CHECK       = NOT READ
CS200                   = NOT READ
ATTACK                  = NOT RUN
```
