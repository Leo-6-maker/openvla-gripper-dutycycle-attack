# C2g Detector Model Ladder

Date: 2026-07-10

Status: `PASS_SPEC`; no dataset was materialized and no model was trained.

## Model ladder

| Model | Inputs | Purpose |
|---|---|---|
| A: C2g-Temporal | causal 25D history | temporal-only lower bound |
| B: C2g-Global | temporal history + global pooled SigLIP | visual baseline without language |
| C: C2g-Global-Lang | temporal history + global visual + language FiLM/gating | primary compact baseline |
| D: C2g-PatchAttn | temporal history + projected SigLIP spatial tokens + language-query attention | spatial grounding candidate |

The primary API accepts no task index, task hash, episode identity, teacher phase, contact object identity, target pose, attack outcome, or post-intervention state. C2f classes/checkpoints remain untouched.

## Visual representation

Global pooled SigLIP remains the required fallback. Model D accepts upstream spatial tokens and applies a learned projection plus language-query attention. Initial materialization candidates are adaptive 2x2 (4 tokens) and 4x4 (16 tokens), projection dimension 128.

Projected activation cost per endpoint in FP32 is approximately 2 KiB for 4 tokens or 8 KiB for 16 tokens, before batch/time expansion. Attention is `O(tokens * 128)` per endpoint. Input read cost remains `tokens * D_siglip`; the exact source dimension and measured latency must be recorded by any future materializer. If only a global embedding exists, the run must use Model B/C and cannot claim patch-attention evidence.

## Temporal encoder

The baseline is one causal GRU or TCN over 25D. Prefix outputs must be invariant to future input changes. A diagnostic dual-stream refinement may separately encode action/policy fields and physical-state fields before fusion, but it requires an ablation and is not presumed superior.

## Outputs and online gate

Primary output: `vulnerability_logit`.

Auxiliary outputs: `release_safe_logit`, `contact_stable_logit`, and `grounding_confidence_logit`.

Initial online gate:

```text
vulnerability_probability >= tau_vulnerability
AND release_safe_probability < tau_release
AND grounding_confidence_probability >= tau_ground
AND 2-of-3 persistence
```

One isolated spike cannot trigger. The old phase/corridor/release/event-role conjunction is not the primary path.

## Losses

Window loss uses unknown-masked, active-weight-mass-normalized BCE or a preregistered focal alternative. Auxiliary losses use separate masks. Teacher confidence may weight rows only after calibration audit.

Episode terms penalize persistent early emit before the first known causal interval, miss of all persistent scores within a known interval, persistent emit in explicitly fully-known negative episodes, and persistent emit in known release-safe intervals. Unknown windows cannot support positive or negative persistence. A short interval with fewer than two known eligible windows cannot be satisfied by one spike and is excluded from the persistence-miss term.

## Dataset modes and controls

`NO_CONTEXT` means no raw suite/task identity: temporal + visual + language. Required diagnostics are `TEMPORAL_ONLY`, `TEMPORAL_PLUS_GLOBAL_VISUAL`, `TEMPORAL_PLUS_PATCH_VISUAL`, `NO_LANGUAGE`, `LANGUAGE_DROPOUT`, `SHUFFLED_LANGUAGE`, `WRONG_LANGUAGE_CROSS_TASK`, `SUITE_ONLY`, `FULL_CONTEXT_LEGACY`, and `PERMUTED_TASK_CONTEXT`.

`FULL_CONTEXT_LEGACY` cannot support task-generalization claims. Wrong-language donors stay in the same split, differ in task identity, and are selected deterministically.

## Splits and viability

Use episode-level `WITHIN_TASK_EPISODE_SPLIT` only as an in-distribution reference. `LEAVE_ONE_TASK_OUT` is the primary generalization evaluation; `LEAVE_ONE_SUITE_OUT` is harder diagnostic evidence. Every window inherits its episode split, and each frozen split manifest has a SHA256.

Each fold reports rows, episodes, known positives, known negatives, unknowns, attackable episodes, fully-known negative episodes, tasks, and suites. Missing support is a hard HOLD before training.

## Selection metrics

Window metrics: precision, recall, F1, PR-AUC, Brier score, ECE, and ROC-AUC as secondary evidence.

Deployment metrics: episode any-emit false-positive rate, attackable-episode no-emit rate, first-emit precision/timing error, early-trigger rate, release-safe emit fraction, persistent-trigger coverage, suite/task macro metrics, and leave-one-task-out metrics. Pooled window F1 alone cannot select a model.

## Confidence and falsification review

| Working confidence | Claim | Increase evidence | Decrease/falsify evidence |
|---:|---|---|---|
| 90% | C2f failures mainly reflect labels, grounding, shortcuts, and deployment mismatch | Teacher-v2 improves held-out deployment metrics at fixed capacity | larger C2f models solve matched controls without label changes |
| 85% | counterfactual Teacher-v2 is better aligned | restore-parity replay labels predict matched online harm | phase and command-open harm are weakly related; restore is unstable |
| 75% | no-context + leave-one-task-out reduce shortcut inflation | legacy context wins within-task but not LOTO; controls degrade as expected | task-identity models alone generalize under clean held-out tests |
| 65% | C2g timing beats matched random-time force-open | paired TRUE harms more parents than matched random timing | ORACLE or C2g is no more harmful than matched random timing |
| 55% | patch attention materially helps | patch path improves grounding/LOTO with acceptable latency | no gain over global visual across seeds/folds |
| 40% | one detector/threshold works equally across suites | all suites satisfy preregistered deployment gates | persistent suite-specific calibration or a failed suite |

Additional falsification conditions are no-context model failure while task identity succeeds, wrong-language controls showing no degradation, inability to approach ORACLE timing despite adequate labels, and online effects disappearing under matched-payload replication.

```text
C2G_MODEL_LADDER = PASS_SPEC
PATCH_ATTN_PATH = PASS_STATIC_SKELETON
CAUSAL_PERSISTENCE_LOSS = PASS_STATIC
MODEL_TRAINING = NOT_RUN
```
