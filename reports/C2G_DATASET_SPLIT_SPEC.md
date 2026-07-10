# C2g Dataset and Split Specification

Date: 2026-07-10

Status: `PASS_STATIC_SCAFFOLD`; no dataset was materialized.

## Context modes

```text
no_context
suite_only
full_context_legacy
```

- `no_context`: primary candidate; 25D temporal, visual, and language inputs only.
- `suite_only`: optional four-dimensional suite one-hot diagnostic.
- `full_context_legacy`: existing `ctx_*`/task-hash inputs, shortcut diagnostic only. It cannot support a task-generalization claim.

The primary C2g model API has no raw task-index/hash input. Shuffled-language and permuted-task-context controls are deterministic episode-level permutations performed within each split.

## Split modes

```text
within-task
leave-one-task-out
leave-one-suite-out
```

- `within-task` deterministically hashes whole episodes to train/val/test inside the observed task distribution.
- `leave-one-task-out` uses a namespaced `suite:task` fold as test; all other episodes are train/val.
- `leave-one-suite-out` uses one full suite as test; all other episodes are train/val.

Every window inherits its episode split. Duplicate episode assignments across splits are a hard error. Future materializers must also freeze each split manifest and SHA256, report rows, episodes, known positives, known negatives, unknowns, attackable episodes, fully-known negative episodes, task count, and suite count per fold, and fit normalization/calibration on train/validation only. A fold lacking required positive, negative, episode, task, or suite support is a hard HOLD.

## Weighting

For task `k`, episode `e`, and row `i`, the scaffold assigns unnormalized weight:

```text
w_i = 1 / (number_of_tasks * episodes_in_task_k * rows_in_episode_e)
```

Weights are normalized to mean one. Therefore each task has equal total mass, each episode within a task has equal mass, and long/high-density episodes cannot dominate. Unknown/abstain label masks remain separate from weights.

## Shortcut diagnostics

- `shuffled-language`: deterministic donor-episode language permutation within split.
- `wrong-language-cross-task`: deterministic same-split donor from a different task; identity inconsistency or absence of a valid donor is a hard error.
- `permuted-task-context`: deterministic donor-episode legacy context permutation within split.
- Neither permutation crosses train/val/test or becomes a primary model input.

CPU tests cover all context/split modes, episode leakage rejection, fold viability, task/episode mass balance, and deterministic split-local diagnostic permutations.

```text
NO_CONTEXT_AND_GENERALIZATION_SPLITS = PASS_STATIC
EPISODE_LEAKAGE = 0_IN_TESTS
DATASET_MATERIALIZATION = NOT_STARTED
EMBEDDING_MATERIALIZATION = NOT_STARTED
```
