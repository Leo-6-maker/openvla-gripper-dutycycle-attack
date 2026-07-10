# C2g Causal Vulnerability Detector Architecture

Date: 2026-07-10

Status: `PASS_SKELETON_ONLY`; no checkpoint was trained.

## Primary student inputs

- causal clean 25D history `[t-W+1, ..., t]`;
- current/aligned OpenVLA SigLIP visual feature;
- task language embedding available before rollout;
- no privileged object/target/contact state;
- no raw task index, task hash, episode identity, teacher phase, or attack outcome.

## Skeleton

`C2gCausalVulnerabilityDetector` uses:

- a single-layer GRU over the 25D history; each output depends only on current/past rows;
- a compact SigLIP projection;
- language-conditioned FiLM plus a learned gate over temporal/visual state;
- one direct `vulnerability` logit;
- auxiliary `release_safe`, `contact`, and `grounding` logits.

The module accepts no suite/task-index tensor. `FULL_CONTEXT_LEGACY` remains a dataset diagnostic baseline, not the primary C2g interface.

## Loss interface

Window losses use masked BCE. Unknown Teacher-v2 rows are removed by `known_mask`, not assigned zero. Auxiliary losses are separately weighted.

The episode interface exposes:

```text
L_window_vulnerability
L_early_emit
L_episode_miss
L_any_emit_negative_episode
L_release_safe + L_contact + L_grounding
```

`L_early_emit` penalizes probability before the first known vulnerable interval. `L_episode_miss` rewards at least one emit in a vulnerable interval. `L_any_emit_negative_episode` penalizes any emit in a fully known negative episode. Task/episode-balanced sample weights are accepted by the window loss.

## Deployment contract

The primary online trigger uses calibrated direct vulnerability probability with optional 2-of-3 persistence/hysteresis. `release_safe` is a safety veto only after Teacher-v2 establishes valid positive/negative release labels. Primary claims use a single held-out-task calibration; per-suite thresholds are secondary diagnostics only.

The CPU test verifies output heads, causal prefix invariance, absence of a task-index input, finite episode losses, and unknown masking.

```text
C2G_ARCHITECTURE_SPEC = PASS
C2G_MODEL_SKELETON = PASS_STATIC
C2G_TRAINING = NOT_STARTED
```
