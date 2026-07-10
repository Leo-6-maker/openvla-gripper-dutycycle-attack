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
- a compact global SigLIP projection with an optional language-query patch-token path;
- language-conditioned FiLM plus a learned gate over temporal/visual state;
- one direct `vulnerability` logit;
- auxiliary `release_safe`, `contact_stable`, and `grounding_confidence` logits.

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

`L_early_emit` penalizes probability before the first known vulnerable interval. `L_episode_miss` rewards a 2-of-3 persistent emit in a vulnerable interval; one isolated spike cannot satisfy it. `L_any_emit_negative_episode` penalizes persistent emit only in an explicitly fully-known negative episode. `L_release_safe_emit` penalizes persistent vulnerability during known release-safe intervals. Task/episode-balanced sample weights are normalized by active weight mass.

## Deployment contract

The primary online trigger uses calibrated direct vulnerability probability, a release-safe veto, a grounding-confidence floor, and mandatory 2-of-3 persistence. `release_safe` becomes a deployment veto only after Teacher-v2 establishes valid release labels. Primary claims use a single held-out-task calibration; per-suite thresholds are secondary diagnostics only.

The CPU test verifies output heads, causal prefix invariance, absence of a task-index input, finite episode losses, and unknown masking.

```text
C2G_ARCHITECTURE_SPEC = PASS
C2G_MODEL_SKELETON = PASS_STATIC
C2G_PATCH_TOKEN_SKELETON = PASS_STATIC
C2G_TRAINING = NOT_STARTED
```
