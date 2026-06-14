# M3 Step78 True-PGD Fixed-Frame Preregistration

This preregistration covers the first real-model fixed-frame canary for the M3 true-PGD Layer3 route. It does not report scientific results.

## Scope

Input frame: `tomato_sauce`, `libero_object`, state `0`, clean trajectory absolute step `78`.

Primary target: exact token `31744`, classified by the production execution decoder as `CLIP_MEDIATED_OPEN`.

Primary objective: `autoregressive_prefix_gripper_target_token_cw_v1`.

Primary comparison:

`TRUE_PGD_FINAL` vs `PGD_DELTA0`, `RAND20`, and `SHUFFLED_GRAD_PGD20`.

Legacy Tomato conditions are historical context only and are not part of the fixed-frame superiority claim.

## Required Gates

1. Clean input reproduction must freeze the raw pre-processor observation and processed tensor hashes.
2. Zero-step preflight must compare manual surrogate score path against official `generate(..., output_scores=True)` for `CLEAN` and `PGD_DELTA0`.
3. TRUE_PGD must prove strict route metadata: no fallback, `TokenPrefixPGDAttacker`, `adv_inputs` decode path, `20` backwards, exact `7` generated tokens, and valid processor-space budget.
4. Surrogate margin must improve from `PGD_DELTA0` to `TRUE_PGD_FINAL`.
5. Official target margin must improve from `PGD_DELTA0` to `TRUE_PGD_FINAL`.
6. TRUE_PGD official target margin must exceed both `RAND20` best and `SHUFFLED_GRAD_PGD20`.
7. Actual generated arm prefix must match the clean generated prefix in at least `5/6` positions.

## Result Classes

- `FULL_TOKEN_FLIP`: official gripper token changes from `31872` to `31744`.
- `MARGIN_ONLY_TRANSFER`: official target margin improves but token does not flip.
- `SURROGATE_TO_OFFICIAL_TRANSFER_FAIL`: surrogate improves but official margin does not.
- `RANDOM_NOT_BEATEN`: official TRUE_PGD margin does not exceed the random controls.
- `ARM_NONSELECTIVE`: route and margin pass, but actual arm-prefix gate fails.
- `INFRA_INVALID`: any route, budget, exact-token, score-invariant, or provenance assertion fails.

Only `FULL_TOKEN_FLIP` or `MARGIN_ONLY_TRANSFER` with random controls beaten permits the fixed-frame panel.

## Allowed Claims

- The runner is designed to test fixed-frame official-token transfer for target token `31744`.
- Passing this canary would support a fixed-frame margin effect only, not a closed-loop or task-level Layer3 claim.

## Forbidden Claims

- No task effect is claimed here.
- No closed-loop critical-closure disruption is claimed here.
- Historical Tomato legacy behavior is not treated as true-PGD evidence.
- Surrogate-only improvement is not treated as official generation improvement.
- A result that does not beat `RAND20` and `SHUFFLED_GRAD_PGD20` is not `VIS > random`.
