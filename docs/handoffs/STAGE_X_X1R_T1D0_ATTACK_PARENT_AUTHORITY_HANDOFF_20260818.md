# Stage X X1R T1-D0 — attack-load and fresh-parent authority freeze

Status: `STAGE_X_X1R_T1D0_HOLD_TIMING_ANCHOR_AUTHORITY`

This stacked change is static/authority-only. It does not load a model, start a
clean rollout, optimize a perturbation, call `env.step`, read `V_phys`, or read
an attack result.

## Source boundary

- parent PR: #124
- base commit: `c6a4c5a9e7d63121a75814b3071c9047e1d9e0d0`
- base tree: `aa22ae95ed760a32cde01729c47b40b3331f668a`
- branch: `codex/stage-x-x1r-t1d0-attack-parent-authority-20260818`
- official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

The live GitHub head/tree of this stacked PR is deliberately reported
separately from the runtime base binding above.

## Frozen numerical load

T1-D0 freezes only the pre-frozen Stage IX numerical budget:

```text
epsilon=0.10
step_size=0.020
num_steps=20
cw_margin=5.0
random_start=false
temporal_init=prev_delta
temporal_smoothing_lambda=0
master_dtype=fp32
iterate_selection=FINAL_ONLY
```

Stage IX victim identity, tokenizer authority, fixed global token IDs, and
historical outcomes are not inherited. Future X1R must use the suite-local
native ActionTokenizer authority from T1. The later primary validity rule also
requires arm token IDs `[0:6]` to remain equal to the clean arm token IDs; the
direct-generated gripper token must belong to the native OPEN class.

The frozen future controls are `CLEAN`, `TRUE_PGD`, `RAND_UNIFORM`, and
`SHUFFLED_GRAD_20`. TRUE and shuffled controls share the numerical budget and
final-iterate rule.

## Timing anchor hold

The frozen scheduler receipt binds `T5=5`, `H_phys=10`, one-shot execution, and
the emit eligibility rule. It does not contain an explicit attack execution
anchor. Therefore `emit+5` is not inferred from `T5=5`; a future authority must
bind the exact attack-start relation to an immutable scheduler/protocol field.

## Parent authority result

The candidate universe and all exclusion inputs were frozen before any new
clean rollout. The deterministic order is:

```text
SHA256(STAGE_X_X1R_T1D0_PARENT_AUTHORITY_V1_20260818::canonical_parent_key)
then canonical_parent_key
```

Using the bound G10 non-protected universe (1,200 identities) and the union of
prior clean-attempt, exposure, physical-intervention, Stage V formal/matrix,
rejected-candidate, and Stage VI-B2 exclusion sets leaves 210 candidates. The
first-per-task selection yields 39 identities. `libero_goal/task_01` has zero
remaining candidates, so the required 40-parent authority cannot be honestly
sealed. No replacement or exclusion relaxation was performed.

This is a static population hold, not a scientific outcome. The future funnel
remains frozen as:

```text
IDENTITY_FROZEN
→ CLEAN_ROLLOUT_MATERIALIZED
→ CLEAN_SUCCESS / CLEAN_FAILURE
→ EMIT / NO_EMIT
→ LEGAL / ILLEGAL_HORIZON
→ ATTACK_ELIGIBLE
```

## Authorization and protected boundary

```text
pgd_authorized=false
env_step_authorized=false
physical_intervention_authorized=false
attack_outcome_authorized=false
next_gate=CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED
```

`Eval160` remains `UNREAD`; protected evaluation remains `UNREAD`; all attack,
environment-step, intervention, `V_phys`, and protected counters remain zero.

T1-D0 stops here for owner/GPT review. No X1R PGD or clean parent rollout is
authorized by this PR.
