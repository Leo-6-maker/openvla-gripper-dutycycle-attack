# STAGE X X1R T1-D0R — timing and fresh-parent authority repair

Status: `STAGE_X_X1R_T1D0R_STATIC_ONLY`

This stacked, append-only handoff is based on PR #125 HEAD
`2881722b51cc5365205a575eddb42f4d22456f09` / tree
`77a17257c609f5c98200697ebd7d83f309766bda`. PR #125 remains the immutable
T1-D0 HOLD artifact; this branch does not modify it.

## Scope and hard stop

The official runtime remains `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
This gate is CPU/static only. It does not load models, materialize clean
parents, run PGD, create adversarial inputs, call `env.step`, read V_phys or
attack outcomes, or access Eval160/protected evaluation. All authorization
flags and protected counters remain zero. Even a recomputation PASS stops at
`CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED`.

## Prospective timing freeze

Historical scheduler evidence is bound only for `T5=5`, `H_phys=10`, one-shot,
and the emit rule. It has no identifiable historical attack-start field.
The new prospective PI freeze is:

```text
timing_semantic_origin = NEW_PROSPECTIVE_PI_FREEZE_20260818
emit_step = t_emit
attack_start_step = t_emit
attack_window = [t_emit, ..., t_emit+4]       # 5 steps
physical_followup = [t_emit+5, ..., t_emit+14] # 10 steps
legal_horizon: t_emit + 5 + 10 <= episode_length
```

`NO_EMIT` remains `NO_EMIT`. `prev_delta` is zeroed at attack-window entry and
may carry only inside the same five-step window and condition/episode. TRUE
and SHUFFLED share the reset/carry boundary; RAND_UNIFORM has one deterministic
draw per frame and no gradient carry. The existing historical 10-step helper
is not relabeled as proof of this new timing; a future runner must bind this
contract before clean-parent materialization.

## Source-driven parent reconstruction

The auditor loads the bound G10 identity manifest and every bound exclusion
source, verifies the supplied SHA256/listing hashes, performs canonical
identity joins, derives the exclusion union, and ranks only the resulting
fresh identities by `SHA256(selection_salt::canonical_parent_key)`. It does
not consume success, detector score, emit, V_phys, attack, or protected fields.

The design is 40 nominal suite/task cells. No replacement is allowed. A cell
with no fresh identity is recorded as structural pre-execution missingness,
not as clean failure or attack failure. The machine-readable outputs are:

- `STAGE_X_X1R_T1D0R_SOURCE_RECOMPUTE_AUDIT_V1.json`
- `STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json` (JSONL)
- `STAGE_X_X1R_T1D0R_DESIGN_CELL_LEDGER_V1.json` (JSONL)
- `STAGE_X_X1R_T1D0R_PARENT_LEDGER_V1.json` (JSONL)
- `STAGE_X_X1R_T1D0R_ROOT_SEAL.json` and its SHA256 sidecar
- `STAGE_X_X1R_T1D0R_SHA256SUMS.txt`

The root seal separately records the reviewed PR #125 binding, the D0R source
commit/tree used by the auditor, all external input hashes, generated report
hashes, the prospective timing semantics, and the non-self-referential live
GitHub final HEAD/tree handoff field. The final pushed commit/tree is reported
after publication rather than falsely embedded in its own containing commit.

## Decision boundary

Expected successful static state is:

```text
STAGE_X_X1R_T1D0R_AUTHORITY_PASS
model_inference_authorized=false
clean_parent_materialization_authorized=false
pgd_authorized=false
env_step_authorized=false
physical_intervention_authorized=false
attack_outcome_authorized=false
protected_authorized=false
next_gate=CLEAN_PARENT_MATERIALIZATION_REVIEW_REQUIRED
```

Any missing or changed source, ambiguous identity join, directory-listing
change, timing-source mismatch, malformed ledger, nonzero protected counter,
or forbidden outcome read is a fail-closed T1-D0R HOLD. This handoff does not
authorize clean-parent materialization or X1R.
