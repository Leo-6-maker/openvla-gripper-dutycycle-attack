# Stage X X1R victim provenance closure

Status: `STAGE_X_X1R_HOLD_CLEAN_FORWARD_TOKEN_PARITY`.

The suite-level contract and processor parity are closed, but the independent
clean-forward audit found three token-identity mismatches in 8 canary rows.
This branch therefore does not reach `STAGE_X_X1R_PROSPECTIVE_VICTIM_CONTRACT_READY`
and does not authorize X1R.

This stacked PR is based on `codex/stage-x-x1r-protocol-repair-20260817` and
does not modify PR #120. PR #120 remains the immutable forensic-audit PASS.

## Scope

The only runtime permitted by this handoff is clean/no-op model inference on
immutable Stage V and Stage VI-B2 Q00 snapshots. The worker does not optimize
pixels, create attacked actions, call `env.step`, read V_phys, run physical
intervention, or read Eval160/protected evaluation.

The prospective contract is
`configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json`. It binds one
current model/processor/action-semantic identity per suite across the proposed
clean policy, snapshot producer, future V_phys intervention policy, and future
PGD victim. The future roles are marked `NOT_EXECUTED`; this PR does not grant
X1R authority.

## Historical boundary

Stage V and Stage VI-B2 launch-time checkpoint weights were not recorded well
enough to identify exact historical weight identity. They are therefore
`NOT_IDENTIFIABLE`; current directory hashes are prospective bindings only.
Stage IX F0 and historical Stage X X1 remain immutable diagnostic evidence and
non-promotional for cross-suite PGD-to-V_phys inference. No historical row is
rerun or promoted.

## Current prospective bindings

The four suite-matched model directories exist under the official A800 server
layout. The contract records separate full tree, safetensors-weight, semantic
file, config, processor, tokenizer, and normalization-statistics digests. The
OPEN target is derived from each loaded suite checkpoint through the official
raw-action rule (`raw > 0.5` -> physical OPEN); no global token ID is assumed.

## Observed canary gate

The canary used 8 sealed Q00 snapshots (one Stage V and one Stage VI-B2 per
suite). Processor input IDs, attention mask, and pixel values after the
predeclared dtype cast were exact for 8/8. Clean decoded actions were within
`1e-6` for 8/8, with a maximum error below `3e-8`. Per-suite OPEN/CLOSE sets
were nonempty and disjoint.

However, clean generated action tokens were exact for only 5/8 rows. Three
rows differ only at the gripper token (`31744` generated versus `31745`
reference-derived), but X1R is token-level PGD, so this is not a harmless
numeric tolerance. The token gate is not widened after observation.

If historical identity remains unidentifiable, that is an explicit historical
boundary, not a reason to substitute current hashes into old results.

The terminal state after this PR is:

`STAGE_X_X1R_HOLD_CLEAN_FORWARD_TOKEN_PARITY`

X1R/PGD, fresh V_phys, X2, timing matrix, Eval160, and protected evaluation
remain unauthorized and unread. Historical launch-time checkpoint weights
remain `NOT_IDENTIFIABLE`; no current directory hash is used to rewrite that
boundary. Any repair or prospective token-equivalence decision requires a
separate owner review and must not read attack outcomes.
