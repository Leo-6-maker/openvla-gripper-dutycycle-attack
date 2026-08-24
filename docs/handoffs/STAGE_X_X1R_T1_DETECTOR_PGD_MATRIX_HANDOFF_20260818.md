# Stage X X1R T1 handoff — detector closure complete, attack-load authority HOLD

Status: `STAGE_X_X1R_T1_HOLD_ATTACK_LOAD_AUTHORITY`

This is the terminal state for the current autonomous T1 run. T1-A, T1-B,
and T1-C passed clean-only audits. T1-D stopped before PGD or environment
execution because no outcome-blind X1R attack load is frozen.

## Source and ancestry

- stacked branch: `codex/stage-x-x1r-t1-detector-pgd-matrix-20260818`
- current HEAD: `26e45707447aff46fa2b88eba372752e6cd3a4c6`
- current tree: `7b671acb44cea682555e60c99c0562b364502a97`
- base PR: `#122`, immutable base head `6160d3b47138166a9159453d463ac062f8df4f95`
- official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- server clean run worktree: `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-x-t1-detector-pgd-matrix-20260818-r7`
- server worktree status: clean

PR #122 and historical T0/PR #121 artifacts were not modified.

## Gate results

### T1-A — token and score-path authority: PASS

Receipts:

- native audit: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_detector_authority_20260818T034500Z/NATIVE_ACTION_TOKEN_AUTHORITY_AUDIT_V1.json`
- aggregate score audit: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_detector_authority_20260818T034500Z/SCORE_PATH_AUTHORITY_RECEIPT_V2.json`

Native V2 parity is `28740/28740` across four suites. The old helper remains
`HISTORICAL_COMPATIBILITY_ONLY`; its boundary mismatches are retained as
forensic diagnostics, not silently repaired. All eight clean score receipts
(4 baseline plus 4 fresh cross-GPU processes) bind official preprocessing,
manual cached autoregressive rows, target-token gradients, and arm-preservation
gradients. Cached rows are exact to official greedy rows. The no-cache route is
characterized and is not the prospective authority. Iterate selection remains
`final_iterate_only`.

### T1-B — frozen Teacher → Student detector authority: PASS

The detector is reused, not retrained:

- final decision: `PASS_VALID_NEGATIVE_CONCLUSION`
- scientific decision: `STUDENT_HELDOUT_GENERALIZATION_NOT_ESTABLISHED`
- Student promotion: false; frozen Student retained for record
- checkpoint SHA256: `e24d00ca30c8fe0d5ef066e90872f010556bfabec13f78d4275962c6b35ca227`
- model source: `n5/phase3_student/n5_student_model.py`, SHA256 `7945e464130d8bcb4b4fd85475dffc3024ef2bc419b5d1ea812b29156b96a0fd`
- canonical D8 adapter SHA256 (LF): `2b71f3f29ac3352bf945fdfe1e171013d5b6b598d8bcbd451836acb0b8ddcae3`
- canonical 25D feature source SHA256 (LF): `c3c96090498d9d2da2047215387fee7d94a8a4b2782f2b00f9a551e3344311b9`
- normalization SHA256: `66e24b18a8fa5e46eca41bcdfa8b8aff7c9d05feeb6fcce8d6a62193a469fd6c`
- threshold SHA256: `5236884ea746adc6a5388ab6e8058adb717b86ee2814b71a748c7792d051de3e`
- scheduler: frozen `H_phys=10`, `T5=5`, physical threshold `0.55`, closing threshold `0.80`, one-shot

The old N4 runtime adapter is explicitly rejected as stale 51D/old-checkpoint
authority and was not consumed. The active clean shadow binds D8 V3 directly
to the frozen N5 checkpoint.

### T1-C — clean online shadow: PASS

Receipt:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_detector_authority_20260818T034500Z/DETECTOR_AUTHORITY_RECEIPT_V1.json`

Four suite-matched clean score receipts and four deterministic clean replay
parents passed. The shadow uses only raw/env action, EEF position, and gripper
qpos; velocity is reconstructed exactly as the existing canonical telemetry
loader does. Prefix parity spot checks use the frozen numeric tolerance
`1e-6`; repeated probabilities are identical and scheduler first-emits are
identical. `libero_object` produced a legitimate no-emit observation and it was
retained, not filtered or replaced.

## T1-D — explicit HOLD

Receipt:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x_t1_detector_authority_20260818T034500Z/ATTACK_LOAD_AUTHORITY_HOLD_V1.json`

The current T1 protocol declares:

`status=HOLD_ATTACK_LOAD_AUTHORITY`, `pgd_authorized=false`.

The audit found incompatible historical candidates:

- Stage IX canonical: `epsilon=0.10`, `step_size=0.020`, `PGD-20`, `cw_margin=5.0`, single-L10 F0 contract;
- FEC experimental canaries: `epsilon=0.03`, `step_size=0.006`, `num_steps=5`, with legacy target/objective variants;
- SC5 historical contract: different processor epsilon and target semantics.

The T1 suite-matched victim/tokenizer contract does not itself freeze which
one is the X1R load, nor does it freeze a fresh X1R parent population. No load
was selected by appearance or prior outcome.

## Protected boundary and next legal action

All counters remain zero:

`pgd_calls=0`, `env_step_calls=0`, `attack_outcome_reads=0`,
`physical_interventions=0`, `vphys_reads=0`, `eval160_reads=0`,
`protected_reads=0`.

`Eval160=UNREAD`; protected evaluation is `UNREAD`.

The next legal action requires owner/GPT review and a fresh append-only X1R
authority that freezes, before any outcome is read:

1. one suite-matched attack load and objective;
2. epsilon, step size, PGD steps, projection/dtype rule, random-control rule;
3. target-token semantic binding for every suite;
4. fresh nonprotected parent identities and arm population;
5. the new protocol/source/provenance hashes.

After that authority is independently audited, T1-D may resume. Until then,
do not run PGD, `env.step`, physical intervention, timing matrix, X2, Eval160,
or protected evaluation.
