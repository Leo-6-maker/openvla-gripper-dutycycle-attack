# STAGE X X1R T0 PGD / Token Authority Audit Handoff

Status: `STAGE_X_X1R_T0_REVIEW_REQUIRED`

Derived audit outcome: `STAGE_X_X1R_T0_HOLD_TEACHER_FORCED_ROW_PARITY`

No real PGD, attacked action, `env.step`, physical intervention, V_phys read,
or protected read was executed. Existing attack code and historical artifacts
remain immutable. No helper repair or V2 authority was introduced.

## 1. Identity and source binding

- repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- branch: `codex/stage-x-x1r-t0-pgd-alignment-20260818`
- parent: PR #121 head `a11d042d255100a87b38c279df893d3f9f685375`
- clean-forward runtime source: commit `7389331ac88c3d5b884b8f8551314d9a400e0217`, tree `9b7b901fb64ba71ad56182037fe6a04c5535916e`
- final audit-code source before evidence sealing: commit `a403a2dea4ba24e7536c74d5f11da3747437f997`, tree `8be140a128350f6244ae38432082b97d07880dc0`
- official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- canonical tokenizer: `action_tokenizer.py`, SHA256 `fdc98fcbf5b0926ef2181db71946d23ffbfa052cf8443dc933d52c42a191352c`
- common suite decoder SHA256: `2e672e75958205b05f40f4cd2467d3763b8e36eb2728289cd055c54213338e85`

## 2. Existing PGD standardization verdict

The frozen project implementation is source-aligned for most numerical and
strict-route primitives, but it is not a prospective X1R authority. The
project action-to-token helper is not equivalent to the checkpoint-local
native `ActionTokenizer` at endpoint/bin-edge cases. The clean-forward audit
also found a reproducible cache/full-forward numerical near-tie in one arm
row. These are separate findings.

## 3. Component audit A-AH

| Components | Result |
|---|---|
| A-B preprocessing and prompt | source/config aligned; clean canary required |
| C-D input and attention-mask handling | source aligned; processor-bound only |
| E-F clean pixels and victim dtype | clean parity verified; synthetic dtype audit passed |
| G-H target/physical OPEN semantics | canonical semantic mapping verified |
| I, AG action-to-token authority | `DEFECT_FOUND`; nearest-center helper differs from native endpoint semantics |
| J token-to-action decode | clean-only verified; native endpoint mapping is not bijective |
| K causal row indexing | formula and toy 7/7 pass; clean evidence has one stable parity hold |
| L-O CW objective, loss/update sign, gradient | nonzero synthetic audit passed |
| P-R epsilon, step size, iteration count | frozen config/source aligned; no attack execution authorized |
| S-T random start and temporal carry | frozen config/source aligned; no direct runtime attack test |
| U-X fp32 master, projection, cast, correction | synthetic audit passed for fp16/bf16 |
| Y best-iterate selection | `FROZEN_PROTOCOL_GAP`; existing code records metric but uses final iterate |
| Z adversarial re-decode | strict source contract; no attack execution |
| AA-AD target validation, strict route, fallback/direct-actuator prohibitions | source/negative route tests passed; fallback forbidden |
| AE GPU/runtime provenance | clean-only worker receipts valid; foreign workloads untouched |
| AF historical victim identity | launch-time identity not identifiable |
| AH historical generated-token identity | direct historical generated IDs not identifiable |

## 4. Reusable primitives

Reusable only under a future versioned authority: official preprocessing,
prompt/input construction, native checkpoint tokenizer, native decoder,
OPEN semantics, strict `TokenPrefixPGDAttacker` route, CW loss, targeted sign
update, fp32 master projection, dtype-budget correction, re-decode contract,
and GPU receipt schema. The old helper is not reusable as token authority.

## 5. Test coverage

Before T0, the exact PGD primitives had no direct unit coverage. T0 added
targeted tests for endpoint differential behavior, all seven row indices,
fp16/bf16 projection, strict fallback rejection, nonzero CW descent, and the
no-execution audit path: `6 passed`.

## 6. Historical generated-token evidence

`historical_generated_token_ids = NOT_IDENTIFIABLE`. Historical artifacts
contain reference/re-encoded token fields, not an immutable direct record of
the token IDs emitted by the historical model generation call.

## 7. Correct interpretation of 31744 / 31745

The observed Q00 `31744` versus `31745` is not evidence of a PGD gradient
failure. Native encoding uses 256 endpoint bins and can emit endpoint token
`31744`; native decoding clips that endpoint to the final 255-bin center, and
re-encoding that center returns `31745`. Thus the token-to-center map is
non-bijective. The project nearest-center helper is independently defective
on general edge/nextafter cases, but it agrees with the eight Q00 reference
raw actions.

## 8. Tokenizer differential census

General canonical-vs-helper mismatches:

- `libero_10`: 2,927
- `libero_goal`: 3,136
- `libero_object`: 3,052
- `libero_spatial`: 2,843

The eight existing Q00 raw/reference rows are canonical/helper exact: `8/8`.

## 9. Causal-LM row audit

The tested 7-dimensional row formula is `[-8,-7,-6,-5,-4,-3,-2]`, with
gripper at `-2`. The synthetic teacher-forced audit is `7/7`. Clean forward
checked `168` actual rows (`4 suites x 3 replicas x 2 stages x 7 dims`):
`165` exact, with three identical failures at `libero_goal / stage_v / dim=3`.

The reproducible mismatch is a numerical near tie, not a row-index shift:
autoregressive top1 `31918` / `26.375`, top2 `31932` / `26.25`; full
teacher-forced top1 `31932` / `26.25`, expected `31918` / `26.0`.

## 10. Numerical primitive audit

- CW loss: `1.0 -> 0.7999999523` after sign descent; gradient `[-1, 1]`.
- update rule: `adv -= step_size * sign(gradient)`; PASS.
- master tensor: fp32; projection invariant: PASS.
- fp16 cast: actual L-inf `0.0`, correction count `2`; PASS.
- bf16 cast: actual L-inf `0.099609375`, correction count `1`; PASS.

## 11. Clean-forward determinism

Twelve valid workers completed: four suites, three fresh processes per suite,
two physical GPUs, one project worker per GPU. Autoregressive token sequences
were deterministic and processor input/attention/pixel parity was exact for
all rows. The only aggregate failure is the three repeated teacher-forced
row mismatches described in section 9.

## 12. Versioned repair

None. The historical helper and production PGD path were not modified. A
future canonical adapter and a decision for cache/full-forward parity require
owner review and a new prospective authority.

## 13. Validation and CI

Local targeted T0 tests: `6 passed`. `py_compile` and `git diff --check`
passed. The draft PR must run the repository CI before review; green CI is
engineering evidence only and does not authorize X1R.

## 14. Protected counters

All T0 worker and CPU reports record zero: PGD calls, env steps, physical
interventions, V_phys reads, attack-outcome reads, protected reads, and
Eval160 reads. `Eval160=UNREAD`; protected evaluation=`UNREAD`.

## 15. Final T0 state

`STAGE_X_X1R_T0_REVIEW_REQUIRED`, with the specific clean-forward hold
`STAGE_X_X1R_T0_HOLD_TEACHER_FORCED_ROW_PARITY`. The tokenizer conclusion is
`PGD_CORE_ALIGNED_WITH_PROJECT_HELPER_EDGE_DEFECT_AND_HISTORICAL_TOKEN_NOT_IDENTIFIABLE`.

## 16. Explicit non-authorization

X1R targeted PGD is **not executed or authorized**. No fresh V_phys/M4,
attacked image/action, physical intervention, X2, timing matrix, Eval160, or
protected evaluation may start from this handoff. Next action requires owner
review of the clean causal-row parity hold and any future versioned token
authority; no rerun-to-pass is permitted.

