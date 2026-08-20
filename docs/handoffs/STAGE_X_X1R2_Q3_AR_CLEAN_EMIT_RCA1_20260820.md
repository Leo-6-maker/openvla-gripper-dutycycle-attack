# STAGE_X1R2 Q3-AR Clean Emit RCA1

Status: `PASS_RCA1_STOP_OWNER_REVIEW`

This is a static and sealed-artifact offline RCA only. It produced no new
OpenVLA inference, simulator environment, `env.step`, PGD, attack arm,
physical intervention, `V_phys`, Eval160, or protected read.

## Bound source and identity

- PR #135 audit input: HEAD `404b3793fd103bdcff269502561470c3e6cca13f`, tree `9b94be132fbe4a987a2dc8073d29fca1fac1e9b5`.
- Identity: `M012`, ordinal 10, `libero_10/task_09/state_43`, seed `1436562779`.
- Historical runtime receipt: D1R commit/tree `d74b8b7aff311c4ebbd51bf83ff026efe48d0236` / `2ee7425fc9177d70abb61f12b644833ec20d0a06`; receipt SHA `cbe1bd28968bb5b30bdd7622209675edcc4ad0425e9cf749189f3c537f968b25`; sealed `first_emit_step=133`.
- Q3-AR runtime receipt: commit/tree `b7237611c466077a9a7e6f0b1102e9176cfa2c88` / `fd5eeef98480b4c608ebd4eafb8e325afa8cd17a`; receipt SHA `ed7143d6bd1c50655fb5b61295ca2ed9ced965069251b366a3ad154600912e38`; observed `first_emit_step=null`.
- The clean runner Git blob is identical in both source snapshots: `b17761a158aca448610c251d17843c658392479b`.

The historical `133` chain is mechanically closed through the D1R receipt,
per-step telemetry, D1R census row, and Q3-AR fixture expectation. It remains
authoritative only as the historical M012 clean-screening observation; it is
not a new attack authority.

## Offline result

The frozen Student contract is byte/semantically equivalent for the available
Git source, checkpoint, feature/adapter, normalization, threshold, and
scheduler artifacts. CPU-only Student replay was deterministic twice on each
sealed feature sequence:

- historical features: 250 rows, first emit `133`, repeat max difference `0`, sealed-trace max difference `0`;
- Q3-AR features: 256 rows, first emit `None`, repeat max difference `0`, sealed-trace max difference `0`.

The already-sealed trajectories are not equivalent:

- raw observation hash first differs at step 49;
- action token IDs first differ at step 111;
- qpos/E‑EF/25D features/Student probabilities first differ at step 112;
- candidate-close first differs at step 132;
- historical terminates at step 249, Q3-AR at step 255.

At step 133 both traces are legal and candidate-close. Historical
`gripper_closing_state=0.8626471758` passes `0.8`; Q3-AR
`gripper_closing_state=0.6740048528` does not. The same frozen scheduler
therefore emits only in the historical trace.

## RCA classification

- Primary: `CLEAN_TRAJECTORY_DRIFT`.
- Secondary: `UNRESOLVED_MULTI_FACTOR` for the origin of the first observation
  divergence and incomplete launch-time victim/processor binding.
- Not supported: expectation provenance error, Student artifact/runtime drift,
  or step-index/reset contract drift.
- Student stochasticity is not claimed.

The historical external Student copy has a different sealed current-server
hash (`ceb761...`) than the frozen Git source (`30cf...`), but the runner
imports the Git source. Historical launch-time external-copy identity and the
full clean OpenVLA/victim checkpoint/processor binding are therefore recorded
as `NOT_IDENTIFIABLE`, not silently upgraded to parity.

## Immutable boundaries

- Q3-AR-F01 remains a permanently excluded engineering fixture.
- Q3-F01 remains immutable runtime-invalid; Q3-F02–F04 remain sealed/not-started.
- Arm-isolation repair remains `UNQUALIFIED`.
- No scientific X1R2 population was selected.
- Eval160 and protected evaluation remain `UNREAD`.
- RCA1 counters for new OpenVLA inference, simulator steps, PGD, attack,
  physical intervention, `V_phys`, Eval160, and protected reads are all zero.

The machine-readable evidence is in
`reports/STAGE_X_X1R2_Q3_AR_CLEAN_EMIT_RCA1_V1.json` (SHA256
`7a3dfc047c1040b2b72989cd215adb75c6531b0c38df9ecd4288a5cad10b8214`). Next gate is
`OWNER_REVIEW_RCA1_ONLY`; no rerun or engineering repair is automatic.
