# Stage X X1R2 Q3R2 Engineering Fixture Pool — 2026-08-20

## Decision

`STAGE_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_FROZEN`

This is an outcome-blind engineering pool. Every selected identity is
permanently excluded from every scientific population and cannot become X1R2
evidence.

## Mechanical construction

- source universe: 1200 rows from
  `reports/STAGE_X_X1R_T1D0R_G10_IDENTITY_EXCLUSION_LEDGER_V1.json`
- source SHA256:
  `eadad7440ee67b9aeff0bb73ceab7af3b27fc1cf1ccb6e4bebf953660f75a015`
- G10 fresh after its existing exclusion union: 210
- Stage-X1 scientific parent exclusion ledger: 39 rows
- post-exclusion engineering universe: 171
- deterministic salt:
  `STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1_20260820`
- order: `sha256(salt|canonical_parent_key)`, then canonical key
- selected: 12 per suite, 48 total

The explicit historical Q3/Q3-AR keys are also excluded, including Q3-F01,
Q3-F02, Q3-F03, Q3-F04 and Q3-AR-F01. No replacement, reranking, or outcome
filter was used.

## Sealed artifact

- report:
  `reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json`
- report SHA256:
  `4a703459336c7fa1c93d7e8dc7fe6c9391ac9c3c2986bd5bf443d083ef7fa0cb`
- suites: `libero_10=12`, `libero_goal=12`, `libero_object=12`,
  `libero_spatial=12`
- selected rows: all `permanent_exclusion=true`, `scientific_use=false`,
  `outcome_read=false`

Eval160 and protected evaluation remain `UNREAD`; protected counters remain
zero. No model inference, clean rollout, simulator step, PGD, physical
intervention, V_phys read, or attack outcome read occurred.

## Next legal gate

Proceed to Q3R2-C. Scan only this frozen pool in the sealed order using the
current runtime. For each suite, the first fixture that independently obtains
current-runtime clean success, valid current emit, and legal `[t_emit,t_emit+4]`
window may be the suite's determinism fixture. Selection must not use Student
score, attack result, V_phys, or any scientific outcome.
