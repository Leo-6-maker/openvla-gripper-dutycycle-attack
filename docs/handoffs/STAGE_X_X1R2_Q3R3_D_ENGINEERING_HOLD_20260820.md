# Stage X1R2 Q3R3-D engineering hold — 2026-08-20

Status: `HOLD_Q3R3_D_ENGINEERING_MATRIX`

This is an engineering-only hold. It is not attack efficacy evidence, not a
scientific population result, and not a V_phys result.

## Immutable bindings

- D execution source commit: `3225c1cbeab4cdf481f5ad140fc277c0dcad6c9a`
- D execution source tree: `400471bfb12651ae0f40153b4f49a3145508ef7b`
- D protocol SHA256: `019b0203dcb4eb0973fdea18377dd63531ebabfe2e20eedb3deccfac1a9f821d`
- Q3R3-C root seal SHA256: `9159b6601c856b45fee13e9e1dfb689cd7194949fd0094f6951d54e3fe19b698`
- D durable root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_d_20260820`
- official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

The first D launch stopped before model exposure because one frozen suite-report
SHA was one character short in the protocol. The corrected retry was then run
on the source binding above. The original logs remain under the D root.

## Retry observations

The retry used four workers on physical GPUs 0, 1, 2, and 3. Before launch the
strict free-memory values were 74470, 81210, 26772, and 27440 MiB respectively;
all satisfy `free_memory_mib > 20480`. No foreign process was signalled,
paused, killed, migrated, or otherwise modified.

- `libero_10`, `libero_goal`, and `libero_object`: `CLEAN_ENGINEERING` arm
  completed, then the first true-PGD arm failed in the canonical shared
  `TokenPrefixPGDAttacker` strict-candidate path with:
  `AttributeError: 'NoneType' object has no attribute 'detach'`.
  The failure occurs because the target-token objective does not populate
  `region_token_ids`, while the strict candidate audit dereferences it as the
  native-open token set. No attacked `env.step` was reached.

- `libero_spatial`: deterministic random-time prefix replay produced equal
  simulator branch state at `atol=1e-12, rtol=0`, but the two replayed
  agentview byte arrays were not equal. The random-time common-observation
  contract therefore failed before model loading for that suite.

The partial receipts are preserved. The aggregate audit is:

`STAGE_X1R2_Q3R3_ENGINEERING_MATRIX_AUDIT_V1 = HOLD_Q3R3_D_ENGINEERING_MATRIX`

No D root PASS seal was created.

## Protected boundary

Across the failed retry receipts:

- physical interventions: `0`
- V_phys reads: `0`
- attacked env steps: `0`
- protected reads: `0`
- Eval160: `UNREAD`
- protected evaluation: `UNREAD`

All four project workers were stopped after the hold; no project worker remains
running. The final GPU snapshot still showed the foreign workloads untouched.

## Next gate

`OWNER_REVIEW_Q3R3_D_ENGINEERING_HOLD`

Do not rerun the engineering fixture, start R0/R1/R2, select an X1R2
scientific population, run physical intervention, read V_phys, or read
Eval160/protected evaluation until the owner reviews and explicitly authorizes
the shared strict-candidate repair and a new deterministic visual replay
contract.
