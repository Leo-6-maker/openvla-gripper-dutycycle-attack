# STAGE X X1R2 Q3R3 E2 pre-attack frozen-pool hold — 2026-08-20

## Decision

Status: `HOLD_Q3R3_E2_FEASIBILITY_EVIDENCE_INSUFFICIENT_FROZEN_POOL_EXHAUSTED`

This is a pre-attack authority/evidence hold. It is not a strict-method
infeasibility result, not a negative attack result, and not an efficacy claim.
No E2 TRUE candidate audit, OpenVLA inference, simulator run, PGD call, or GPU
worker was started.

## Static pool audit

The frozen Q3R2 engineering pool contains 12 permanently excluded identities per
suite. The exact union of the Q3R2 already-exposed binding, sealed Q3R3-C and
Q3R3-C2 scan lists, and the D2 selected identities was checked against the pool
JSON. D2 selections are already present in the C2 selected rows and therefore do
not consume a new identity.

| suite | frozen pool | consumed | never-started remaining |
| --- | ---: | ---: | ---: |
| `libero_10` | 12 | 7 | 5 |
| `libero_goal` | 12 | 8 | 4 |
| `libero_object` | 12 | 12 | 0 |
| `libero_spatial` | 12 | 6 | 6 |
| **total** | **48** | **33** | **15** |

The object suite is exhausted exactly: `Q3R2-LIBERO_OBJECT-01..12` all have
durable prior exposure evidence. Therefore the frozen-pool-only E2 four-suite
prospective gate cannot be completed as authorized: the required object suite
has no legal never-started identity, while V4 forbids a top-up or replacement
pool.

## Evidence bindings

- Current PR source binding at audit: `782a00db94cf0644ebec96caf4134dc3b2187ff7`, tree `15303ec92b0c39e69d7c9a62d04880cf1798c5a4`.
- Frozen pool raw SHA-256: `4a703459336c7fa1c93d7e8dc7fe6c9391ac9c3c2986bd5bf443d083ef7fa0cb`.
- Q3R3-C audit/root seal: `ddd25e2eb4c7c377d4765174b9d276fa29dbe5149c3e9ef6dbf2b98503cfd31d` / `9159b6601c856b45fee13e9e1dfb689cd7194949fd0094f6951d54e3fe19b698`.
- Q3R3-C2 audit/root seal: `ca8f5075073b04ee8db4ff9c83faa7382ed90aecffb3f7b21e6fbf5b2721f571` / `fe23991c94852bc65269d740a6c67f782350dc61e600f766fdfbf85461abbdcc`.
- Q3R3-D2 audit: `daa74f4279ac4672adc7b233b2c7c72ea444aa9e13f24c0d4c03b8c2fb6617ca`.

The complete per-suite IDs and canonical parent keys are in the companion
`STAGE_X_X1R2_Q3R3_E2_PREFLIGHT_POOL_AUDIT_V1.json`.

## Boundary and next action

The frozen method and estimand remain unchanged. Do not run the 15 remaining
non-object identities, because that would consume engineering identities while
leaving the authorized four-suite gate impossible to finish. Do not create a
new outcome-ranked/top-up pool, use scientific/protected identities, tune the
attack, weaken the exact-arm gate, or enter R0/R1/R2.

Owner/PI review is required for either a new frozen engineering pool or an
explicit authority change. If later authorized, the A800 environment remains
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`; use at most one
worker per physical GPU and at most eight workers, with free memory above 20 GiB
checked immediately before launch and all foreign GPU processes left untouched.

Mandatory stop: `STOP_BEFORE_E2_GPU_OR_MODEL_EXECUTION`.
