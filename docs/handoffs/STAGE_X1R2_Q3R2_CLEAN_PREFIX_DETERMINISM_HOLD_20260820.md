# Stage X1R2 Q3R2-C handoff — clean prefix determinism HOLD

Status: `OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED`

This is an engineering-only hold. No PGD, attack arm, physical intervention,
V_phys read, protected read, or scientific population selection occurred.

## Binding

- Repository source: `af8dbcf6c1a8f22bd80b974dc5836a9a0d2e724b`
- Repository tree: `077b9bcaa41d7e312cf509f01b4122dced98142a`
- Runtime authority source: `85fa8e678ca599f21f5a69d180c7179f9ef99478`
- Runtime authority tree: `f6555a5d49dda45f29ef64ca8ae4b65b7b08d3f9`
- Durable root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r2_clean_determinism_20260820`

The official aggregate auditor was invoked after the authorized stop and
returned `MISSING_SUITE_REPORT:libero_10`; therefore no aggregate PASS is
claimed.

## Suite closure

| suite | result | frozen evidence |
|---|---|---|
| `libero_10` | report missing after stop | first scanned `libero_10/task_08/state_44`; clean failure; 520 policy steps; first emit 138; protected counters zero |
| `libero_goal` | HOLD | `libero_goal/task_02/state_37`; emit 63/63; prefix mismatch at step 13, only `raw_agentview` and processor-pixel hashes differed; action, robot state, and direct tokens matched |
| `libero_object` | PASS | `libero_object/task_01/state_34`; emit 86; prefix 87 and full trace exact |
| `libero_spatial` | HOLD | `libero_spatial/task_09/state_29`; emit 68/65; prefix mismatch at step 14, only `raw_agentview` and processor-pixel hashes differed; action, robot state, and direct tokens matched |

The goal and spatial mismatches are currently classified exactly as
`CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED`. The fact that the first observed
differences are isolated to rendered visual hashes does not waive the frozen
exact-prefix gate.

## Boundary and stop

- Q3R2-D engineering matrix: not authorized.
- X1R2 scientific population: not selected.
- PGD/RAND/SHUFFLED: not run.
- Physical intervention and `V_phys`: not run/read.
- `Eval160`: `UNREAD`.
- Protected evaluation: `UNREAD`.

Do not rerun or replace the exposed engineering fixtures. Any repair or
prospective relaxation of the exact visual-prefix requirement requires a new
owner/PI review.
