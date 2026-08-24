# Stage X1R2 Q3R3-C handoff

Status: `STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS`

Q3R3-C is engineering qualification only. It did not execute PGD, RAND,
SHUFFLED, physical intervention, V_phys, attack outcome reads, Eval160, or
protected evaluation. Historical `OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED`
remains immutable and was not relabeled.

## Execution binding

- Runtime execution source: commit `3df84eaf0a4a137eaaee05b58974088463ffacff`, tree `bb18054d9e2251feb8ee5472253777ee36013893`.
- Append-only receipt repair/audit source: commit `a6bf6f654a6b4f065398cf940fe024234f7ab907`, tree `413ee30245ac262b359107305e167149adf1bd2e`.
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
- Durable root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_c_20260820`.
- Root seal SHA256: `9159b6601c856b45fee13e9e1dfb689cd7194949fd0094f6951d54e3fe19b698`.

## Qualification closure

Each suite selected the first current-runtime clean-success + valid-feature +
one-shot-emit + legal-window fixture in the existing permanently excluded
Q3R2 engineering pool order. No exposed identity was reused and no
replacement was performed.

| suite | scanned | selected | t_emit | repeats | state | direct tokens | follow-up |
|---|---:|---|---:|---:|---|---|---:|
| libero_10 | 4 | `libero_10/task_06/state_38` | 68 | 2 | exact | match | 15 |
| libero_goal | 5 | `libero_goal/task_04/state_39` | 53 | 2 | exact | match | 15 |
| libero_object | 8 | `libero_object/task_04/state_46` | 62 | 2 | exact | match | 15 |
| libero_spatial | 3 | `libero_spatial/task_08/state_35` | 64 | 2 | exact | match | 15 |

All eight branch receipts passed the fixed state contract (`atol=1e-12`,
`rtol=0`), pre-branch OpenVLA/Student calls were zero, and the first clean
direct token sequence matched the reference. The original receipts had one
known success-path schema omission (`protected_reads`); eight derived zero
fields were added in an append-only repair manifest, with raw receipt SHA256
bindings and `raw_receipts_unchanged=true`.

## Resource and protected boundary

Workers used GPUs 0–3, one project worker per GPU. Each launch gate required
strictly more than 20480 MiB free; foreign processes were not killed,
paused, signaled, reniced, or migrated. After completion the project workers
exited and the external compute PIDs remained present.

Protected counters are all zero; `Eval160=UNREAD` and
`protected_evaluation=UNREAD`.

## Next legal gate

`STAGE_X1R2_Q3R3_ENGINEERING_MATRIX` is now authorized by the frozen
Q3R3-A/B/C plan. It remains engineering-only and must preserve the same
reference-clean/action-prefix/common-observation/state contract. No
scientific population is selected by this result, and no attack efficacy
claim may be made from Q3R3-C.
