# V5 R3 Fast-Closure Handoff

Date: 2026-07-29  
Branch: `codex/v5-student-learnability-r3-20260728`  
Execution source commit: `8fe7f507b7c828c97f29245019ae9642606444e0`  
Execution source tree: `f5c1911ceab981904cc5370ead19f1b6b06dfee9`

## Decision

`R3-1A = HOLD_INCOMPLETE_PUBLICATION_OR_WORKER_CLOSURE` for the live formal
FIT670 root.  The formal root was read only for metadata/seal reconciliation
and was not consumed as a development tranche.

`R3-1 CANARY = PASS_ENGINEERING_CONSUMABLE_INPUT_GATE` for the independently
sealed eight-episode canary.  `R3-2 = PASS_CONTRACT / DEVELOPMENT_NONCONSUMABLE`
for label generation and provenance closure.  `R3-3 = HOLD_COVERAGE`.

Student training, model inference, held-out development, and shadow execution
were not started.  The 8-episode canary is insufficient for learnability or
scientific claims.

## Environment and source

The required environment path was used:

`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

Its existing Python symlink resolved to
`/home/sz/miniconda3/envs/hallo/bin/python3.10`, Python 3.10.16.  The server
worktree was detached, clean, and checked out at the execution source commit.
No GPU task was started; all work in this handoff was CPU-only or metadata-only.

## R3-1A formal-root reconciliation

The audit output was created at:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_1a_reconciliation_5f7d73a32`

Observed in one live snapshot:

| Field | Observed |
|---|---:|
| published episode directories | 665 |
| valid sealed episodes | 665 |
| worker manifests | 7 |
| worker-unique identities | 586 |
| per-shard unique sum | 586 |
| missing worker | `gpu_1` |
| missing allowlist identities | 5 |
| staging residues | 1 |
| unallowlisted identities | 0 |
| bad episode seals | 0 |

The supplied historical counters 472 and 463 do not match this live snapshot.
The closure equalities therefore do not hold, and no formal identity was
selected from this root.

The audit recorded `metadata_only=true`,
`payload_semantics_read=false`, and `protected_reads=[]`.

Audit root seal: `SHA256SUMS=52002c20da1b7a73e026fa90d1fec1ebf61f56f36515212e262687ab58b82db6`; sidecar `96f2a9428e150aaf87b4f8606b97b7dfcd68c28aea784560e08ee9c14ce539ed`.

## R3-1 canary input

Input:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_canary`

The canary review is sealed and reports
`PASS_ENGINEERING_CONSUMABLE_INPUT_GATE`, 8 episodes and 3009 steps.  The
review cryptographically binds
`fit670_transition_v2_canary/TRANSITION_MANIFEST.json`, whose
`SHA256SUMS` digest is
`f29c1fdff29e3d57ce6b631856ea2126edc8e97ed075c8dcd75b7a45982f663e`.

The separately supplied `fit670_transition_v2_formal` root has a different
`SHA256SUMS` digest (`591601ca0fd69ac075f2a4a5ce02fbc35aa537f742b6f5e44c083f93df04177e`),
so it was rejected for this canary rather than silently substituted.  The
bound canary transition was used only because the canary's own sealed
`bindings.transition_receipt_sha256` identifies it.

Canary audit output:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_1_canary_input_audit_8fe7f507_bound_transition`

Reported input seal digest:
`43cfee475f3f8b4bb25312d50df5f1a929cbe9fff4f24e3267f6dea94987b69f`.

Audit root seal: `SHA256SUMS=170cc1354886d2bd3e7edf924baef17698510f0e28ada4778b5f9aa796b97e0f`; sidecar `15d59aa565e1b30a8ec2c0cf7e87e41e2cdc4e4fe2c68d242f1774270d7ec442`.

Checks passed: 8/8 identity closure, 3009/3009 step closure, source lineage,
contact schema, object/gripper binding, finite telemetry, attack disabled,
and protected reads 0.

## R3-2 Teacher canary

Teacher output:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_2_teacher_canary_8fe7f507`

The output is explicitly `DEVELOPMENT_NONCONSUMABLE`, contains all five heads,
uses the amended protocol SHA
`3f8bfa7de9f16773f78b92e0acb820c0227101a7657caabcf0a5f3300d24e9d8`,
and records `protected_reads=0`, `future_fields_used=false`,
`outcome_fields_used=false`, and `unknown_to_negative=false`.

Teacher output `SHA256SUMS` digest:
`bcacf18a80ac6cf2c41ee882cae7ec3191ed1a4f67325ef02fd5aaf089c88dca`.
Sidecar: `b951cd392b18b0efd0e7dcccdbdf5247a9c543253cc9b5aa29e460576f9978a9`.

## R3-3 coverage

Coverage output:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r3_3_teacher_coverage_8fe7f507`

Status: `HOLD_COVERAGE`.

Coverage root seal: `SHA256SUMS=56f3a551f908dad72bf350e1d348aea9a36c22f2fa5e7f248fb054f28fbbb3a5`; sidecar `95c9305760b5792a4369ec6e5d562aaa8fb699e46ad27231c93ed6e956e1804e`.

| Head | known steps | TRUE steps | FALSE steps | UNKNOWN steps | positive events | negative events | positive episodes | tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| physical_criticality | 0 | 0 | 0 | 3009 | 0 | 0 | 0 | 1 |
| k10_feasibility | 1339 | 1319 | 20 | 1670 | 82 | 5 | 8 | 1 |
| safe_release | 1339 | 0 | 1339 | 1670 | 0 | 74 | 0 | 1 |
| instability | 3009 | 26 | 2983 | 0 | 7 | 93 | 4 | 1 |
| gripper_closing_state | 3009 | 148 | 2861 | 0 | 16 | 84 | 2 | 1 |

No head reaches the frozen minimum of 20 known positive events, 20 known
negative events, 5 positive episodes, and 2 tasks.  Physical criticality is
entirely UNKNOWN in this tranche; safe release has no positive event.  These
are coverage limitations, not evidence that Student learning is impossible.

## CPU verification

At the final execution source commit, the official environment produced:

`37 passed in 9.04s`

and the targeted Python compilation passed.  The independent subagent review
returned `COMMIT_SAFE` before the final code commit.  The final code changes
were limited to the R3 teacher contact binding and its regression tests.

## Authorization and next step

| Activity | Status |
|---|---|
| formal-root metadata reconciliation | HOLD: live publication incomplete |
| sealed 8-episode canary input | PASS engineering consumable |
| five-head Teacher canary | PASS contract, nonconsumable |
| coverage gate | HOLD_COVERAGE |
| Student training | NOT STARTED |
| model inference | NOT STARTED |
| held-out development | NOT AUTHORIZED |
| shadow execution | NOT AUTHORIZED |
| OpenVLA rollout | NOT STARTED |
| attack | NOT STARTED |
| protected/CAL/CHECK/G10/T2R-D reads | 0 |

Wait for a separately sealed tranche with sufficient per-head event/task
coverage (next frozen tranche, not a live formal-root snapshot).  Do not alter
Teacher semantics or use UNKNOWN as negative to make the canary trainable.
