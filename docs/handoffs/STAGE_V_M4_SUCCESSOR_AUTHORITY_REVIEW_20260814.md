# Stage V M4 successor authority review — 2026-08-14

## Decision

Successor authority validation and snapshot rebinding pass, but formal M4 remains **HOLD**.
The independent queue-contract review is `HOLD_QUEUE_CONTRACT_REPAIR_REQUIRED`; no formal
runtime authorization was issued and no formal parent was launched.

## Immutable bindings

- Successor source: commit `af51f3649d7bed218903fd5a0acdc6a97435badc`, tree `f38ee96997d5b4e3d167ed6441fbf8e7d2eaff8e`.
- Successor protocol: `configs/STAGE_V_M4_MATCHED_ACTION_PROTOCOL_SUCCESSOR_20260814.json`, SHA256 `da91e0ba8e6906f0067b402a0d89c33e1e31ee6111cdd8f1173a8a793f058ef1`.
- Snapshot rebind receipt: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_SNAPSHOT_REBIND_SUCCESSOR_C61_TO_AF51F364_20260814T131000Z/SNAPSHOT_REBIND_RECEIPT.json`, SHA256 `fde7d15b4fd8d01c4c5af3121a91e7d7028fa59a1f3ac254811833652f452fdc`.
- Snapshot inventory SHA256: `5dad4265f1c65fe22f8534f6dbc8dbb688f7129965080803918cb40f3292b1eb`.
- Compatibility runtime: commit `c61b53d42124ef093fe8946be8c87e68ad55845c`, tree `f2f9a226e39058d480778727df2dc960aa768e25`.
- Static audit: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_SUCCESSOR_AUTHORITY_AF51F364_20260814T131000Z/STAGE_V_M4_STATIC_AUDIT.json`, SHA256 `dd879f259a19c160d7f8d7680aa78ae3ff55ddfe62c518aad928372290e26103`, status `PASS_STATIC_DESIGN_ONLY`.
- Governance validator: `PASS_SUCCESSOR_AUTHORITY_VALIDATION` under `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`.

## Compatibility boundary

- Q00 canary: exactly one zero-treatment canary, `PASS_ZERO_TREATMENT_COMPATIBILITY`.
- 960 compatibility: exactly 960 immutable tasks, `PASS_960_ZERO_TREATMENT_COMPATIBILITY`.
- Treatment steps, labels, outcomes, and `V_phys`: zero/false.
- Protected counters: all zero, including `eval160_reads=0`.
- These artifacts are compatibility evidence only and do not authorize formal M4.

## Resource reservation

Direct SSH alias `dty` was used; no jump host. The official environment is
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` (Python 3.10.16).
The post-960 keeper PID `3579794` remains active with reservation leases on GPUs
2, 3, 5, and 7; `foreign_processes_modified=false`. The strict contract remains
`free_memory_mib > 20480`, one project worker per GPU, and foreign workloads are
not modified. Current reservation telemetry therefore leaves GPUs 0, 1, 4, and 6
as the unreserved candidates; this is a live snapshot, not a launch authorization.

## Queue-contract finding

The current formal runner has per-parent `O_EXCL` claims and parent-level structural
failure sealing, but it is a one-parent CLI requiring externally supplied
`--parent-index` and `--gpu`. It does not yet provide the required single global
worker dispatcher with rolling replenishment and dynamic admission. Its current
`CLAIM.json` also lacks the required atomic binding of worker ID, GPU, PID, source
commit/tree, authority SHA, and attempt ordinal.

Review receipt:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_SUCCESSOR_AUTHORITY_REVIEW_AF51F364_20260814T085419819167Z/SUCCESSOR_AUTHORITY_REVIEW.json`

Review root seal SHA256: `eab79d08c980a9007701c3b3d331ec8017cc87f22d6e145721126fdd452bb063`.

## Next legal gate

Repair the single global formal-parent scheduler and claim identity binding; verify
one-worker-per-GPU reservation integration, strict dynamic admission, max eight
in-flight parents, rolling replenishment, and pre-execution-only retry semantics.
Then rerun independent static/CPU/server audits and obtain a fresh explicit owner
formal authorization before issuing a launch gate. Do not read protected Eval160
or downstream formal outcomes during this hold.
