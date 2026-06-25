# A800 Migration Receipt

Date: 2026-06-25

## Status

```text
A800_ENVIRONMENT_BOOTSTRAP: PASS
A800_EGL_SMOKE: PASS
A800_OBJECT_MODEL_LOAD: PASS
A800_GOAL_DEPENDENCY_TRANSFER: PASS
A800_C3_MIGRATION_PARITY: FAIL
TABLE1_ATTACK_EXECUTION: NOT_STARTED
```

## Frozen Runtime

| Field | Value |
|---|---|
| Host | `pm-364c0001` |
| Repository branch | `codex/a800-table1-v0-20260625` |
| PR #39 base head | `fad471e6b97db6f6b11aad30f71fd50055274d35` |
| Diagnostic code head | `8bbe1dea6c956b391a651f2e08939c0e3b0c1da0` |
| Python environment | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` |
| Python | `3.10.20` |
| PyTorch | `2.2.0+cu121` |
| Transformers | `4.40.1` |
| Object model | `/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object` |
| Goal model | `/mnt/sdc/dty_user/table1_dependencies/openvla-7b-finetuned-libero-goal` |
| Goal detector | `/mnt/sdc/dty_user/table1_dependencies/detector_goal/model.pt` |
| EGL smoke GPU | physical GPU 6 |
| C3 GPU pair | physical GPUs 5,6 |

The root filesystem was full during bootstrap. `TMPDIR`, `HOME`, caches, models,
and outputs were therefore pinned to `/mnt/sdc`. No files in the pre-existing
dirty checkout at `/mnt/sdc/dty_user/openvla_attack` were modified.

## Verification

- EGL produced a `256x256x3` `uint8` observation.
- The Object model loaded in `bfloat16` with action dimension 7.
- Goal model shards, metadata, and detector matched the source SHA-256 values.
- PR #39 Stage-B CPU suite passed: `271 passed`.
- After the diagnostic patch, the targeted exact-restore suite passed:
  `84 passed`.
- No C3 run produced an OOM, deterministic-operation exception, or new Xid.

## Boundary

This receipt establishes runtime and dependency availability only. It does not
establish exact replay parity, attack effectiveness, VIS superiority, or any
Table 1 result.
