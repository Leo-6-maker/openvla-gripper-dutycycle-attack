# M3 Arm-V5.1 Fresh Capture Restart Amendment - 2026-06-16

```text
STAGE: M3_ARM_V5_1R_CAPTURE_INFRA_FINAL_SEAL
AMENDMENT_STATUS: PREREG_RESTART_ONLY
GPU_EXECUTION: NOT_AUTHORIZED_BY_THIS_COMMIT
V5_2_ATTACK: NOT_AUTHORIZED_BY_THIS_COMMIT
```

## Reason

The previous V5.1 capture root is invalid as a denominator:

```text
old_root:
  /data/liuyu/outputs/m3_arm_v5_clean_capture_c2_1f2e84d_20260616_100039
status:
  V5_CAPTURE_PARTIAL_POST_ACTION_TERMINATION
  INVALID_FOR_V5_1_DENOMINATOR
```

It contains five completed clean-capture states and one post-action interrupted
state (`butter_s23`). Since `butter_s23` crossed `FIRST_ACTION_TAKEN` without
`CAPTURE_COMPLETED`, the old root cannot be repaired by resuming, filling, or
reusing a subset of successful states.

## Fresh Epoch Rule

Any future V5.1 run must be treated as a new capture epoch:

```text
new output root required
all 20 frozen task-state candidates rerun from scratch
old 5 CAPTURED states not reused
old butter_s23 not filled within the old root
state pool unchanged
event selection rule unchanged
post-generation or post-action failure remains terminal
only pre-generation infra failure may receive attempt_1
```

The purpose is to distinguish a valid newly authorized capture epoch from an
illegal retry of a partially executed old attempt.

## Environment Binding

The only authorized runtime environment for future runs is:

```text
/home/liuyu/.conda/envs/openvla_official_libero_20260525
```

Automatic fallback to the following similarly named environment is forbidden:

```text
/data/aviary/envs/openvla_official_libero_20260525
```

That path failed the V5 `use_fast=false` processor load due missing protobuf in
the observed server environment.

## Hardware Binding

Future V5.1 capture must pass ordered GPU binding:

```text
CUDA_VISIBLE_DEVICES physical index list
==
ordered expected GPU UUID list
```

Set membership is insufficient. A run that merely proves both UUIDs exist on
the machine, without proving their ordered binding to the requested physical
indices, is infra-invalid.

## Gate Before V5.2

V5.2 remains blocked until a fresh V5.1 root passes independent audit:

```text
20-state attempt ledger complete
infra invalid = 0
eligible clean CLOSE events >= 8
exact selected inputs = 8
all selected artifact bindings complete
independent external auditor PASS
```

No `seed428198`, `TRUE_PGD21`, `RAND21`, `SHUFFLED_GRAD21`, final eight-frame
execution, or LIBERO attack rollout is authorized by this amendment.
