# Stage V post-HOLD composite handoff

## Decision boundary

The CPU/read-only composite gate passed and is sealed. This handoff does not authorize the exact 40×24 plan gate, Teacher, Student, formal M4, protected branches, visual evaluation, or Eval160.

Current terminal state:

```text
V2 terminal status       = HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT
post-HOLD composite      = PASS_POST_HOLD_COMPOSITE_CORRIDOR_40_40
exact55 firewall         = FROZEN
FINAL40                  = FROZEN
FINAL_SPLIT              = FROZEN
exact 40x24 plan gate    = NOT_STARTED
formal Teacher           = NOT_STARTED
formal Student           = NOT_STARTED
formal M4                = NOT_AUTHORIZED
V_phys                   = NOT_GENERATED
protected counters       = 0
outcomes_read            = false
intervention_executed    = false
```

The V2 HOLD remains immutable and is not upgraded or repaired. The 32 current-source PASS/PASS pairs are carried as predecessor eligibility evidence; the eight new PASS/PASS rows are inputs to this independent composite only.

## Immutable source binding

```text
science commit = 3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2
science tree   = 2492a075e782a112d1e857248956b2647e751039
corridor runner sha256 = 26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279
official Python = /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
```

The corridor science runner was not modified. The new CPU-only producer and auditor are governance tools in PR #112, head `771a2c02f290c921bc5d2312c08480d5b33569ba`:

```text
scripts/detector_v5/reconcile_stage_v_m4_post_hold_composite.py = 94dc775ab242aeed8e094e124e7bc005ac151e8cc4b3b7dee9ebfdb0aec04d9a
scripts/detector_v5/audit_stage_v_m4_post_hold_composite.py      = 1dfecdcf19aaadb00bce39e323261651e878b0531615ded9244d0d3ed692bcda
```

## Reconciled population

```text
predecessor PASS/PASS = 32
new V1.1 PASS/PASS    = 8
final population       = 40
attempted identities   = 55 unique
```

Attempt firewall origin counts:

```text
V2_CURRENT_SOURCE_FINAL40       = 40
POST_HOLD_V1_ENGINEERING_INVALID = 3
POST_HOLD_V1_1                   = 12
duplicate_count                  = 0
```

Final split:

```text
TRAIN = 24
VAL   = 8
TEST  = 8
each suite = TRAIN 6 / VAL 2 / TEST 2
```

The eight new assignments are mechanically fixed:

| suite | parent | slot |
|---|---|---|
| libero_10 | `libero_10/task_00/state_27` | VAL |
| libero_goal | `libero_goal/task_03/state_36` | TRAIN |
| libero_goal | `libero_goal/task_03/state_41` | TRAIN |
| libero_goal | `libero_goal/task_02/state_40` | TEST |
| libero_spatial | `libero_spatial/task_09/state_34` | TRAIN |
| libero_spatial | `libero_spatial/task_06/state_24` | TRAIN |
| libero_spatial | `libero_spatial/task_06/state_34` | TRAIN |
| libero_spatial | `libero_spatial/task_04/state_44` | VAL |

The ten remaining reserve identities remain unattempted and are not used as fallback candidates.

## Sealed server root

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1_20260813T060000Z
SHA256SUMS sha256   = 52002c0d0f171389e893b91cb16280b727c7909d0a2d2112d0470ba00fb5a9e7
ROOT_SEAL file sha256 = 36a4f1e09646e7965608a7264350f25a7577396109b47c8cf0a0e0d2af554503
ROOT_SEAL content     = 52002c0d0f171389e893b91cb16280b727c7909d0a2d2112d0470ba00fb5a9e7  SHA256SUMS
```

The server verified every entry in `SHA256SUMS` as `OK`, then verified `ROOT_SEAL.sha256` against `SHA256SUMS`.

| artifact | sha256 |
|---|---|
| `STAGE_V_M4_POST_HOLD_COMPOSITE_RECONCILIATION_V1.json` | `1091d4911fa7f8c5d270dc4f76c5e54fe21918cffae984a1e8e2fbb8da809b21` |
| `STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_EXACT55_V1.json` | `d353016e093107158883e8922a2d3c98edfc0cca83c656795c5d7e1cb34f86a2` |
| `STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2.json` | `23e4f7b5a823abf84e9694605443a6e47449efacdad27b672a599309d0dd492c` |
| `STAGE_V_M4_FINAL_PARENT_SPLIT_V2.json` | `bc6ca7a5c85fb432f8e3f6ad94d4fef2b8c6db4884df5dd0f520241597e4d216` |
| `STAGE_V_M4_POST_HOLD_COMPOSITE_INDEPENDENT_AUDIT_V1.json` | `b000ca675a558079e0411f720bc073b021897103ab3aebad6224330929e47b64` |

Independent audit result:

```text
status = PASS_INDEPENDENT_EXACT55_FINAL40_SPLIT
independent_of_producer = true
source_receipts_reverified = true
exact55_reverified = true
final40_reverified = true
final_split_reverified = true
queue_prefix_reverified = true
failure_count = 0
```

## Input anchors

```text
V2 terminal HOLD report       = 866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9
predecessor inventory         = 4020a3f45efefb704c7110fd20e243adf84594a2c0920c413ee44928284baf14
pre-HOLD exact43 registry     = 7d5cfd1b3396f6af4ecd6f3de9b9d6ef454bb927c14a6619a90f14b27a273968
invalid V1 HOLD               = 2e2863ff9c84c2b004df32d258dadc9dba320b2ff9a6ed42f0873d6bc36843e0
V1.1 runtime reconciliation   = 6fce4411d737f16be7e01b76475d714bbef28006c51e667dec93afca22191a6a
candidate manifest             = 8f13e8427d19118cf7988be57a46c19fd1226af1941e00156b626e42f1c88be3
compatibility audit            = 8edf5e048b6dfb448fe0b5381e4cea221bc2a438f322a303ad6310fcc14a94e2
historical V2 split (mapping only) = f76ababf0750a78ee7adb3c81e0d15b945275d7210e2d0e41de1a623c6549cc4
historical V2 manifest (not reused) = f2c142b7c140b8412113e14a52d243eb8d9dafa9f9c4970456a91f2e41a479fa
```

## Next legal action

Stop here. The next separately reviewed gate is exact `40 × 24 = 960` plan-and-snapshot-only authority. It must use the frozen final40/split and must not recompute probe selection or read any intervention branch. Teacher/Student formal work remains behind the architecture freeze and primary-data firewall; the existing 670-identity engineering outputs remain development-only and nonconsumable.
