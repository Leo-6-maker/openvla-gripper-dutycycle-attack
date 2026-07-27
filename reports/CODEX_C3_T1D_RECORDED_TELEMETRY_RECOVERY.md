# C3-T1D recorded-telemetry geometry recovery

Date: 2026-07-27

## Decision

The old action-replay geometry is diagnostic only. This recovery uses the
sealed per-step `object_state` stream for object body origins and the frozen
LIBERO model for static site/body/geometry transforms. It does not call
`env.step()`, load OpenVLA, generate labels, train a Student, run rollout, or
run an attack.

The resulting root is a derived, non-consumable geometry snapshot. It is not
a V23 Teacher freeze and must be independently reviewed before any label
consumer uses it.

## Source provenance

The live branch was verified with:

```text
branch: codex/detector-completion-20260726
remote HEAD: 9f0edeb4a5ab65f89e36bec82d089e7222c1b519
local HEAD:  9f0edeb4a5ab65f89e36bec82d089e7222c1b519
worktree: clean
tree: 3d5219128e2c0fb7eca708e7599a185a83857bdd
```

Four collector variants cover the frozen 40-episode DEV pilot. Their
relevant telemetry-capture AST signature is identical:

```text
capture_ast_sha256: f38472217cc13d01bcc452a4746fa91f21ada1ae8260b6669f8264386ec2c8f0
```

| worker source SHA | episodes | steps | collector commit |
|---|---:|---:|---|
| `39cf1008474c0ae9b3abaef555b1b0018cb1b62af88180575c0dbdbeee12f628` | 7 | 1651 | `4d7a9daeb2bf9cf6b5d911f0ddcd93c254d0362b` |
| `6ec63c2e4ee488fc1c62c94eb37c049385a88480a6a898c8bfa4a901a51e91eb` | 3 | 532 | `e61cc30e55addbcbac6e60d8c9e295b751dd5487` |
| `96c618507b1bb39fb0e97d3e10c29c84e50c48b247ca66afbc32d9e9ef26963b` | 11 | 2916 | `9661ce80c04231e2e5ec64e19b364f6ea2d13dcb` |
| `a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311` | 19 | 4323 | `943b02749dce4414ec6791b15ceec87dbd3be1ba` |

The variants have the same relevant field capture (`robot0_eef_pos`,
`robot0_eef_quat`, `robot0_gripper_qpos`, `object_state`, and contact pairs),
but remain separate provenance variants; global source equivalence is not
claimed.

## Object-state index recovery

The actual official LIBERO object order was instantiated and compared with
the parsed BDDL order for all 40 tasks:

```text
map root: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/object_state_index_map_v1_8cda654_20260727_1115_gpu1_r2
map SHA256SUMS: de71b8e5e608ea1a5b62aaaf4113dc967c630f53038e9ae41f3ed01c90378155
map code snapshot: 8cda65437c63434382e31661d4eedd318b1aef4f
LIBERO source commit: 8f1084e3132a39270c3a13ebe37270a43ece2a01
task coverage: 40/40
order check: PASS
reset body-origin position max error: 0
reset body-origin rotation max error: 0
```

Each object record is 14 values: position, quaternion `xyzw`,
object-to-Eef position and object-to-Eef quaternion. The map is the binding
used by the materializer; BDDL order is not re-inferred at materialization
time.

## Geometry construction

For dynamic objects and dynamic site parents:

```text
recorded object_state body origin
  compose frozen body/site/geom local transform
  -> derived world geometry landmark
```

For static sites/bodies, the frozen model is used directly. Body-origin and
geometry-center are emitted as distinct fields. The materializer never uses
action-replay poses.

One registry inconsistency is explicit in the output. The
`libero_10/task_05` `black_book_1` row is recorded by the legacy registry as
`EXACT_GEOM id=5`, but that id is a same-named world init marker; the physical
object body is `black_book_1_main` and is bound by the sealed index map. The
217 affected rows are marked:

```text
registry_identity_status = INIT_GEOM_ALIAS_TO_INDEX_BODY
source = RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL_INIT_GEOM_ALIAS
```

This is an auditable derived alias, not a silent exact-geometry claim.

## Execution evidence

### Four-suite canary

```text
root: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/recorded_geometry_canary_v3_987680f_20260727_1155
SHA256SUMS: c55225b6e97b8efb831c7277723c0b9af6a695c708bb6b0bdbfb6329b02c0185
episodes: 4/4
relation rows: 1272/1272
known: 1272
unknown: 0
seal: PASS
```

### Full derived root

```text
root: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/recorded_geometry_full_v4_9f0edeb_20260727_1230
schema: V23_RECORDED_TELEMETRY_GEOMETRY_V1
status: DERIVED_RECORDED_GEOMETRY_NONCONSUMABLE
code snapshot: 9f0edeb4a5ab65f89e36bec82d089e7222c1b519
builder SHA256: 46e995cd68a912ceed45f8d67f8903982abf822a286893caabe7203d381fec20
SHA256SUMS: 82fcf73aee4f28bdc525cc04a14cc9271bbc43dadd3cde8e929d0457fed6a2cf
episodes: 40/40
source steps: 9422/9422
relation rows: 11880/11880
known rows: 11880
unknown rows: 0
seal: PASS
protected payload read: false
action replay: false
model inference: false
teacher labeling: false
consumer eligible: false
```

Relation totals are `In=5671` and `On=6209`. Target sources are:

```text
RECORDED_OBJECT_STATE_FROZEN_SITE_LOCAL: 4446
FROZEN_MODEL_SITE: 4105
RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL: 3329
```

Object sources are:

```text
RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL: 11663
RECORDED_OBJECT_STATE_FROZEN_BODY_LOCAL_INIT_GEOM_ALIAS: 217
```

The two registry tasks classified as `ARTICULATED_UNSUPPORTED` have zero
supported placement relations in the frozen registry; no articulated row was
converted to a negative label in this snapshot.

Earlier roots with all-UNKNOWN free-joint handling, incomplete exact-geom
handling, or an incorrect manually supplied commit string are retained as
failed/diagnostic evidence and were not overwritten.

## Boundary

```text
recorded telemetry geometry recovery = DERIVED SEALED SNAPSHOT
action replay geometry as truth       = REJECTED
V23 Teacher labels                    = NOT GENERATED
Student training/inference            = NOT STARTED
OpenVLA/model rollout                 = NOT STARTED
attack                                = NOT STARTED
```

Next required review is independent validation of the 40-task derived root,
especially the explicit `INIT_GEOM_ALIAS_TO_INDEX_BODY` exception and the
body-origin versus geometry-center landmark semantics. No protected or
locked split is opened by this handoff.
