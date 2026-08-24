# Official V3 Physics Task Decoder Audit

Date: 2026-07-19 CST
Code HEAD used on server: `3c53bcd4244de865191a76b000f20e945d6c0090`
Scope: FIT 800 only; decoder/source audit only; no Teacher materialization,
training, protected-split read, replay, or attack.

## Result

`PHYSICS_TASK_DECODER = PASS_TASK_CONDITIONAL_DECODER`

- FIT identities checked: `800`
- task count: `40`
- task pass: `40/40`
- decoder holds: `0`
- source mutation: `0`
- formal training authorized: `false`
- formal attack authorized: `false`

The sealed server audit root is:

`OFFICIAL_V3_PHYSICS_TASK_DECODER_V1_3c53bcd_20260719`

Its `SHA256SUMS` SHA is
`168afe0b1e68ed18d8b066e4a30e750e6495eb199f2ac51c9e308e9a5da86fd9`.
The sidecar SHA is
`31e190fc88094288107e8dce3f5de4571a77fcb867c9e08538a3baa8ecf31a01`.
Independent `sha256sum -c SHA256SUMS` and sidecar verification both passed.

## Decoder basis

The audit binds the task decoder to the actual Official V3 sources:

- collector script SHA:
  `39cf1008474c0ae9b3abaef555b1b0018cb1b62af88180575c0dbdbeee12f628`
- LIBERO domain source SHA:
  `4f4da47dd241ac6590d66c7559d76e44b68669924207f53143ca3a4962921f24`
- Robosuite base source SHA:
  `3d55a1d8b0350478b7e7e8d1c7a1250c8f173b69a67016ecb9e65c6965e741f6`

The collector explicitly reads `obs["object-state"]`. The audited LIBERO
domain source iterates the ordered `self.objects` collection and appends, per
object, four sensors in this order:

`pos`, `quat`, `to_eef_pos`, `to_eef_quat`

The audited Robosuite source concatenates observations in modality insertion
order. The decoder therefore assigns 14 contiguous values per BDDL object:

| component | slice within one object |
|---|---:|
| position | `[0:3]` |
| quaternion | `[3:7]` |
| relative EEF position | `[7:10]` |
| relative EEF quaternion | `[10:14]` |

The disabled `world_pose_in_gripper` observable is not included in the
object-state modality.

## Task-conditional dimensions

The decoder reads the official benchmark BDDL files and preserves declaration
order, including multiple object names on one BDDL line. Across the 40 FIT
tasks, object-state widths are task-conditional and equal `14 * object_count`:

| object-state width | object count | task count |
|---:|---:|---:|
| 28 | 2 | 5 |
| 56 | 4 | 10 |
| 70 | 5 | 13 |
| 98 | 7 | 10 |
| 112 | 8 | 2 |

Every FIT state `0..19` for every task was checked for:

- task name and language matching the official benchmark task;
- expected object-state width;
- constant width within the task;
- contiguous sidecar step indices;
- sidecar length matching episode metadata.

No object slice was inferred from model scores, Teacher labels, timestamps,
or attack evidence.

## Boundary

This closes only the physical object-state layout decoder. It does not yet
freeze the Physics Teacher's continuous scoring formulas or tier thresholds.
Those rules must be written and sealed before a Physics Teacher V2 root is
generated. The current result therefore does not authorize GPU training.

