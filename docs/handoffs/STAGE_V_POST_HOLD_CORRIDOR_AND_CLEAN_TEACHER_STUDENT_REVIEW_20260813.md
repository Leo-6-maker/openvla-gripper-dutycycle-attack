# Stage V post-HOLD corridor and clean Teacher–Student review handoff — 2026-08-13

## Reviewer request

Please perform a read-only scientific and provenance review of this handoff and its linked repository artifacts. Return P0/P1/P2 findings with exact file-and-field evidence, then give separate `PASS`, `HOLD`, or `FAIL` decisions for:

1. carrying forward the 32 immutable predecessor `PASS/PASS` pairs;
2. accepting the new 8 `PASS/PASS` pairs for independent composite reconciliation;
3. constructing and freezing the exact final40 and split;
4. proceeding later to the exact 40×24 plan-and-snapshot-only gate;
5. treating all existing Teacher/Student outputs as development-only and non-consumable;
6. the proposed clean Teacher → causal Student → held-out M4 ordering.

No runtime action, branch merge, protected-data read, or formal M4 authorization is requested from the reviewer. Do not recommend rerunning a completed identity to improve its status.

## Executive state

The post-HOLD clean corridor replenishment reached its preregistered targets and terminated normally:

- predecessor evidence: 32 immutable current-source `PASS/PASS` pairs;
- new stable evidence: 8 `PASS/PASS` pairs;
- prospective suite targets reached: `libero_10=1`, `libero_goal=3`, `libero_spatial=4`;
- new identities attempted: 12 of the frozen 22-candidate queue;
- new clean replicates completed: 24 of 24 attempted;
- structural runtime failures: 0;
- output root hash and read-only seal checks: PASS;
- formal M4: not authorized, not executed, outcomes unread;
- `V_phys`: not generated;
- primary Teacher/Student: no formally consumable checkpoint, manifest, feature selection, or threshold;
- protected counters remain zero.

The immediate legal next action is an independent composite reconciliation. The current run does **not** itself freeze final40 or a split.

## Non-negotiable scientific order

The append-only architecture remains:

```text
CLEAN_ROLLOUT
  → PRIVILEGED_CLEAN_TEACHER_C_t
  → CLEAN_TEACHER_SUPERVISED_CAUSAL_STUDENT_C_HAT_t
  → HELD_OUT_MATCHED_COUNTERFACTUAL_VALIDATION_V_t_d
  → TIMING_VIS_DEFENSE_LATER
```

The semantic separation remains:

```text
C_t != V_t(d) != E_t
```

The operational sequence currently required is:

```text
V2 terminal HOLD
  → post-HOLD corridor replenishment                         [complete]
  → independent composite corridor reconciliation           [next]
  → final40 freeze
  → split freeze
  → exact 40×24 plan-and-snapshot-only gate
  → independent exact-manifest audit
  → primary data firewall
  → clean privileged Teacher package and freeze
  → Teacher coverage/reliability audit
  → causal Student training on clean Teacher labels
  → Student checkpoint/feature/threshold freeze
  → independent Teacher/Student audit
  → formal M4 authorization
  → held-out CONTROL/T3/T5/T10 execution and outcome read
```

Formal M4 must not precede the Teacher/Student freeze. The exact plan-and-snapshot gate is clean-only and must not execute an intervention or expose an outcome.

Authoritative architecture references:

- [`configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json`](../../configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1.json)
- [`configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM.json`](../../configs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM.json)
- [`docs/handoffs/STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM_20260813.md`](STAGE_V_SCIENTIFIC_ARCHITECTURE_FREEZE_V1_1_STATUS_ADDENDUM_20260813.md)

## Immutable V2 predecessor state

V2 remains permanently sealed as:

```text
HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT
PASS/PASS                    32/40
PASS/CLEAN_FAILURE            3
CLEAN_FAILURE/CLEAN_FAILURE   4
INELIGIBLE/INELIGIBLE         1
```

It was not repaired, upgraded, reopened, or rerun. The authoritative terminal report is:

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/
STAGE_V_M4_ARCHITECTURE_FREEZE_REBUILD_V1_20260813T000000Z/
STAGE_V_M4_CORRIDOR_CURRENT_SOURCE_AB_RECONCILIATION_HOLD_V1.json
SHA-256: 866ce90f73cd542584c4db3fca4b590ebc014e7e7e9dbd2a91adcdee210c7fd9
```

The predecessor suite counts are:

| Suite | Stable predecessor pairs |
|---|---:|
| `libero_10` | 9 |
| `libero_goal` | 7 |
| `libero_object` | 10 |
| `libero_spatial` | 6 |

The carry-forward claim is narrowly limited to: the new protocol references 32 immutable current-source `PASS/PASS` receipt pairs as predecessor eligibility evidence.

## Two P0 corrections already enforced

### P0-1 — held-out ordering

The repository now explicitly forbids `M4 → Teacher/Student` as a primary path. Teacher semantics, Student inputs, checkpoint, feature schema, and thresholds must be frozen before any held-out M4 outcome is read.

### P0-2 — source homogeneity

The science runner was not modified. New receipts use the exact same science execution plane as the 32 predecessor pairs:

| Binding | Exact value |
|---|---|
| science worktree | `/mnt/sdc/dty_user/openvla_attack_worktrees/stage-v-m4-governed-20260812` |
| commit | `3b2ffdb5809710e3c7f2e0a529600c8d7a79b9b2` |
| tree | `2492a075e782a112d1e857248956b2647e751039` |
| corridor runner SHA-256 | `26ceed23646177ce675e32eba6617ade7b02804a3c372a756b1ebe098ef72279` |

All new dispatch, ranking, reconciliation, and diagnostics live outside that science core. The compatibility audit completed 47 checks with zero failures.

## Invalid first launch remains non-consumable

An earlier V1 launch was stopped before governance closure and permanently sealed as engineering history:

```text
status: HOLD_ENGINEERING_INVALID_PRELAUNCH_GOVERNANCE_INCOMPLETE
root: /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/
      STAGE_V_M4_POST_32_OF_40_HOLD_V1_20260813T000000Z
hold report SHA-256: 2e2863ff9c84c2b004df32d258dadc9dba320b2ff9a6ed42f0873d6bc36843e0
```

The following identities are permanently counted as corridor-attempted and are never reused or rerun:

- `libero_10/task_06/state_38`
- `libero_goal/task_04/state_39`
- `libero_goal/task_04/state_43`

No receipt from that root is included in the new 8 stable pairs.

## V1.1 frozen governance chain

| Artifact | SHA-256 |
|---|---|
| [`STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1_1.json`](../../configs/STAGE_V_M4_CORRIDOR_REPLENISHMENT_POST_32_OF_40_HOLD_V1_1.json) | `dac39bf54007b6ae99de74cf8d5c9f44eeb5175b3411a4bc1615015a36ac7abf` |
| [`STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_POST_HOLD_V1_1.json`](../../configs/STAGE_V_M4_CORRIDOR_QUALIFICATION_PROTOCOL_POST_HOLD_V1_1.json) | `b4a52ad87ce4fcd39a619ec8512cef377e51ad64296d0dd92153b40f833393c8` |
| [`STAGE_V_M4_CORRIDOR_PREDECESSOR_PASS_PASS_INVENTORY_V1.json`](../../configs/STAGE_V_M4_CORRIDOR_PREDECESSOR_PASS_PASS_INVENTORY_V1.json) | `4020a3f45efefb704c7110fd20e243adf84594a2c0920c413ee44928284baf14` |
| [`STAGE_V_M4_POST_HOLD_STATIC_TAXONOMY_AUDIT_V1.json`](../../configs/STAGE_V_M4_POST_HOLD_STATIC_TAXONOMY_AUDIT_V1.json) | `6e86158e9c830daa85311636138a01b6e0142bd6d7f989ac51ca7d3a0c900912` |
| [`STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1.json`](../../configs/STAGE_V_M4_CORRIDOR_ATTEMPT_REGISTRY_V1_1.json) | `7d5cfd1b3396f6af4ecd6f3de9b9d6ef454bb927c14a6619a90f14b27a273968` |
| [`STAGE_V_M4_POST_HOLD_V7_QUALIFIED_ROWS_V1_1.jsonl`](../../configs/STAGE_V_M4_POST_HOLD_V7_QUALIFIED_ROWS_V1_1.jsonl) | `809d8ae739c67e90e856677a2249127b383ccb885dcbab680ee5c4e9840d2c6c` |
| [`STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1_1.json`](../../configs/STAGE_V_M4_POST_HOLD_CANDIDATE_PARENT_MANIFEST_V1_1.json) | `8f13e8427d19118cf7988be57a46c19fd1226af1941e00156b626e42f1c88be3` |
| [`STAGE_V_M4_CORRIDOR_STATIC_AUDIT_POST_HOLD_V1_1.json`](../../configs/STAGE_V_M4_CORRIDOR_STATIC_AUDIT_POST_HOLD_V1_1.json) | `511b471286eee67a90f84edde9c3623e1e68aa5fda672fbe29566e0dd07d95f1` |
| [`STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION_POST_HOLD_V1_1.json`](../../configs/STAGE_V_M4_CORRIDOR_RUNTIME_AUTHORIZATION_POST_HOLD_V1_1.json) | `ffce2bac5e90aa33350f73a1143a1776f622eaf35ebb61a65994791c5a1a2203` |
| [`STAGE_V_M4_CORRIDOR_PREDECESSOR_COMPATIBILITY_AUDIT_V1.json`](../../configs/STAGE_V_M4_CORRIDOR_PREDECESSOR_COMPATIBILITY_AUDIT_V1.json) | `8edf5e048b6dfb448fe0b5381e4cea221bc2a438f322a303ad6310fcc14a94e2` |

Static taxonomy was applied uniformly to all pre-static candidates: 25 `SUPPORTED`, 1 `UNSUPPORTED_TAXONOMY`, 0 `ABSTAIN_STATIC`. No task-specific denylist exists.

## Runtime environment and resource contract

| Binding | Exact value |
|---|---|
| Python entrypoint | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python` |
| resolved Python | `/home/sz/miniconda3/envs/hallo/bin/python3.10` |
| executable SHA-256 | `65985bd811a03dc934e83a100bac8ff6e5ab6adc462ad6de7d3ed9ea4e11ca57` |
| version | Python 3.10.16 |
| official snapshot | `/mnt/sdc/dty_user/openvla_attack_official_v3_20260716` |
| snapshot commit/tree | `4d7a9daeb2bf9cf6b5d911f0ddcd93c254d0362b` / `cd5fe5263afd66d9131159d97bba1d1d660be5df` |
| upstream | `/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4` |
| upstream commit/tree | `c8f03f48af692657d3060c19588038c7220e9af9` / `c326be57bb61629a3efd2b968ef141fa576a623f` |

All GPU indices 0–7 were admitted only when free memory was strictly greater than 20,480 MiB at lease time. Foreign processes were observational and non-gating; none was modified. The protocol limited useful concurrency to six replicates because there were three suite workers with one A/B pair each. No extra identity was started merely to occupy the remaining devices.

## Terminal replenishment evidence

Server root:

```text
/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/
STAGE_V_M4_POST_32_OF_40_HOLD_V1_1_20260813T030000Z
```

Repository copy for review:

- [`reports/server_evidence/STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION.json`](../../reports/server_evidence/STAGE_V_M4_POST_HOLD_V1_1_RUNTIME_RECONCILIATION.json)
- SHA-256: `6fce4411d737f16be7e01b76475d714bbef28006c51e667dec93afca22191a6a`

Independent server verification after completion:

- terminal status: `PASS_POST_HOLD_CORRIDOR_TARGETS_REACHED`;
- `attempted_identity_count=12`;
- `pair_count=12`;
- science receipt count: 24;
- `consumable_for_composite_reconciliation=true`;
- `retry_forbidden=true`;
- `outcomes_read=false`;
- `intervention_executed=false`;
- all root entries non-writable;
- `sha256sum -c SHA256SUMS --quiet`: PASS;
- `sha256sum -c ROOT_SEAL.sha256 --quiet`: PASS;
- `SHA256SUMS` SHA-256: `117e5b5c9b026d85173440e39ea8c850d86375afb502a6fa2b3124a4c971c7f0`;
- `ROOT_SEAL.sha256` SHA-256: `b33474a6f65396c5dde59d7c9f9fe06c081735e78232f70f3a252234ca535811`.

### Exact pair ledger

| Suite | Rank | Identity | A | B | Stable |
|---|---:|---|---|---|---|
| `libero_10` | 1 | `task_09/state_42` | `CLEAN_FAILURE` | `CLEAN_FAILURE` | no |
| `libero_10` | 2 | `task_00/state_27` | `PASS` | `PASS` | yes |
| `libero_goal` | 1 | `task_03/state_36` | `PASS` | `PASS` | yes |
| `libero_goal` | 2 | `task_03/state_41` | `PASS` | `PASS` | yes |
| `libero_goal` | 3 | `task_02/state_45` | `CLEAN_FAILURE` | `PASS` | no |
| `libero_goal` | 4 | `task_02/state_40` | `PASS` | `PASS` | yes |
| `libero_spatial` | 1 | `task_08/state_44` | `PASS` | `INELIGIBLE` | no |
| `libero_spatial` | 2 | `task_09/state_34` | `PASS` | `PASS` | yes |
| `libero_spatial` | 3 | `task_06/state_24` | `PASS` | `PASS` | yes |
| `libero_spatial` | 4 | `task_06/state_34` | `PASS` | `PASS` | yes |
| `libero_spatial` | 5 | `task_00/state_45` | `INELIGIBLE` | `INELIGIBLE` | no |
| `libero_spatial` | 6 | `task_04/state_44` | `PASS` | `PASS` | yes |

Status-pair totals are 8 `PASS/PASS`, 1 `PASS/INELIGIBLE`, 1 `CLEAN_FAILURE/PASS`, 1 `CLEAN_FAILURE/CLEAN_FAILURE`, and 1 `INELIGIBLE/INELIGIBLE`. Ten frozen queue identities remain unattempted because every suite reached its preregistered target.

## Expected composite result, not yet frozen

If the independent reconciler validates all bindings, the suite population becomes exactly 10 per suite:

| Suite | Predecessor | New stable | Composite candidate |
|---|---:|---:|---:|
| `libero_10` | 9 | 1 | 10 |
| `libero_goal` | 7 | 3 | 10 |
| `libero_object` | 10 | 0 | 10 |
| `libero_spatial` | 6 | 4 | 10 |
| **Total** | **32** | **8** | **40** |

The frozen slot rule mechanically implies the following additions:

| Suite | Identity | Expected missing slot |
|---|---|---|
| `libero_10` | `task_00/state_27` | `VAL` |
| `libero_goal` | `task_03/state_36` | `TRAIN` |
| `libero_goal` | `task_03/state_41` | `TRAIN` |
| `libero_goal` | `task_02/state_40` | `TEST` |
| `libero_spatial` | `task_09/state_34` | `TRAIN` |
| `libero_spatial` | `task_06/state_24` | `TRAIN` |
| `libero_spatial` | `task_06/state_34` | `TRAIN` |
| `libero_spatial` | `task_04/state_44` | `VAL` |

These are expected outputs of the frozen rule, not yet authoritative manifests. The earlier V2 final40/split artifacts must remain historical and must not be edited or silently reused.

The primary Teacher/Student identity firewall must exclude the union of:

- 40 original V2 corridor-attempted identities;
- 3 invalid-V1 corridor-attempted identities;
- 12 V1.1 corridor-attempted identities.

Therefore the next firewall audit must account for exactly 55 unique attempted identities before any primary FIT, CAL, CHECK, threshold selection, or model selection is authorized.

## Teacher–Student completion audit

### Architecture: frozen

The intended structure is complete at the specification level:

- Teacher target: clean-derived privileged physical criticality `C_t`;
- Teacher inputs: clean state/telemetry, clean action and end-effector history, clean contact/phase evidence, and clean horizon/mask metadata;
- Student supervision: clean Teacher labels only;
- Student inputs: deployment-visible current and finite causal history only;
- prohibited Student inputs: future fields, privileged unavailable fields, M4 condition/outcomes, later evaluation outcomes, identity leakage, and post-treatment fields;
- model selection: clean-only locked splits;
- held-out M4 role: validation only, never training supervision.

### Existing engineering outputs: useful diagnostics, not formal evidence

The server contains a non-consumable engineering chain produced outside a complete final40/firewall-bound primary line. It is preserved but cannot be promoted automatically:

| Output | Key state |
|---|---|
| `STAGE_V_PRIMARY_FIT_TO_TEACHER_TRANSITION_20260813T190000Z/FIT_TO_TEACHER_TRANSITION.json` | `PASS_FIT_TO_TEACHER_AUTHORIZATION`; 670 identities; only Teacher label generation allowed; training and rollout permissions false; SHA `4cb435a83d7b3840332e8b3ae113e1096972eadbbdee868cbe0619a5ec430754` |
| `STAGE_V_PRIMARY_CLEAN_TEACHER_20260813T191000Z/teacher_manifest.json` | 670 identities, 196,483 steps, labels generated; no future or outcome fields; final status `DEVELOPMENT_NONCONSUMABLE`; formal training/inference false; SHA `27cec6d0fd6a3181843d2ceea23235b9b41970dd52811b56c3275905f4abff19` |
| `STAGE_V_PRIMARY_CLEAN_TEACHER_COVERAGE_20260813T193500Z/coverage_report.json` | `HOLD_COVERAGE`; four heads pass, `safe_release` held; formal training false; SHA `2820c22d405d51046cdde99bcd97a6010dd27f66a919a557e9de9111587b5afa` |
| `STAGE_V_PRIMARY_CLEAN_TS_TRANSITION_20260813T194500Z/TEACHER_STUDENT_TRANSITION.json` | `PASS_DEVELOPMENT_ELIGIBLE_HEADS`; full-five status `HOLD_COVERAGE`; formal training false; labels not copied; SHA `875a542f753d140f6325aed14d6270a7dcdf9863af8aa7e4ef762169ae6b2991` |
| `STAGE_V_PRIMARY_CLEAN_G0_20260813T195000Z/G0_LABEL_BASELINE_AUDIT.json` | baseline audit passes, but `consumable=false`; held-out evaluation false; formal training/inference false; SHA `b5611f9fead5e667374a131a8c93d13af3ad1a78dbcfbcbbcf8719737217b748` |

Coverage details:

| Head | Coverage | Positive events | Positive suites | Positive tasks |
|---|---|---:|---:|---:|
| `physical_criticality` | PASS | 673 | 3 | 15 |
| `k10_feasibility` | PASS | 2,894 | 4 | 37 |
| `instability` | PASS | 222 | 3 | 17 |
| `gripper_closing_state` | PASS | 1,129 | 4 | 36 |
| `safe_release` | HOLD | 408 | 1 | 7 |

The transition exposes four heads for development-only work and holds `safe_release`. There is no formally consumable Student dataset, trained Student checkpoint, CAL/CHECK result, selected checkpoint, frozen feature schema for the primary line, frozen threshold, or formal inference authorization.

Accordingly:

| Layer | Current scientific state |
|---|---|
| architecture semantics | `FROZEN` |
| engineering Teacher label path | demonstrated, development-only |
| formal primary Teacher | `NOT COMPLETE` |
| formal Teacher coverage/reliability gate | `HOLD` |
| Student development transition | partial, development-only |
| formal causal Student training | `NOT STARTED` |
| Student checkpoint/features/threshold freeze | `NOT STARTED` |
| held-out Teacher/Student localization evaluation | `NOT STARTED` |

The reviewer should not interpret directory names containing `PRIMARY` as scientific promotion. The controlling fields are the status, permission, consumability, and freeze bindings above.

## Protected boundary

The final replenishment report records:

```text
protected_reads = 0
eval160_reads = 0
attack_rollouts = 0
vis_pgd_attack_rollouts = 0
outcomes_read = false
intervention_executed = false
```

There is no formal M4 authorization and no `V_phys` map. Eval160 remains unread and requires separate owner authorization.

## GitHub state

- PR #111 remains open and draft at `fcaa59cacf1895cc9f1d372944366b7b2952911c`; it is not being enlarged with this work.
- Stacked PR: [#112](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/112)
- base: `codex/stage-v-resource-contract-20260810`
- branch: `codex/m4-corridor-replenishment-post-32-of-40-hold-20260813`
- evidence head before this handoff commit: `2cbe78674f8b34aae691b6264661eb469341fc63`
- evidence tree: `51b9d5823283a359141e8c45535bf1e8c9c19b6d`
- checks at that head: `cpu-stageb`, `cpu-detector-v5`, and `cpu-b3-official-v3` all successful; these are engineering checks only.

## Requested reviewer decisions

Please answer these questions explicitly:

1. Does the 47-check compatibility audit adequately support combining the immutable predecessor 32 with the new 8 while preserving the V2 terminal HOLD claim?
2. Does the 12-pair terminal report satisfy the frozen sequential stopping rule without outcome-driven candidate substitution?
3. Are the eight mechanical slot assignments above the only assignments allowed by the frozen rule?
4. Must the independent reconciler reject any duplicate, missing receipt, source mismatch, attempted-identity mismatch, counter drift, or nonzero intervention flag? Expected answer: yes, fail closed.
5. Is the next valid sequence exactly `composite reconciliation → final40/split → exact 40×24 plan/snapshot → firewall → formal clean Teacher → formal causal Student → formal M4`?
6. Do the historical Teacher/G0/transition artifacts remain development-only despite their successful engineering subchecks? Expected answer: yes.
7. What is the minimum additional evidence package required to turn the clean Teacher and causal Student into formally consumable frozen artifacts without reading M4 outcomes?
8. Are any claims broader than the prospectively defined clean-successful, taxonomy-supported, A/B corridor-stable critical-opportunity population unsupported? Expected answer: yes.

Until this review is resolved, do not authorize formal M4, do not read held-out outcomes, and do not promote the historical Teacher/Student outputs.
