# C3-T1D-R2B-R1 Head-wise Reference Recovery Handoff

## Scope

- Branch: `codex/detector-completion-20260726`
- Code snapshot: `0e97c986575b2e404e1b91f8250c2db2c784c100`
- Input: frozen non-protected DEV pilot, 40 identities / 9,422 steps
- Output root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c3_t1d_r2b_r1_reference_pi0_20260727_1140`
- Protected payload, T2R-D, 670 relabel, Student training, OpenVLA inference, rollout, and attack were not read or started.

This handoff records the first failed R2B gate. No new label run, A/B pilot, or quality report was started after the failure.

## R2B-R1A — recursive field and provenance audit

The audit read exactly the three bound files per identity: `episode_metadata.json`, `step_records.jsonl`, and `privileged_teacher_sidecar.jsonl`. It verified 40/40 identities, 9,422/9,422 step identities, recursive nested field paths, types, dimensions, finite/non-finite counts, and the sealed file closure. The recursive field inventory contains 6,326 field profiles.

Result: `HOLD_SCHEMA_UNBOUND`.

The metadata declares four collector source SHA variants:

```text
39cf1008474c0ae9b3abaef555b1b0018cb1b62af88180575c0dbdbeee12f628
6ec63c2e4ee488fc1c62c94eb37c049385a88480a6a898c8bfa4a901a51e91eb
96c618507b1bb39fb0e97d3e10c29c84e50c48b247ca66afbc32d9e9ef26963b
a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311
```

Only `a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311` was recoverable in the available official source files. The other three declared variants were not found in the searched official source/evidence/worktree files. This is an unresolved source-lineage block, not a parser or field-coverage pass.

The recovered source binding is:

- Official LIBERO source: `/mnt/sdc/dty_user/pi0_openpi/third_party/libero`
- LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Collector source: `/mnt/sdc/dty_user/openvla_attack_official_v3_legacy_20260716/scripts/official_clean_worker.py`
- Collector SHA: `a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311`
- Protocol: `/mnt/sdc/dty_user/openvla_attack_official_v3_20260716/configs/OFFICIAL_PROTOCOL_CONFIG_V1.json`
- Schema documentation: `/mnt/sdc/dty_user/openvla_attack_official_v3_20260716/docs/official_clean_detector_schema.md`

## R2B-R1B — object-state semantics and head coverage

The recovered collector/source contract is explicit:

- `object_state` is concatenated per object with width 14;
- component order is position (3), quaternion (4), object-to-EEF position (3), object-to-EEF quaternion (4);
- collector quaternion order is `xyzw`; MuJoCo geometry cases use `wxyz`, converted before geodesic comparison;
- object ordering is taken from the official parsed task object mapping, not guessed from task success or label output;
- qpos, EEF, object-state, and contact fields are available for all 44 supported relation rows.

R1B result: `R1B_PARTIAL_REFERENCE_HOLD`.

| Head | Available | Partial | Unknown | Decision |
|---|---:|---:|---:|---|
| `gripper_closing_state` | 44 | 0 | 0 | available |
| `physical_criticality` | 44 | 0 | 0 | available |
| `instability` | 44 | 0 | 0 | available |
| `placement` | 15 | 25 | 4 | partial |
| `safe_release` | 15 | 25 | 4 | partial |

The 25 partial placement/safe-release rows lack an independent static region/target pose in the three bound source files. Four articulated target rows remain unsupported and are explicitly `UNKNOWN`; they were not converted to FALSE.

## R2B-R1D — independent numerical fidelity

The comparison used two declared independent chains:

- source: recorded privileged `object_state`;
- replay: deterministic MuJoCo `geometry_cases` direct-simulation state;
- `same_action_replay_pose_chain = false`.

The frozen thresholds from `configs/C3_S3_NUMERICAL_THRESHOLDS_V1.json` are position p99 `<= 1e-4 m` and rotation p99 `<= 1e-3 rad`.

| Quantity | Count | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| object position (m) | 11,880 | 0.0200170 | 0.0554937 | 1.5565255 | 1.6495998 |
| object rotation (rad) | 11,880 | 0 | 0.0651953 | 0.7354752 | 1.9971308 |
| target position (m) | 3,329 | 0.0070165 | 0.0268644 | 0.0268644 | 0.0268644 |
| target rotation (rad) | 3,329 | 0 | 0.2112857 | 0.2112863 | 0.2289165 |

Result: `FAIL_NUMERICAL_FIDELITY`. The thresholds are exceeded by orders of magnitude. Geometry cases did not carry qpos or predicate streams, so qpos/comotion/predicate flip counts are `unavailable`, not zero. Near-boundary handling remains `UNKNOWN`.

## Sealed evidence

Root SHA256SUMS sidecar SHA:

```text
f426256955fefce552a705e9513584f98061e1d0b003a4380fe8aedae2ac2c9e
```

Payload SHA256 values:

```text
R1A_FIELD_AUDIT.json       eb768a804caee0859a8b446e8db977c9a2645379bf593b342dc8d56cce82e01c
R1B_REFERENCE_AUDIT.json   994b8a0927ca62d9867d826099d92e280ac5fe02d483930a9748100287abdffa
R1D_REFERENCE_DECISION.json e854db35f011b98d2946752813785b69115e4888533417115947e3081e72a96c
R1A_FIELD_PATHS.csv        3f06cfa1f60e24491fe40791d7846fd6c84f5897f4f17da077722a9b5d331a5f
HEAD_REFERENCE_COVERAGE.csv 2c21ff080f318da0535261a8f2e5b0af827bddcc7105dd8b1aeeb7358d65c853
R2B_R1_RECEIPT.json        3639aeb242352dc95c4ab12828e0527f4fc823fe0da1682452ce8d9e32375308
```

The root was independently rehashed after generation; the sidecar digest matched the recorded root digest. Prior failed/held roots remain immutable and were not overwritten.

## Gate state and stop decision

```text
C3_T1D_R2B_R1A = HOLD_SCHEMA_UNBOUND
C3_T1D_R2B_R1B = HOLD_REFERENCE_PARTIAL
C3_T1D_R2B_R1C = NOT_PASSED
C3_T1D_R2B_R1D = FAIL_NUMERICAL_FIDELITY
C3_T1D_R2C_DIAGNOSTIC = NOT_RUN
C3_T1D_R2C_FORMAL = NOT_RUN
T2R-D = NOT_READ
670 = NOT_READ
STUDENT = NOT_STARTED
OPENVLA = NOT_STARTED
ROLLOUT = NOT_STARTED
ATTACK = NOT_STARTED
```

R2B-R1 is not consumable. The next remediation must recover the three missing collector source variants and reconcile the deterministic replay initial-state/action timing and reference chain. It must rerun the audit from a clean detached code snapshot without weakening the frozen thresholds. No later gate is authorized from this evidence.
