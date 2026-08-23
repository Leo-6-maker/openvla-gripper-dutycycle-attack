# Active code surface V1

Status: `CODE_R2_ACTIVE_CODE_SURFACE_PASS`

Audit baseline:

- commit: `8ca76bbd8eed9841bf9264021d68a7ee19592779`
- tree: `83f9b6e15a3ae14219a9d2b1356eca789e1fc82c`
- scope: tracked repository state only

This is a navigation and lifecycle index, not scientific authority and not an
execution authorization. Scientific interpretation remains bound to root
seals, manifests, claim ledgers, and handoffs protected by
`IMMUTABLE_AUTHORITY_PATHS_V1.json`.

## Read first

1. `docs/repository/REPRODUCIBILITY_ENTRYPOINTS_V1.md` for safe commands.
2. `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json` for Paper V1 sources and
   claim boundaries.
3. `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json` for the terminal Stage X/F1
   synthesis boundary.
4. `reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json` for the current Stage Z HOLD.
5. `docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.md` before changing any
   historical path.

## Core library: `src/gripper_attack`

All 75 Python files are accounted for below. Static repository-wide reference
search found no unreferenced module safe to delete. A reference is not proof
that a historical execution path is currently authorized.

### Current execution primitives

These encode canonical action semantics, adapters, snapshots, or replay
contracts still consumed by current tests and frozen workflows:

`action_contract`, `attack_adapter`, `execution_target`, `failure_evidence`,
`openvla_libero_exec_spec`, `openvla_preprocess`,
`stage_v_canonical_execution_core`, `stage_v_causal_observation_snapshot`,
`stage_v_m3_5_phase_classifier`, `stage_v_m3_5_physical_taxonomy`, and
`stage_x_q3r3_branch_replay`.

They are implementation primitives only. Calling a runner that imports them
still requires separate phase-specific scientific and resource authorization.

### Current audit, route, and detector contracts

These remain active for validation, provenance reconstruction, sealed-result
interpretation, or compatibility tests:

- B3/official contracts: `b3_formal`, `b3_official_v3_s1`, `b3_retention`,
  `b3_teacher_training_adapter`, `b3_training_protocol`,
  `b3_v3_attack_protocol`, `b3_v3_dataset`, `b3_v3_runtime`,
  `b3_v3_viability_decision`, `official_v3_contract`, and
  `official_v3_sprint0`.
- Factorized routing: `factorized_calibration`, `factorized_runtime`,
  `factorized_scheduler`, `factorized_scheduler_adapter`,
  `factorized_scheduler_bridge`, `route_contract`, and `seal_utils`.
- Stage X contracts: `stage_x_t1_native_token_authority`,
  `stage_x_x1r_d1_clean_runtime_contract`,
  `stage_x_x1r_student_head_contract`, and
  `stage_x_x1r_v2_schedule_contract`.
- Detector lineage: `v5_dataset`, `v5_factorized_dataset`,
  `v5_factorized_loss`, `v5_factorized_loss_v2`, `v5_factorized_student`,
  `v5_factorized_student_v2`, `v5_factorized_student_v2_recommended`,
  `v5_factorized_teacher`, `v5_factorized_v2_splits`, `v5_physics`,
  `v5_protocol`, `v5_r3_features`, `v5_r3_student`, `v5_r3_teacher`,
  `v5_ranker`, `v5_scheduler`, `v5_teacher`, `v6_critical_dataset`,
  `v6_critical_loss`, `v6_critical_student`, and
  `v7_localization_student`.

The detector lineage contains historical training/runtime code as well as
current validation contracts. It is retained for reproducibility; this index
does not make training or attack execution a recommended entry point.

### Compatibility-only legacy modules

These preserve older imports, schemas, or public V4/SC5 behavior:

- V4 utility surface: `budget`, `directional`, `grasp`, `io`,
  `libero_v4_env_factory`, `logging_schema`, `metrics`, `triggers`, `types`,
  and `uncertainty`.
- Explicit compatibility wrapper: `gripper_semantics`. New code should use
  `openvla_libero_exec_spec`; `attack_adapter` still consumes the wrapper, so
  its path must remain available.
- Earlier controls and detector adapters: `m3_controls`, `sc5_dedup`,
  `sc5_detector_runtime`, `sc5_event_segmenter_v2`,
  `sc5_schema_adapter_v2`, `sc5_streaming_features_v2`,
  `d8_streaming_features_v3`, `v2_privileged_teacher`, and
  `v3_generation_parity`.
- `__init__.py` is the package marker and remains part of the import surface.

### Unused or uncertain

No module is classified as confirmed unused. Dynamic imports, historical
commands, and sealed source bindings make absence-of-use claims unsafe without
a narrower compatibility audit. No core module is moved or deleted in V1.

## Stage X

The family index is `scripts/stage_x/README.md`.

| Surface | Lifecycle | Scientific boundary |
| --- | --- | --- |
| Q3, Q3-AR, Q3R2, Q3R3 scripts/configs/tests | Historical engineering qualification and branch-replay infrastructure | HOLDs and infrastructure results are not negative attack evidence. Do not rerun-to-pass or relabel. |
| E3/E4 strict selective realizability | Sealed structural/model-side evidence | Parent-level descriptive evidence; candidate slots are non-iid diagnostics; no physical efficacy. |
| F1-A/B/C/C4 development and execution qualification | Closed development/canary lineage | F1T is terminal and sealed for PI; BRIDGE/F1-D was not opened. No tuning, top-up, recycle, or reopen. |
| `build_stage_x1r2_f1t_synthesis.py` and F1T outputs | Immutable historical paper-analysis producer/output | Read sealed outputs. Do not rerun or mutate the producer in this lane. Paper V2 gets a new export surface in CODE-R5. |

The current Stage X entry point is audit-only CPU testing plus immutable-path
verification. No `run_*`, `freeze_*`, `repair_*`, `seal_*`, or historical
`audit_*` producer is made canonical by this index.

## Stage Z

The family index is `scripts/stage_z/README.md`.

The controlling root is `reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json`, whose status
is `HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED`. It records no
scientific rollout, zero model/GPU/simulator/environment/protected counters,
and the next legal action as repair of the listed authority blocker with no
Z1.

`build_stage_z_runner_preparation.py` and
`configs/STAGE_Z_MULTI_MODEL_RUNNER_PREP_V1.json` are preparation surfaces,
not scientific evidence. Their execution-disable guards must remain intact.
The Z0/Z0R1/Z0R2 builders and audit producer write authority artifacts and are
historical producers, not safe read-only entry points. Download, upload, and
receipt helpers are operational tools requiring separate authorization.

The only canonical CODE-R2 Stage Z command is the synthetic CPU/static test in
`REPRODUCIBILITY_ENTRYPOINTS_V1.md`; it performs no model inference or
environment step.

## Paper-support tooling

The family index is `scripts/paper/README.md`.

- `scripts/paper/check_paper_v1_claims.py` is the canonical read-only semantic
  claim check.
- `scripts/repository/audit_immutable_authority_paths.py` is the canonical
  static byte/path authority check.
- `scripts/paper_v2/export_paper_v2_evidence.py` is the canonical deterministic
  Paper V2 CSV/JSON/TeX exporter/checker; `exports/paper_v2/` is its digest-bound
  output.
- Paper V1 builders/sealers are immutable historical producers and must not be
  rerun for cleanup.
- The Paper V2 surface does not mutate Paper V1 or treat the separate paper
  repository as scientific source.

## Change policy

This classification supports indexing, lifecycle labels, and compatibility-safe
deprecation. It does not justify moving or deleting sealed authority bytes,
historical runtime sources, or compatibility paths. Before any future cleanup,
run the immutable authority audit and trace imports, subprocess paths, docs,
configs, manifests, root seals, and Git history.
