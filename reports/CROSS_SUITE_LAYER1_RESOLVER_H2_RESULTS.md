# Cross-Suite Layer 1 Resolver H2 Results

## Decision Boundary

```text
H2_LAYER1_RESOLVER_IMPLEMENTATION = REPAIRED_FOR_REVIEW
FULL_CLEAN300_BATCH_LABELING = NOT_RUN
DETECTOR_TELEMETRY_READ = FORBIDDEN / NOT_USED_BY_RESOLVER
GPU_LIBERO_VIS_RAND_SHUFFLED_ATTACK = NOT_RUN
MANUAL_REVIEW = NOT_COMPLETE
NEXT_GATE = HUMAN GATE H2_LAYER1_FREEZE
```

This H2 pass repairs the previous fail-closed zero-event behavior. The resolver
now accepts physical event existence when grasp, lift, and stable carry are
present. Target proximity and placement completion are recorded as outcome
metadata and are not required for Teacher carry-window existence.

## Provenance

```text
branch = feature/sc5-cross-suite-layer1-resolver-20260619
tooling_commit = a161c1fedde47977e83c9bcb9d504caf3f75d4c6
server_worktree = /data/liuyu/repos/layer1_h2_a161c1f
server_output_root = /data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620
clean300_deep_ledger = /data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447/tables/cross_suite_clean_300_master_ledger.csv
python_env = /data/aviary/envs/openvla_official_libero_20260525
physics_config = configs/cross_suite_teacher_physics_v1.yaml
```

Server-side `py_compile` passed. Server-side pytest was not run because the
approved OpenVLA environment is missing `pygments` for pytest import. The
targeted local CPU suite passed:

```text
35 passed
```

## Outputs

```text
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/manifests/layer1_dev_canary_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/manifests/layer1_blind_review_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/dev_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/dev_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/blind_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/blind_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/blind_review_package/blind_review_queue.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/blind_review_package/blind_review_hidden_audit_manifest.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_a161c1f_20260620/blind_review_package/blind_review_instructions.json
```

Only small CSV/JSON evidence is committed under `tables/layer1_h2_20260620/`
and `reports/layer1_h2_20260620/`. Raw videos remain on the server as symlinks
in the blind review package.

## Development Canary Validation

```text
episode_rows = 12
event_rows = 6
failure_count = 0
validation_error_count = 0
teacher_status_counts = {
  'ELIGIBLE_EVENT': 6,
  'CORRECT_SEMANTIC_ABSTAIN': 2,
  'TARGET_BINDING_AMBIGUOUS': 2,
  'RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM': 2
}
```

The six accepted event rows are proposal labels only. They still require human
review before they can become frozen Teacher labels.

## Blind Human-Review Package

```text
episode_rows = 24
event_rows = 6
failure_count = 0
validation_error_count = 0
review_queue_rows = 24
review_queue_nonempty_event_rows = 6
teacher_status_counts = {
  'TARGET_BINDING_AMBIGUOUS': 4,
  'RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM': 8,
  'NO_RELEVANT_GRASP_EVENT': 2,
  'ELIGIBLE_EVENT': 6,
  'CORRECT_SEMANTIC_ABSTAIN': 4
}
```

The blind package now includes non-empty event proposals for review. It excludes
task-success fields from the reviewer queue; those are kept only in
`blind_review_hidden_audit_manifest.csv`.

## Key Contract Changes

- Event existence requires grasp, lift, and stable carry.
- Target proximity and placement completion are outcome fields, not existence
  gates.
- All close onsets are enumerated; the first close candidate with complete
  physical evidence is selected.
- Gripper site binding is exact and fail-closed; no `site_names[0]` fallback is
  allowed.
- Thresholds are versioned in `configs/cross_suite_teacher_physics_v1.yaml`.

## Forbidden Field Audit

```text
blind_review_queue.csv:
  task_success field present = false
  detector fields present = false

Resolver inputs:
  detector telemetry = stripped from step rows
  VIS/RAND/shuffled/attack outputs = not read
```

## Remaining Limitations

- Manual review is not complete.
- Supplementary multi/mixed mechanisms are not promoted to the primary
  denominator.
- Some primary episodes remain target-binding ambiguous.
- `sim_state_timing_convention` remains explicitly marked
  `unknown_pending_manual_alignment` in the physics config; final Teacher
  freeze still needs timing-alignment review.
- Teacher-only overlays are not yet populated; `teacher_only_overlay_path` is
  present but empty.

## Allowed Claims

- Layer 1 resolver-v1 now uses structured binding plus sim-state physical event
  evidence.
- A deterministic development-canary manifest and disjoint blind-review manifest
  were generated from the frozen CLEAN300 deep-integrity ledger.
- Resolver-v1 produced non-empty dev/blind physical-event proposals with zero
  internal validation errors.
- A blind human-review package was generated without detector fields, attack
  outputs, or task-success leakage in the reviewer queue.

## Forbidden Claims

- Do not claim full CLEAN300 Teacher labels exist.
- Do not claim human review is complete.
- Do not claim Layer 2 timing transfer, detector localization, VIS/RAND, or
  attack effectiveness.
- Do not use these proposal labels as frozen Teacher labels before H2 approval
  and human review.

## Stop Point

```text
HUMAN GATE H2_LAYER1_FREEZE
```
