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

This H2 pass replaces the earlier scaffold output. The resolver now requires
physical sim-state evidence for accepted Teacher events; close-onset-only
windows are not accepted Teacher labels.

## Provenance

```text
branch = feature/sc5-cross-suite-layer1-resolver-20260619
tooling_commit = 592adc0426f4f46aed574d1b6c98f437f17d445f
server_worktree = /data/liuyu/repos/layer1_h2_592adc0
server_output_root = /data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620
clean300_deep_ledger = /data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447/tables/cross_suite_clean_300_master_ledger.csv
python_env = /data/aviary/envs/openvla_official_libero_20260525
```

Server-side `py_compile` passed. Server-side pytest was not run because the
approved OpenVLA environment is missing `pygments` for pytest import. The
targeted local CPU suite passed:

```text
33 passed
```

## Outputs

```text
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/manifests/layer1_dev_canary_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/manifests/layer1_blind_review_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/dev_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/dev_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/blind_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/blind_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/blind_review_package/blind_review_queue.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/blind_review_package/blind_review_hidden_audit_manifest.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_592adc0_20260620/blind_review_package/blind_review_instructions.json
```

Only small CSV/JSON evidence is committed under `tables/layer1_h2_20260620/`
and `reports/layer1_h2_20260620/`. Raw videos remain on the server as symlinks
in the blind review package.

## Development Canary Manifest

```text
selected = 12
suite_counts = {'libero_spatial': 4, 'libero_goal': 4, 'libero_10': 4}
```

Selection is deterministic by SHA over `(suite, task_idx, state_id, eval_seed,
source_episode_sha)` within preregistered buckets.

## Resolver-v1 Dev Canary Validation

```text
episode_rows = 12
event_rows = 0
failure_count = 0
validation_error_count = 0
teacher_status_counts = {
  'OBJECT_BINDING_AMBIGUOUS': 4,
  'NO_RELEVANT_GRASP_EVENT': 2,
  'CORRECT_SEMANTIC_ABSTAIN': 2,
  'TARGET_BINDING_AMBIGUOUS': 2,
  'RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM': 2
}
```

The zero accepted event count is intentional fail-closed behavior under the
repaired resolver. It means this canary did not produce frozen Teacher labels.
It does not license Layer 2 training or evaluation.

## Blind Human-Review Package

```text
episode_rows = 24
event_rows = 0
failure_count = 0
validation_error_count = 0
review_queue_rows = 24
teacher_status_counts = {
  'TARGET_BINDING_AMBIGUOUS': 4,
  'RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM': 8,
  'NO_RELEVANT_GRASP_EVENT': 4,
  'CORRECT_SEMANTIC_ABSTAIN': 4,
  'OBJECT_BINDING_AMBIGUOUS': 4
}
```

The blind package is event-schema-compatible but currently contains abstention
or unresolved proposals rather than positive Teacher event rows. It excludes
task-success fields from the reviewer queue; those are kept only in
`blind_review_hidden_audit_manifest.csv`.

## Forbidden Field Audit

```text
blind_review_queue.csv:
  task_success field present = false
  detector fields present = false

Resolver inputs:
  detector telemetry = stripped from step rows
  VIS/RAND/shuffled/attack outputs = not read
```

## Allowed Claims

- Layer 1 resolver-v1 now uses structured object/target binding plus sim-state
  physical event evidence.
- A deterministic development-canary manifest and disjoint blind-review manifest
  were generated from the frozen CLEAN300 deep-integrity ledger.
- Resolver-v1 produced fail-closed dev/blind proposal packages with zero
  internal validation errors.
- A blind human-review package was generated without detector fields, attack
  outputs, or task-success leakage in the reviewer queue.

## Forbidden Claims

- Do not claim full CLEAN300 Teacher labels exist.
- Do not claim this H2 sample contains accepted positive Teacher anchors.
- Do not claim human review is complete.
- Do not claim Layer 2 timing transfer, detector localization, VIS/RAND, or
  attack effectiveness.
- Do not use these proposal labels as frozen Teacher labels before H2 approval
  and human review.

## Stop Point

```text
HUMAN GATE H2_LAYER1_FREEZE
```
