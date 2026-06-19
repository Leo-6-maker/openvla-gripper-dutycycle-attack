# Cross-Suite Layer 1 Resolver H2 Results

## Decision Boundary

```text
H2_LAYER1_RESOLVER_IMPLEMENTATION = COMPLETE_FOR_REVIEW
FULL_CLEAN300_BATCH_LABELING = NOT_RUN
DETECTOR_TELEMETRY_READ = FORBIDDEN / NOT_USED_BY_RESOLVER
GPU_LIBERO_VIS_RAND_SHUFFLED_ATTACK = NOT_RUN
MANUAL_REVIEW = NOT_COMPLETE
NEXT_GATE = HUMAN GATE H2_LAYER1_FREEZE
```

## Provenance

```text
branch = feature/sc5-cross-suite-layer1-resolver-20260619
tooling_commit = 68f02ddbc088c9c3e885266a89e734052f57d8d7
server_worktree = /data/liuyu/repos/layer1_h2_68f02dd
server_output_root = /data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619
clean300_deep_ledger = /data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447/tables/cross_suite_clean_300_master_ledger.csv
python_env = /data/aviary/envs/openvla_official_libero_20260525
```

Server-side pytest was not run because the approved environment is missing `pygments` for pytest import; the script compiled there and the targeted test suite passed locally before generation.

## Outputs

```text
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/manifests/layer1_dev_canary_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/manifests/layer1_blind_review_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/resolver_dev_canary/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/resolver_dev_canary/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/resolver_blind_review/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/resolver_blind_review/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/blind_review_package/blind_review_queue.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_68f02dd_20260619/blind_review_package/blind_review_instructions.json
```

Only small CSV/JSON evidence is committed under `tables/layer1_h2_20260619/` and `reports/layer1_h2_20260619/`. Raw videos remain on the server as symlinks in the blind review package.

## Development Canary Manifest

```text
selected = 12
suite_counts = {'libero_spatial': 4, 'libero_goal': 4, 'libero_10': 4}
mechanism_counts = {'single_object_pick_place': 8, 'articulated_only': 2, 'mixed_articulated_pick_place': 1, 'multi_object_transfer': 1}
bucket_counts = {'spatial_primary': 4, 'goal_primary_success': 2, 'goal_negative_or_abstain': 2, 'libero10_single_event': 2, 'libero10_multi_or_mixed': 2}
```

Selection is deterministic by SHA over `(suite, task_idx, state_id, eval_seed, source_episode_sha)` within preregistered buckets.

## Resolver-v1 Dev Canary Validation

```text
episode_rows = 12
event_rows = 8
teacher_status_counts = {'ELIGIBLE_EVENT': 8, 'CORRECT_SEMANTIC_ABSTAIN': 2, 'MULTI_EVENT_AUDIT_ONLY': 2}
failure_count = 0
validation_error_count = 0
```

Dev canary resolver labels remain proposals only; all positive event rows still require human review before use as frozen Teacher labels.

## Blind Human-Review Package

```text
review_rows = 24
suite_counts = {'libero_10': 8, 'libero_goal': 12, 'libero_spatial': 4}
mechanism_counts = {'single_object_pick_place': 12, 'multi_object_transfer': 4, 'push_or_planar': 1, 'articulated_only': 3, 'mixed_articulated_pick_place': 4}
teacher_status_counts = {'ELIGIBLE_EVENT': 12, 'MULTI_EVENT_AUDIT_ONLY': 8, 'CORRECT_SEMANTIC_ABSTAIN': 4}
video_status_counts = {'symlink': 24}
manual_review_complete = false
```

The blind package includes raw-video symlinks and proposed resolver fields, plus empty human-review fields. It does not include detector overlays, detector telemetry, VIS/RAND/shuffled outputs, or attack outputs.

## Forbidden Field Audit

```text
layer1_dev_canary_manifest_v1.csv: forbidden_fields_present = []
layer1_blind_review_manifest_v1.csv: forbidden_fields_present = []
resolver_dev_canary_teacher_episode_labels_v1.csv: forbidden_fields_present = []
resolver_blind_review_teacher_episode_labels_v1.csv: forbidden_fields_present = []
resolver_dev_canary_teacher_event_labels_v1.csv: forbidden_fields_present = []
resolver_blind_review_teacher_event_labels_v1.csv: forbidden_fields_present = []
blind_review_queue.csv: forbidden_fields_present = []
```

## Allowed Claims

- Layer 1 resolver-v1 tooling exists and passes CPU mock/schema tests.
- A deterministic development-canary manifest and disjoint blind-review manifest were generated from the frozen CLEAN300 deep-integrity ledger.
- Resolver-v1 produced proposal labels for the dev canary and blind-review set with zero internal validation errors.
- A blind human-review package was generated with raw-video links and empty human-review fields.

## Forbidden Claims

- Do not claim full CLEAN300 Teacher labels exist.
- Do not claim human review is complete.
- Do not claim Layer 2 timing transfer, detector localization, VIS/RAND, or attack effectiveness.
- Do not use these proposal labels as frozen Teacher labels before H2 approval and human review.

## Stop Point

```text
HUMAN GATE H2_LAYER1_FREEZE
```
