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
tooling_commit = 6eb88630f20f99aac4d64356974b5a18345c3673
server_worktree = /data/liuyu/repos/layer1_h2_6eb8863_bundle
server_output_root = /data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620
clean300_deep_ledger = /data/liuyu/audit_outputs/cross_suite_clean_300_final_deep_integrity_20260619_202447/tables/cross_suite_clean_300_master_ledger.csv
python_env = /data/aviary/envs/openvla_official_libero_20260525
physics_config = configs/cross_suite_teacher_physics_v1.yaml
```

Server-side resolver and Teacher-only review package generation were rerun after
the region-specific target binding repair. The targeted local CPU suite passed:

```text
111 passed
```

## Outputs

```text
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/manifests/layer1_dev_canary_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/manifests/layer1_blind_review_manifest_v1.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/dev_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/dev_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_resolver/teacher_episode_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_resolver/teacher_event_labels_v1.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/blind_review_queue.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/blind_review_hidden_audit_manifest.csv
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/blind_review_instructions.json
/data/liuyu/layer1_outputs/h2_layer1_resolver_6eb8863_20260620/diagnostic_review_package/teacher_only_overlay_timeline_manifest.csv
```

Only small CSV/JSON evidence is committed under `tables/layer1_h2_20260620/`
and `reports/layer1_h2_20260620/`. Raw videos remain on the server as symlinks
in the diagnostic holdout review package.

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
target_binding_status_counts = {
  'BOUND_STRUCTURED_FALLBACK': 6,
  'FAILED': 2,
  'AMBIGUOUS': 2,
  'NOT_APPLICABLE': 2
}
```

The six accepted event rows are proposal labels only. They still require human
review before they can become frozen Teacher labels.

## Diagnostic Holdout Review Package

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
target_binding_status_counts = {
  'FAILED': 9,
  'AMBIGUOUS': 3,
  'BOUND_STRUCTURED_FALLBACK': 8,
  'NOT_APPLICABLE': 4
}
```

This 24-row package is now classified as `resolver_diagnostic_holdout_v1`
because it was observed during resolver development. It is useful for resolver
debugging and independent review of proposed event semantics, but it is not the
final unbiased validation set. A new disjoint final blind package must be
selected only after resolver code, thresholds, target-binding rules, and timing
offsets are frozen.

The diagnostic holdout package includes non-empty event proposals for review. It
excludes task-success fields from the reviewer queue; those are kept only in
`blind_review_hidden_audit_manifest.csv`.

Teacher-only review aids were generated for every diagnostic row:

```text
teacher_only_timeline_rows = 24 / 24
teacher_only_overlay_mp4 = 24 / 24
overlay_manifest_sha256 = 3f9d4b4dee34a8cdd459ce48e33ac9c670af61240499f660e65064d1f6dc2366
```

The overlays contain only resolver proposal markers. They do not include
detector telemetry, detector overlays, VIS/RAND/shuffled outputs, attacks, or
task-success fields.

## Key Contract Changes

- Event existence requires grasp, lift, and stable carry.
- Target proximity and placement completion are outcome fields, not existence
  gates.
- All close onsets are enumerated; the first close candidate with complete
  physical evidence is selected.
- Gripper site binding is exact and fail-closed; no `site_names[0]` fallback is
  allowed.
- Thresholds are versioned in `configs/cross_suite_teacher_physics_v1.yaml`.
- Timing alignment was checked on six development-canary event proposals:
  `step_telemetry.csv`, `frame_index.csv`, `sim_state_stream.npz`, and
  `rollout_raw.mp4` all have matching row/frame counts. The freeze-candidate
  contract is:
  `sim_state_timing_convention=one_sim_state_row_per_executed_action_step`,
  `sim_row_to_action_step_offset=0`, and `video_frame_to_step_offset=0`.
- Region-specific target aliases now fail closed when no matching site/body
  exists. Generic object default sites cannot satisfy targets such as
  `cabinet_top` or `back_compartment_of_caddy`.

## Region-Specific Binding Refresh

The current evidence tables were regenerated at
`6eb88630f20f99aac4d64356974b5a18345c3673`, after the fail-closed target binding
repair. Accepted region-specific cabinet examples now bind to explicit region
sites, for example `wooden_cabinet_1_cabinet_top`, rather than generic cabinet
default sites. LIBERO-10 task 5 remains unresolved: `back_compartment_of_caddy`
has no accepted explicit target binding in this pass and is reported as
`TARGET_BINDING_AMBIGUOUS` instead of entering the primary Teacher denominator.

Additional committed audit files:

```text
tables/layer1_h2_20260620/resolver_region_repair_summary_20260620.csv
tables/layer1_h2_20260620/resolver_region_repair_fail_closed_rows_20260620.csv
reports/layer1_h2_20260620/resolver_region_repair_summary_20260620.json
tables/layer1_h2_20260620/teacher_only_overlay_timeline_manifest.csv
reports/layer1_h2_20260620/teacher_only_overlay_timeline_summary.json
tables/layer1_h2_20260620/teacher_timelines/*.csv
```

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
- The current 24-row package is diagnostic-only and cannot be used as the final
  unbiased human-validation set.
- Supplementary multi/mixed mechanisms are not promoted to the primary
  denominator.
- Some primary episodes remain target-binding ambiguous.
- Timing alignment has a freeze-candidate contract, but final Teacher freeze
  still needs a new final blind package and independent human review.
- Event proposal tables remain proposal evidence only. They must not be
  promoted to frozen Teacher labels before H2 approval and human review.
- Teacher-only overlays/timelines are populated for the diagnostic package only;
  the final unbiased blind package has not been selected or generated.

## Allowed Claims

- Layer 1 resolver-v1 now uses structured binding plus sim-state physical event
  evidence.
- A deterministic development-canary manifest and disjoint diagnostic holdout
  manifest were generated from the frozen CLEAN300 deep-integrity ledger.
- Resolver-v1 produced non-empty dev/diagnostic physical-event proposals with
  zero internal validation errors.
- A diagnostic holdout human-review package was generated without detector
  fields, attack outputs, or task-success leakage in the reviewer queue.
- Region/directional target aliases now fail closed instead of falling back to
  generic default sites.

## Forbidden Claims

- Do not claim full CLEAN300 Teacher labels exist.
- Do not claim human review is complete.
- Do not claim the current 24-row diagnostic holdout is an unbiased final blind
  validation set.
- Do not treat pre-repair proposal rows as final Teacher labels after the
  region-specific binding rule changed.
- Do not claim Layer 2 timing transfer, detector localization, VIS/RAND, or
  attack effectiveness.
- Do not use these proposal labels as frozen Teacher labels before H2 approval
  and human review.

## Stop Point

```text
HUMAN GATE H2_LAYER1_FREEZE
```
