# Cross-Suite Teacher Label Schema v1

This schema defines derived Layer 1 Teacher outputs. It must not overwrite the
collector's original `privileged_sidecar.json`; derived outputs are versioned as
`privileged_sidecar_resolved_v1.json` and `teacher_event_labels_v1.csv`.

## Forbidden Inputs

Layer 1 resolver implementations and preregistration must not read detector or
attack fields: `mlp_emit_step`, `mlp_triggered`, `corridor_p`, `release_p`,
`pred_phase`, detector overlays, VIS/RAND/shuffled outputs, or attack outcomes.

## Episode-Level Fields

| field | required | description |
| --- | --- | --- |
| `teacher_executed` | yes | Boolean; whether the resolver ran for this episode. |
| `teacher_run_id` | yes | Stable run identifier for the derived labeling pass. Empty only when `teacher_executed=false`. |
| `teacher_version` | yes | Teacher implementation version, initially `cross_suite_teacher_v1`. |
| `ontology_version` | yes | Ontology version, initially `cross_suite_task_ontology_v1`. |
| `resolver_version` | yes | Resolver code version/SHA. |
| `episode_key` | yes | Canonical key `suite|task_idx|state_id|eval_seed|condition`. |
| `suite` | yes | Source suite. |
| `task_idx` | yes | Integer task id within suite. |
| `state_id` | yes | Integer initial state id. |
| `source_episode_relpath` | yes | Episode path relative to the frozen CLEAN300 root registry. |
| `source_episode_abspath_audit_only` | yes | Absolute source path for audit traceability only; reviewer-facing queues must not depend on it. |
| `source_episode_sha` | yes | SHA or recursive artifact hash for the source CLEAN episode. |
| `mechanism_type` | yes | One of the ontology mechanism classes. |
| `mechanism_eligible` | yes | Boolean eligibility for positive timing-transfer evaluation. |
| `object_binding_status` | yes | Object binding enum. |
| `target_binding_status` | yes | Target/site binding enum. |
| `teacher_status` | yes | One of the allowed status classes below. |
| `primary_teacher_status` | yes | Primary single-object Teacher status. `NOT_PRIMARY_DENOMINATOR` for supplementary-only mechanisms. |
| `supplementary_teacher_status` | yes | Supplementary event status. `NOT_APPLICABLE` for primary-only mechanisms. |
| `label_role` | yes | One of `primary_single_object_pick_place`, `supplementary_multievent_grasp_carry_bridge`, `negative_only`, or `ignore`. |
| `primary_or_supplementary` | yes | One of `primary`, `supplementary`, `negative`, or `ignore`. |
| `primary_supplementary_event_id` | yes | Selected supplementary event id for one-shot training/attack semantics; empty outside `SUPPLEMENTARY_EVENT_ELIGIBLE`. |
| `teacher_semantic_abstain` | yes | Boolean semantic abstention decision from resolver, not collector placeholder. |
| `abstain_reason` | yes when abstaining | Reason for abstention or empty string for positive valid events. |
| `event_count` | yes | Number of emitted event rows. |
| `manual_review_required` | yes | Whether human review is required before accepting the label. |

## Event-Level Fields

| field | required | description |
| --- | --- | --- |
| `event_id` | yes | Stable event id within episode. |
| `object_body_name` | yes for positive event | MuJoCo body name for manipulated object. |
| `object_joint_name` | optional | Joint name if applicable. |
| `target_body_or_site_name` | yes for positive event | MuJoCo body or site target. |
| `binding_source` | yes | BDDL/ontology/exact-name/fallback/abstain. |
| `binding_confidence_class` | yes | `high`, `medium`, `low`, or `abstain`. |
| `close_onset_step` | optional | First relevant close command onset. |
| `grasp_established_step` | optional | First stable grasp evidence. |
| `lift_onset_step` | optional | First lift/carry evidence. |
| `stable_carry_start` | optional | Start of stable carry segment. |
| `teacher_window_start` | optional | Start of Teacher valid timing window. |
| `teacher_anchor_step` | optional | Primary anchor step for Layer 2 timing. |
| `teacher_window_end` | optional | End of Teacher valid timing window. |
| `release_onset_step` | optional | Release/opening onset. |
| `target_proximity_step` | optional | First step where the manipulated object is near the bound target. |
| `object_gripper_separation_step` | optional | First post-carry step where object-gripper distance exceeds the separation threshold. |
| `placement_complete` | yes | Boolean placement/outcome metadata; not required for event existence. |
| `object_gripper_min_distance` | optional | Minimum object-to-gripper proxy distance used by the resolver. |
| `object_target_min_distance` | optional | Minimum object-to-target proxy distance used by the resolver. |
| `supplementary_event` | yes | Boolean; true for multi/mixed mechanism event proposals that are not primary denominator labels. |
| `event_valid` | yes | Boolean event validity. |
| `event_invalid_reason` | yes if invalid | Reason for invalid event row. |

## Allowed Binding Status Enums

```text
BOUND_EXACT
BOUND_BDDL_ONTOLOGY
BOUND_STRUCTURED_FALLBACK
AMBIGUOUS
NOT_APPLICABLE
FAILED
```

## Allowed Teacher Status Classes

```text
ELIGIBLE_EVENT
SUPPLEMENTARY_EVENT_ELIGIBLE
CORRECT_SEMANTIC_ABSTAIN
NO_RELEVANT_GRASP_EVENT
OBJECT_BINDING_AMBIGUOUS
TARGET_BINDING_AMBIGUOUS
RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM
RESOLVER_FAILED
SCHEMA_INVALID
```

## Cross-Field Invariants

- teacher_executed=false implies `teacher_run_id=""`, `teacher_status=RESOLVER_FAILED`, `event_count=0`, and `manual_review_required=true`.
- teacher_status=ELIGIBLE_EVENT implies `mechanism_eligible=true`, `teacher_semantic_abstain=false`, `object_binding_status` and `target_binding_status` are one of `BOUND_EXACT`, `BOUND_BDDL_ONTOLOGY`, or `BOUND_STRUCTURED_FALLBACK`, and `event_count>=1`.
- ELIGIBLE_EVENT existence requires physical grasp, lift, and stable carry evidence. Target proximity and `placement_complete` are outcome/placement metadata, not event-existence requirements.
- teacher_status=SUPPLEMENTARY_EVENT_ELIGIBLE implies `mechanism_eligible=false`, `teacher_semantic_abstain=true`, `label_role=supplementary_multievent_grasp_carry_bridge`, `primary_or_supplementary=supplementary`, `event_count>=1`, and a nonempty `primary_supplementary_event_id`.
- SUPPLEMENTARY_EVENT_ELIGIBLE is restricted to `multi_object_transfer` or `mixed_articulated_pick_place`. It establishes supplementary grasp-carry timing only; it is never promoted into the primary single-object denominator.
- SUPPLEMENTARY_EVENT_ELIGIBLE requires unique manipulated-object physical evidence and stable carry evidence. Target binding is retained as evaluator metadata and is not required for supplementary timing.
- The selected `primary_supplementary_event_id` must be chosen without detector telemetry, attack results, task success, or visual ease: earliest `stable_carry_start`, then earliest `close_onset_step`, then lexical `object_body_name`, then lexical `event_id`.
- teacher_status=CORRECT_SEMANTIC_ABSTAIN implies `mechanism_eligible=false`, `teacher_semantic_abstain=true`, `event_count=0`, and a nonempty `abstain_reason`.
- `teacher_status in {OBJECT_BINDING_AMBIGUOUS,TARGET_BINDING_AMBIGUOUS,RESOLVER_FAILED,SCHEMA_INVALID}` implies `mechanism_eligible=false`, `manual_review_required=true`, and no accepted positive event rows.
- teacher_status=RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM implies a supplementary mechanism where the resolver could not produce reliable event segmentation; it must have `mechanism_eligible=false`, `teacher_semantic_abstain=true`, and `event_count=0`.

## Binding Source Priority

1. Explicit BDDL semantics and ontology aliases.
2. Exact MuJoCo body/site/joint alias match.
3. Validated structured fallback.
4. Fail-closed abstention.

Unconstrained fuzzy matching is not allowed as a final positive binding rule.

## Physics Contract

Resolver thresholds are versioned in `configs/cross_suite_teacher_physics_v1.yaml`.
Threshold tuning is allowed only on development canaries plus manual physical
review before H2 freeze. Blind sets, full CLEAN300 labels, and Layer 2 outcomes
must not be used to tune physics thresholds.
