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
| `teacher_run` | yes | Stable run identifier for the derived labeling pass. |
| `teacher_version` | yes | Teacher implementation version, initially `cross_suite_teacher_v1`. |
| `ontology_version` | yes | Ontology version, initially `cross_suite_task_ontology_v1`. |
| `resolver_version` | yes | Resolver code version/SHA. |
| `source_episode_sha` | yes | SHA or recursive artifact hash for the source CLEAN episode. |
| `mechanism_type` | yes | One of the ontology mechanism classes. |
| `mechanism_eligible` | yes | Boolean eligibility for positive timing-transfer evaluation. |
| `object_binding_status` | yes | Object binding result. |
| `target_binding_status` | yes | Target/site binding result. |
| `teacher_status` | yes | One of the allowed status classes below. |
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
| `event_valid` | yes | Boolean event validity. |
| `event_invalid_reason` | yes if invalid | Reason for invalid event row. |

## Allowed Status Classes

```text
ELIGIBLE_EVENT
CORRECT_SEMANTIC_ABSTAIN
NO_RELEVANT_GRASP_EVENT
OBJECT_BINDING_AMBIGUOUS
TARGET_BINDING_AMBIGUOUS
MULTI_EVENT_AUDIT_ONLY
RESOLVER_FAILED
SCHEMA_INVALID
```

## Binding Source Priority

1. Explicit BDDL semantics and ontology aliases.
2. Exact MuJoCo body/site/joint alias match.
3. Validated structured fallback.
4. Fail-closed abstention.

Unconstrained fuzzy matching is not allowed as a final positive binding rule.
