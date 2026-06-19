# Cross-Suite Layer 1 Resolver Preregistration

## Decision Boundary

```text
LAYER1_STAGE = PREREG_ONLY
FULL_RESOLVER = NOT_RUN
DETECTOR_TELEMETRY_READ = FORBIDDEN
VIS_RAND_ATTACK = NO_GO
```

This preregistration defines the offline privileged ontology, output schema, and
initial task mechanism policy for Cross-Suite CLEAN300. It does not generate
Teacher labels, does not evaluate Layer 2 timing transfer, and does not authorize
Layer 3 attack execution.

## Frozen Inputs

```text
CLEAN300 freeze tag: freeze/cross-suite-clean300-20260619
PR #30 merge commit: 141657fdc5d85c5fd564913c955d61e9e6be9ddc
collector_source_commit: 63793972743f667c6a6bcc12e9700f322f261147
audit_tool_commit: 2d3ff1c02995eb8090db4e3604d6fa1ad3f7a3dd
final_evidence_commit: 6f9139ada8ddf1dcf59d1f7cb0e1e1c0e87372c1
valid CLEAN episodes: 300
clean success/failure: 197 / 103
```

## Forbidden Inputs During Layer 1

The resolver and ontology must not read:

```text
mlp_emit_step
mlp_triggered
corridor_p
release_p
pred_phase
detector overlay
VIS/RAND/shuffled outputs
attack outcome
```

Raw clean video may be used only for later blind human review packages without
detector overlays.

## Mechanism Policy

Primary positive timing-transfer denominator:

```text
single_object_pick_place
```

Supplementary event-level audit only:

```text
multi_object_transfer
mixed_articulated_pick_place
```

Negative or semantic-abstention denominator:

```text
articulated_only
push_or_planar
rearrangement_non_grasp
unknown_or_low_signal
binding_ambiguous
```

Correct abstention is a valid Layer 1 outcome. Unsupported mechanisms must not
be converted into positive timing labels to improve denominator size.

## Ontology Summary

The committed ontology covers all 30 suite/task pairs:

```text
libero_spatial: 10 single_object_pick_place
libero_goal: 6 single_object_pick_place, 3 articulated_only, 1 push_or_planar
libero_10: 1 single_object_pick_place, 3 multi_object_transfer, 4 articulated_only, 2 unknown_or_low_signal
```

Primary positive tasks currently preregistered:

```text
Spatial: task 0-9
Goal: task 1,2,4,6,8,9
LIBERO-10: task 5
```

LIBERO-10 multi-object tasks 0, 1, and 7 are event-level audit only. They may
produce separate event rows after resolver support and human review, but they do
not enter the primary positive denominator by default.

## Binding Source Priority

```text
1. explicit BDDL semantics and ontology aliases
2. exact body/site/joint alias match
3. validated structured fallback
4. fail-closed abstention
```

Unconstrained fuzzy string matching is prohibited as a final positive binding
rule.

## Initial Output Contract

Derived outputs must be versioned and separate from collector artifacts:

```text
privileged_sidecar_resolved_v1.json
teacher_event_labels_v1.csv
```

The original collector `privileged_sidecar.json` must not be overwritten.

## Development Canary Plan

After H1 approval, select a detector-independent development canary by a stable
hash rule over `(suite, task_idx, state_id, source_episode_sha)`:

```text
Spatial: 4 single-object episodes
Goal: 2 eligible pick-place + 2 negative articulated/push episodes
LIBERO-10: 2 single-event + 2 multi/mixed-event episodes
```

The canary manifest must be committed before resolver outputs are inspected.
Resolver fixes may use MuJoCo/BDDL semantics, never Layer 2 emit behavior.

## Blind Validation Plan

After resolver v1 is frozen, select a disjoint 18-24 episode blind validation
set stratified by suite, mechanism, task, clean success/failure, and
single/multi-event status. The review package must hide detector outputs and
include raw video, Teacher-only overlays, proposed object/target bindings, event
proposals, and a review CSV.

Codex must stop for real human labels and must not self-certify manual review.

## H1 Gate Checklist

```text
30/30 ontology rows exist
YAML/schema parse passes
ontology does not define temporal trigger windows
forbidden detector fields are not used as resolver inputs
mechanism positive/supplementary/negative sets are explicit
multi-event policy is explicit
schema status classes are frozen
CPU tests pass
full resolver not run
GPU/VIS/RAND/attack not run
```

## Allowed Claims

- Layer 1 ontology/schema/preregistration are defined for review.
- The ontology covers all 30 task identities and explicitly abstains or marks
audit-only where semantics are unsupported.

## Forbidden Claims

- No Teacher labels are generated yet.
- No Layer 2 zero-shot timing transfer is evaluated.
- No Layer 3 payload or attack result is evaluated.
- No manual review is complete.
