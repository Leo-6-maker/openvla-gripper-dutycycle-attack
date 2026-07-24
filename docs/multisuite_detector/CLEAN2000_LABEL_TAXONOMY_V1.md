# CLEAN2000 Label Taxonomy V1

## Primary Event Definition

**First secure object acquisition that initiates task-relevant transport.**

The detector's positive class is: a time window beginning at the step where the robot achieves secure control of the target object sufficient to begin task-relevant transport, and extending through the carry/pre-release phase.

## Unified Phase Labels

| Phase | Description |
|-------|-------------|
| `approach` | Robot moving toward target object |
| `grasp_close` | Gripper closing on object |
| `lift` | Object leaving support surface |
| `carry` | Object in transit to target location |
| `pre_release_hazard` | Object near release point, still under control |
| `release_safe` | Object placed at target |
| `other` | Non-manipulation steps |
| `clean_unstable` | Unclear phase transition |
| `unsupported_mechanism` | Mechanism not covered by primary event definition |

## LIBERO-10 Special Handling

LIBERO-10 tasks may contain:
- Single primary event (standard case)
- Multiple task-relevant transfer events (multi-stage tasks)
- Supplementary events
- Articulated-object tasks
- Unsupported mechanisms

### Primary Training Eligibility

Only episodes with a **clearly identifiable primary secure-acquisition event** mapping to the common definition are eligible for primary training.

### Excluded from Primary Positive Class

- `supplementary-only` events
- Multi-event episodes where event identity is ambiguous
- Articulated-object manipulation
- Planar (non-lift) manipulation
- `unsupported/ambiguous` mechanisms

### Required Event Metadata

```
event_id: unique event identifier
event_role: {primary, supplementary}
event_count: integer (per episode)
selected_event_policy: string
abstain_reason: string | null
```

## Window Semantics

- Fixed-length window anchored at teacher-identified event step
- Window length = K (same as detector runtime K parameter)
- Window defined BEFORE any attack outcome is known
- Same window definition across all four suites
- Unchanged across training, validation, and test

## Teacher Label Confidence

```
teacher_confidence: float in [0, 1]
threshold_for_valid: TBD (default 0.5)
```
Labels below confidence threshold go to abstention set.
