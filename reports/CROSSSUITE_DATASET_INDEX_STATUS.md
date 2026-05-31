# CrossSuite Dataset Index Status

Date: 2026-05-31

## Input

```text
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv
```

## Output

```text
tables/crosssuite_proprio_dataset_index.csv
```

## Episode Counts

- libero_spatial: 100 episodes
- libero_object: 100 episodes
- libero_goal: 100 episodes
- libero_10: 100 episodes
- total: 400 episodes

## Label Availability

Teacher labels are present for all 400 indexed episodes:

- `teacher_phase`
- `teacher_hazard`
- `teacher_release_safe`

## Feature Availability

All suites have partial proprio coverage suitable for metadata indexing:

- `eef_z`
- `gripper_qpos`
- `gripper_width`
- `action_gripper`

Missing for all indexed episodes:

- `eef_x`
- `eef_y`
- `eef_vx`
- `eef_vy`

## Split Candidate Status

All 400 episodes are marked:

```text
partial_eef_z_only
```

This is enough for a future relative-EEF-z smoke design, but not enough for a full relative-EEF-xyz CrossSuite-v2 claim.

## Gate XS-2

Result: PARTIAL / BLOCKED FOR TRAINING.

Metadata and clean teacher labels exist, but full feature coverage is incomplete and `mechanism_eligible` is absent from the 2B student dataset. Do not train CrossSuite-v2 until a richer artifact/label index is available or the v2 scope is explicitly narrowed to EEF-z-only.
