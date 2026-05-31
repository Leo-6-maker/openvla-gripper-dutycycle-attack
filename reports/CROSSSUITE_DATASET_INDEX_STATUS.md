# CrossSuite Dataset Index Status

Date: 2026-05-31

## Inputs

```text
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv
/data/liuyu/outputs/libero_full4_clean_official_aligned_eager_10states_20260525
/data/liuyu/outputs/milestone_3a_crosssuite_proprio_shadow_20260531
/data/liuyu/outputs/milestone_2i_visual_fusion_online_detector_pilot_20260530
/data/liuyu/outputs/milestone_2j_visual_fusion_online_pilot_v5_20260530
/data/liuyu/outputs/table1_clean_detector_dev_audit_20260526/tables/teacher_window_labels.csv
```

The index builder skips obvious non-clean artifact paths such as `sus30`, `oracle`, `random`, and `attack` unless the path is explicitly a clean condition. No rollout or training was launched.

## Output

```text
tables/crosssuite_proprio_dataset_index.csv
```

## Counts

Total rows: 499 episode/run entries.

| Suite | Rows | Full EEF xyz | Full EEF velocity | Teacher labels | Mechanism eligible | Full split candidates | Partial EEF-z candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 131 | 20 | 20 | 120 | 99 | 20 | 100 |
| libero_object | 136 | 24 | 24 | 124 | 88 | 24 | 100 |
| libero_goal | 122 | 12 | 12 | 112 | 84 | 12 | 100 |
| libero_10 | 110 | 0 | 0 | 100 | 53 | 0 | 100 |

Split candidate summary:

- `yes`: 56
- `partial_eef_z_only`: 400
- `no`: 43

Full-feature smoke candidate coverage:

- Object: 24 rows, 4 tasks, states 0-2
- Spatial: 20 rows, 10 tasks, states 0-1
- Goal: 12 rows, 6 tasks, states 0-1
- LIBERO-10: 0 full-feature rows

## Gate XS-2

Result: PASS for a limited offline smoke proposal only.

The index now has full EEF xyz/velocity, mechanism eligibility, and clean teacher labels for Object plus non-Object subsets. This is sufficient to design a small CrossSuite-ProprioNoStep-v2 smoke experiment with strict controls.

It is not sufficient for a production replacement or broad CrossSuite claim because:

- full-feature Object coverage is limited to 4 tasks and states 0-2
- LIBERO-10 has no full-feature rows
- source roots are heterogeneous
- Object clean reproducibility remains a caveat

## Decision

CrossSuite-v2 training is still not started. A smoke-training proposal may be written, but it must include task-only and label-shuffle controls, suite/task holdout splits, and Object retention gates.
