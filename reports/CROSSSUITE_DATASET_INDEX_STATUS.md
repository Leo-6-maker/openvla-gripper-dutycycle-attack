# CrossSuite Dataset Index Status

Date: 2026-05-31

## Inputs

```text
/data/liuyu/outputs/milestone_2b_parser_visual_linkage_20260526/tables/student_train_dataset.csv
/data/liuyu/outputs/libero_full4_clean_official_aligned_eager_10states_20260525
/data/liuyu/outputs/milestone_3a_crosssuite_proprio_shadow_20260531
/data/liuyu/outputs/table1_clean_detector_dev_audit_20260526/tables/teacher_window_labels.csv
```

No rollout was launched. No detector was trained.

## Output

```text
tables/crosssuite_proprio_dataset_index.csv
```

The index is episode-level to avoid committing large per-timestep artifacts. It records feature/label availability, mechanism eligibility, split candidacy, and one representative value for each deployed proprio feature.

## Counts

Total rows: 475 episode/run entries.

| Suite | Rows | Full EEF xyz | Full EEF velocity | Teacher labels | Mechanism eligible | Full split candidates | Partial EEF-z candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 131 | 20 | 20 | 120 | 99 | 20 | 100 |
| libero_object | 112 | 0 | 0 | 100 | 68 | 0 | 100 |
| libero_goal | 122 | 12 | 12 | 112 | 84 | 12 | 100 |
| libero_10 | 110 | 0 | 0 | 100 | 53 | 0 | 100 |

Split candidate summary:

- `yes`: 32
- `partial_eef_z_only`: 400
- `no`: 43

## Interpretation

The richer artifact index found full EEF xyz/velocity in cross-suite shadow artifacts for Spatial and Goal, and clean teacher labels/mechanism eligibility in the Table1 detector-development audit. However, Object still lacks full EEF x/y and x/y velocity in the available clean artifacts.

## Gate XS-2

Result: FAIL / BLOCKED FOR FULL CrossSuite-v2 TRAINING.

Reason:

- Full EEF xyz/velocity is available for some non-Object shadow entries.
- Full EEF xyz/velocity is not available for Object production-reference entries.
- Clean teacher labels are available for the 400 Table1 development episodes.
- Mechanism eligibility is available for the 400 Table1 development episodes.
- At least Object plus non-Object are usable only for `partial_eef_z_only`, not for full relative-EEF-xyz training.

Do not train CrossSuite-ProprioNoStep-v2 from this index unless the scope is explicitly narrowed to an EEF-z-only smoke or richer Object artifact-rich clean data is generated.
