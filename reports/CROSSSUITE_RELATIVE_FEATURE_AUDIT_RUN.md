# CrossSuite Relative Feature Audit Run

Date: 2026-05-31

## Input

```text
tables/crosssuite_proprio_dataset_index.csv
```

The index now includes full-feature clean Object rows from visual-fusion clean pilot artifacts, plus Spatial/Goal shadow artifacts and the Table1 detector-development teacher labels.

## Output

```text
tables/crosssuite_relative_feature_audit.csv
```

## Finding

The previous all-suite relative-EEF-z result remains valid for the 400 Table1 development episodes. The richer index now also supports a limited full-EEF-xyz smoke comparison on:

- Object: 24 full-feature rows
- Spatial: 20 full-feature rows
- Goal: 12 full-feature rows

LIBERO-10 still has no full-feature rows in the current index.

## Gate XS-1

Result: PASS for limited offline smoke design.

Do not interpret this as production CrossSuite readiness. The next allowed step is only a smoke proposal using clean teacher labels and leakage controls.
