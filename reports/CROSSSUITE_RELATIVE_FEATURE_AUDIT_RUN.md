# CrossSuite Relative Feature Audit Run

Date: 2026-05-31

## Input

```text
tables/crosssuite_proprio_dataset_index.csv
```

This index combines:

- Milestone 2B student scaffold rows as episode-level metadata
- artifact-rich official clean run directories where available
- `milestone_3a_crosssuite_proprio_shadow_20260531` shadow artifacts for Spatial/Goal
- clean-only teacher window labels for mechanism eligibility

## Output

```text
tables/crosssuite_relative_feature_audit.csv
```

## Key Finding

The richer index confirms that full EEF xyz/velocity exists for a limited Spatial/Goal shadow subset, but not for Object. Therefore the previous relative-EEF-z conclusion remains the only cross-suite feature result supported across all four suites.

## Gate XS-1

Result: PARTIAL PASS.

Supported:

- relative-to-initial EEF-z can be audited across all 400 Table1 development episodes
- full EEF xyz/velocity can be audited for some Spatial/Goal shadow entries

Not supported:

- full relative-EEF-xyz CrossSuite-v2 training with Object retention gate
- full Object-vs-Spatial/Goal distribution comparison on xyz/velocity

CrossSuite-v2 full training remains blocked until Object artifact-rich clean data includes EEF x/y and velocity or the experiment is explicitly narrowed to EEF-z-only.
