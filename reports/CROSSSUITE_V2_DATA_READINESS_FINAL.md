# CrossSuite-v2 Data Readiness

**Date**: 2026-06-01 | **Status**: Data partially available — training deferred

## Available

- Cross-suite shadow step_records: 32 episodes (Spatial 20, Goal 12)
- Object reference step_records: 50 episodes
- Feature schema audit: complete (eef_z 5.7x shift confirmed)
- Mechanism inventory: complete

## Missing

- `crosssuite_proprio_dataset_index.csv`: Not built (scripts in PR #5 but not executed)
- Per-step teacher labels for Spatial/Goal: Not available in step_records
- Clean teacher label linkage: Object-100 labels exist but not linked to cross-suite episodes

## Requirements for v2 Training

1. Build per-step proprio dataset from cross-suite step_records
2. Implement relative EEF features (eef_xyz - initial_eef_xyz)
3. Obtain clean teacher labels or use heuristic contact detection
4. Design task/suite holdout split

## Decision

**Do not train CrossSuite-v2 offline smoke** — dataset builders need to be run first. Write requirements only.
