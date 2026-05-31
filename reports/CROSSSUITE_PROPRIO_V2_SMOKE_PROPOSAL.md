# CrossSuite-ProprioNoStep-v2 Smoke Proposal

Date: 2026-05-31

## Status

Proposal only. Do not train without explicit approval.

## Data Gate

`tables/crosssuite_proprio_dataset_index.csv` now has enough full-feature rows for a limited offline smoke:

- Object: 24 rows, 4 tasks, states 0-2
- Spatial: 20 rows, 10 tasks, states 0-1
- Goal: 12 rows, 6 tasks, states 0-1

All selected rows must use clean-only teacher labels and mechanism eligibility. No oracle/sus30/VIS/manual outcomes may be labels.

## Models

1. Object-ProprioNoStep frozen baseline
   - no retraining
   - used as reference only
2. CrossSuite-ProprioNoStep-v2-relative
   - inputs: relative EEF xyz, EEF velocity, gripper qpos/width/command, action dx/dy/dz/gripper
   - no `normalized_step`
   - no task_id/state_id/run_id feature
   - no privileged object/target pose
3. Task-only baseline
   - mechanism/task-language features only
   - must be weaker than proprio model
4. Label-shuffle baseline
   - same features with shuffled clean teacher labels
   - must be near chance

## Splits

Required:

- suite-holdout split
- task-holdout split
- episode/run-level split only
- no timestep-random split

## Gates

Object retention gate:

- v2 must not collapse Object trigger alignment relative to the frozen Object-ProprioNoStep reference

Spatial/Goal eligible subset gate:

- v2 must improve trigger alignment on mechanism-eligible Spatial/Goal subsets over Object-only frozen baseline

Leakage gate:

- no oracle/sus30/VIS labels
- no manual outcome labels
- no future timestep features
- no object pose or target pose as deployed input
- no hard-coded windows

## Claim Boundary

If run, this is detector-development evidence only. It is not cross-suite attack readiness and must not be used to launch cross-suite sus30 without a separate approval gate.
