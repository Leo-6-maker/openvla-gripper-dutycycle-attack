# CrossSuite-ProprioNoStep-v2 Experiment Plan

## Scope

CrossSuite-ProprioNoStep-v2 is a future detector-training line. It must be separate from the validated Object-ProprioNoStep production baseline.

## Non-Negotiable Inputs

- clean teacher labels only
- no oracle labels
- no sus30 outcome labels
- no VIS outcome labels
- no manual attack outcome labels
- no privileged state as deployed student input
- no `normalized_step`

## Proposed Features

- gripper command/qpos/width
- action dx/dy/dz/gripper
- EEF velocity
- causal relative EEF position from episode initial pose
- optional mechanism type from clean task parser

## Required Baselines

- frozen Object-ProprioNoStep baseline
- task-only baseline
- label-shuffle baseline
- time-only baseline if any time-like feature is considered

## Splits

- suite-holdout split
- task-holdout split
- Object retention split

No timestep-random split is acceptable for claims.

## Gates

- Object performance retention gate: v2 must not degrade the validated Object baseline beyond a predeclared tolerance.
- Spatial/Goal eligible subset gate: only mechanism-eligible clean-success tasks can be used for attack readiness.
- Abstain gate: uncertain mechanisms must abstain rather than trigger low-quality sus30 runs.

## Blocked Until

- relative feature audit is complete
- data index confirms feature and label availability
- baselines are implemented
- explicit approval is given to train v2
