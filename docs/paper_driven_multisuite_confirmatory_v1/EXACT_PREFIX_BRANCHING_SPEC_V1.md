# Exact Prefix Branching Spec V1

Status: PLANNING_ONLY

## Snapshot Identity

Every matched branch must bind:

| Field | Required |
|---|---|
| simulator_state_hash | yes |
| current_observation_hash | yes |
| policy_rng_state_hash | yes |
| detector_fsm_state_hash | yes |
| detector_feature_history_hash | yes |
| step_index | yes |
| prefix_action_hash | yes |
| prefix_observation_hash | yes |
| victim_checkpoint_sha | yes |
| detector_checkpoint_sha | yes |
| config_sha | yes |
| split_sha | yes |

## Determinism And Branching

- Clean replay tolerance: identical task-state hashes through branch point; numeric
  telemetry tolerance must be frozen before Gate D.
- Branch first frame: attack/control begins at `branch_step`, after the frozen
  prefix action at `branch_step - 1`.
- Off-by-one failures abort the branch family.

## Timing Controls

- Random-time exclusion margin: 10 steps on both sides of detector window.
- Random window distribution: uniform over legal K-length windows in the same
  episode with the same remaining horizon class.
- Random-time ineligible if no legal window exists.
- EARLY_SHIFT offset: `-10` steps from detector anchor.
- EARLY_SHIFT ineligible if the shifted K-window crosses episode start or a
  mechanism-illegal phase.
