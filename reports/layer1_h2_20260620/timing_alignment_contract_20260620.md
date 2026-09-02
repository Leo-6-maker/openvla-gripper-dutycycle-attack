# H2 Timing Alignment Contract

Status: `FREEZE_CANDIDATE_PENDING_H2_REVIEW`

Scope: six accepted development-canary event proposals.

Evidence table:
`tables/layer1_h2_20260620/timing_alignment_contract_20260620.csv`

Frozen convention:

```text
sim_state_timing_convention = one_sim_state_row_per_executed_action_step
sim_row_to_action_step_offset = 0
video_frame_to_step_offset = 0
```

Manual/metadata alignment check:

```text
reviewed_episode_count = 6
step_telemetry_rows == frame_index_rows == sim_state_stream.body_xpos rows == rollout_raw.mp4 frames
event close/grasp/lift/carry/window endpoints are in-range for all reviewed episodes
```

Claim boundary:

This freezes the indexing convention for Teacher proposal review. It does not
complete independent human review, does not validate object/target semantics by
itself, and does not authorize Layer2, VIS/RAND, shuffled, attack, or full
CLEAN300 resolver execution.
