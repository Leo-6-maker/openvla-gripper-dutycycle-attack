# VIS Contact Frame Collection Proposal

## Status

This is a proposal only. No rollout, attack, training, VIS, sus30, or detector-triggered run was launched by this planner.

Source audit: `tables\vis_contact_frame_selection_audit.csv`

Planned rows: 3
Maximum clean-only rollouts if explicitly approved: 3

## Selected Contact Candidates

- pick up the ketchup and place it in the basket state 0: target policy step 98, frames 96..100
- pick up the tomato sauce and place it in the basket state 0: target policy step 134, frames 132..136
- pick up the cream cheese and place it in the basket state 0: target policy step 143, frames 141..145

## Collection Boundary

- Collection mode: clean-only contact frame dump.
- Attack enabled: false.
- VIS enabled: false.
- sus30 enabled: false.
- Detector enabled: false.
- Official config: `libero_object`, `num_steps_wait=10`, `max_steps=280`, center crop, official PIL/Lanczos preprocessing, postprocessed gripper.
- GPU policy in generated commands: `CUDA_VISIBLE_DEVICES=4,5`; local render GPU id 0 maps to physical GPU 4, not physical GPU 0.
- Existing `scripts/run_official_eval_artifact_rich.py` saves RGB as `frames/step_####.png` using policy-step indexing.

## Why This Is Needed

The prior saved VIS diagnostic frames are available but correspond to wait/pre-policy rows. The selector found contact/carry candidates in existing Object artifacts, but those contact candidates do not have RGB frame files. This proposal collects only the missing clean frames needed to rerun one-frame VIS diagnostics at verified contact/carry timesteps.

## Approval Gate

Do not execute the generated commands until explicitly approved. If approved, start with the three planned state-0 clean episodes only and verify that the expected contact-frame PNG files exist before any further VIS diagnostic.
