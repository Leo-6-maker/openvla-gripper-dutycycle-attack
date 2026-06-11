# Runner Taxonomy and Claim Boundary

**Date**: 2026-06-11

## Claim Boundary

- Phase-runner evidence supports Level 1/2 command/qpos bridge claims only.
- Official/v4 runner evidence is required for official clean success and any Level-3 task-effect audit.
- S20c fixed-window L3 runner remains pending until center-crop and preprocessing consistency are proven end to end.

## Runner Table

| Runner | Safe claim level | Valid for | Not valid for | Required config |
|---|---|---|---|---|
| Phase runner | Level 1/2 only | generated OPEN command; matched VIS/RAND command comparison; qpos physical bridge | official task success/failure; task-level SR; Level-3 task effect unless separately aligned | N/A for Level 3 official claims |
| Official/v4 runner | Official clean baseline and candidate Level 3 audit source | official-aligned clean success; state-level success; Level-3 task/contact audit if fixed-window attack is integrated correctly | phase-runner-only command bridge unless attack integration logs matched command/qpos fields | --center_crop; --libero_preprocess_backend official_pil_lanczos; --postprocess_gripper; --success_metric check_success; --num_steps_wait 10; attention_backend=eager when applicable |
| S20c official fixed-window L3 runner | Pending; audit before use | Potential Level-3 fixed-window smoke only after center_crop and preprocess kwargs are proven consistent | Any result from center_crop-missing smoke; phase-runner task-effect claim | center_crop=True end-to-end; official_pil_lanczos; postprocess_gripper; env.check_success; actual task/state provenance |
