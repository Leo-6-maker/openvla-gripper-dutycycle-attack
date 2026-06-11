# Server Source-of-Truth Registry

**Date**: 2026-06-11

## Accepted Level 1/2 Registry

- `tomato_sauce_s0_w70-80`: phase-runner command/qpos bridge anchor; 5/6 PHYS_PASS plus 1 borderline.
- `ketchup_s0_w150-160`: S19 2/3 + S20a 3/3 fresh under explicit random_control_seed; combined 5/6 PHYS_PASS plus 1 RAND-confounded.

These are Level 1/2 physical bridge claims only, not official Level-3 task-effect evidence.

## Current Level 3 Status

- Level 3 is **not established**.
- S20b phase-runner full-episode videos are archived as diagnostics only.
- Official-aligned fixed-window L3 remains pending on verified clean-success states.
- S20c center-crop-missing smoke must not be treated as official-aligned.

## Official Clean Baseline Source of Truth

- R1 official raw per-state manifest: `/data/liuyu/outputs/milestone_r1_official_eval_20260526/tables/object_official_script_100_manifest.csv`.
- R2 official/v4 reconstructed comparison: `/data/liuyu/outputs/milestone_r2_official_v4_object_alignment_20260526/tables`.
- V4 raw full10x10 per-state dirs: `/data/liuyu/outputs/milestone_1d_object_mujoco237_compat_20260526/object_full_10x10`.
- Use `source_confidence=high` rows for state-level claims. Low-confidence W45 reconstructed rows are task-SR inferred only.

## Runner Claim Boundary

- Phase runner: Level 1/2 command/qpos bridge only.
- Official/v4 runner: clean state success and future Level-3 official task/contact audit.
- S20c runner: pending until center_crop/preprocess/action/success provenance is verified for the exact run.

## Next Safe Experiment

Run an official-aligned fixed-window L3 audit only on verified clean-success states, using the official/v4 path with `center_crop=True`, `official_pil_lanczos`, `postprocess_gripper=True`, `success_metric=check_success`, `num_steps_wait=10`, and explicit state IDs.

Initial shortlist:

- rank 1: ketchup state1 steps=157 confidence=high
- rank 2: tomato_sauce state3 steps=135 confidence=high
- rank 3: ketchup state3 steps=142 confidence=high
- rank 4: tomato_sauce state5 steps=171 confidence=high
- rank 5: ketchup state0 steps=155 confidence=high
- rank 6: ketchup state2 steps=160 confidence=high
- rank 7: ketchup state4 steps=155 confidence=high
- rank 8: ketchup state5 steps=142 confidence=high
