# Milestone 2D Phase C.2 — Privileged Artifact-Rich Object Smoke

## Purpose

Rerun the Phase B Object artifact-rich smoke with corrected privileged simulation-state logging. Phase B had empty object_pose/target_pose because the runner tried `obs.get("object_pos")` but LIBERO uses keys like `{object}_1_pos`. Phase C.1 fixed the extraction; Phase C.2 reruns all 30 episodes with working privileged state.

## Why Phase B Was Insufficient

- `object_pose_json` all empty
- `target_pose_json` all empty
- `teacher_privileged_state_available` always false
- Teacher detector could only use gripper+EEF signals (reduced confidence)

## C.1 Micro Smoke Proof

1-episode milk s0 micro smoke confirmed:
- `teacher_privileged_state_available=true` on all policy steps
- `object_pose_json`, `target_pose_json` non-empty
- `object_to_target_distance`, `object_eef_distance` non-empty
- `debug/sim_names.json` correctly matched `milk_1_main` and `basket_1_main`

## C.2 Parallel Rerun

| Worker | GPU | Task | Worker ID |
|--------|-----|------|-----------|
| A | 4,5 | milk | c2_priv_milk_10_w0 |
| B | 2,6 | cream_cheese | c2_priv_cream_cheese_10_w1 |
| C | 1,3 | bbq_sauce | c2_priv_bbq_sauce_10_w2 |

All three ran in parallel with `--save_privileged_teacher_state`.

## Task SR

| Task | N | Success | SR |
|------|---|---------|-----|
| milk | 10 | 9 | 0.900 |
| cream_cheese | 10 | 10 | 1.000 |
| bbq_sauce | 10 | 3 | 0.300 |
| **Total** | **30** | **22** | **0.733** |

## Privileged State Coverage

All 5,604 policy steps have 100% privileged state coverage: object_pose, target_pose, distance, eef_distance. Zero extraction errors. 30/30 sim_names.json present.

## Teacher Labels

- Windows detected: 22/30 (22/22 on success, 0/8 on failure)
- Zero false-positive windows on failed episodes
- bbq_sauce failures correctly marked as `clean_unstable` / `no_lift_detected`
- label_source = `clean_only_privileged_teacher`
- uses_privileged_sim_state = true

## Labeled No-Timestep Dataset

- 5,604 rows, 100% phase-labeled
- 38 hazard-positive, 256 release-safe steps
- normalized_step absent from features
- Zero forbidden feature leakage

## Remaining Blockers

- Frozen visual features (Milestone 2E)
- Larger artifact-rich dataset for generalization
- ProprioNoStep-TCN smoke training
- VisualProprioNoStep-TCN training
- Attack pilot not yet run

## Boundary Statement

Milestone 2D Phase C.2 produces privileged-state clean artifacts and clean-only teacher labels for no-timestep detector development. It does not run attacks. It does not use VIS/random/oracle/manual outcomes. Privileged object/target state is used only for teacher labeling and must not enter deployed student features.
