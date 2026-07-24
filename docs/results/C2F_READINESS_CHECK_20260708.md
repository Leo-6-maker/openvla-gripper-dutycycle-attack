# C2f Readiness Check — 2026-07-08

**Status**: `C2F_SCAFFOLD_READY = PASS` | `C2F_ADAPTER_READY_FOR_SMOKE3 = PASS`

## File Inventory

| File | Status |
|---|---|
| `docs/detectors/C2F_OBSERVATION_LANGUAGE_DATA_SPEC.md` | Present |
| `scripts/stageb/collect_c2f_observation_clean_rollouts.py` | Present |
| `scripts/stageb/c2f_libero_openvla_adapter.py` | Present |
| `tools/multisuite_detector/materialize_c2f_frozen_embeddings.py` | Present |
| `tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py` | Present |
| `docs/detectors/D8F_25D_ONLY_CEILING_CLOSEOUT.md` | Present |

## Adapter Audit (commit `91f5937`)

| Check | Result |
|---|---|
| Import uses `scripts.stageb.collect_c2f_observation_clean_rollouts` | PASS |
| `gripper_qpos = qpos[7] + qpos[8]` (not `qpos.sum()`) | PASS |
| RGB capture failure raises `RuntimeError` (not silent zero) | PASS |
| TeacherLabeler no longer marks all `stable_carry` as primary | PASS |
| `_identify_grasped_object()` via body-gripper distance | PASS |
| `_object_matches_task_target()` via task language substring | PASS |
| Uncertain labeling defaults to `unsupported_or_abstain` | PASS |
| Wrong-object `stable_carry` → `distractor_or_setup` | PASS |
| Privileged values not in student `step_records` | PASS |

## Boundaries

- C2f output writes to new evidence root, not D7 root
- Student input excludes object pose, target pose, attack outcome
- Privileged state used only for teacher labels
- Detector trained on clean-only data

## Next

C2F_SMOKE3 ready to launch on idle GPU.
