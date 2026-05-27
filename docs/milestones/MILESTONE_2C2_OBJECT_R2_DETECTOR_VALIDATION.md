# Milestone 2C.2 — Object R2 Clean Integration + Detector External Validation

## Purpose

Integrate the repaired Object R2 clean denominator (post-PR4 PIL preprocessing, 80/100) into the detector pipeline and externally validate the existing Milestone 2C proprio-only student against the new clean data.

## Key Results

### Object Clean Comparison

| | SR | Source |
|---|---|---|
| Old v4 (TF JPEG) | 68/100 | detector_dev_audit_20260526 |
| Old v4 (horizon-fixed) | 71/100 | MJ38 rerun |
| **Object R2 (PIL Lanczos)** | **80/100** | R3 patched v4 eval |
| Corrected official script | 80/100 | official eval |

**Delta: +12 points over old v4, matches corrected official exactly.**

### Task-Level Changes

| Task | Old v4 | R2 PIL | Delta | Status |
|------|--------|--------|-------|--------|
| cream_cheese | 0.50 | 1.00 | +0.50 | Substantially improved |
| chocolate_pudding | 0.60 | 0.90 | +0.30 | Substantially improved |
| salad_dressing | 0.70 | 0.90 | +0.20 | Improved |
| tomato_sauce | 0.70 | 0.90 | +0.20 | Improved |
| butter | 0.60 | 0.80 | +0.20 | Improved |
| ketchup | 0.90 | 1.00 | +0.10 | Improved |
| orange_juice | 0.70 | 0.80 | +0.10 | Improved |
| alphabet_soup | 0.80 | 0.60 | -0.20 | Regressed |
| milk | 0.90 | 0.70 | -0.20 | Regressed |
| bbq_sauce | 0.40 | 0.40 | 0.00 | Still unstable |

### Verdict

Object R2 is **substantially recovered** (SR >= 0.80) under the corrected official-eval-aligned PIL runner.

bbq_sauce remains a clean instability task (4/10) and should be excluded from attack pilot.

## Action Path Audit

| Attribute | Value |
|-----------|-------|
| Inference API | `model.generate()` |
| Action decoding | Manual v4 decode (not `predict_action()`) |
| action_path label | `generate_manual_decode` |
| Preprocess backend | `official_pil_lanczos` |
| JPEG round-trip | False |
| Prompt format | `In: What action should the robot take to {task}?\nOut:` |

## Detector Validation Status

| Check | Status | Reason |
|-------|--------|--------|
| Teacher detector on Object R2 | **BLOCKED** | No step_records/episode_records in official eval script output |
| Student replay on Object R2 | **BLOCKED** | No timestep-level features available |
| Teacher label regeneration | **BLOCKED** | Requires clean rerun with step_records logging |

## Limitations

1. **Per-state manifest incomplete**: Object R2 CSV was overwritten by parallel workers. Task-level SR reconstructed from logs.
2. **No step_records**: The official eval script (`tmp_official_eval.py`) writes only task-level summaries, not per-step records.
3. **Manifest source confidence**: Medium (task-level from logs) / Low (state-level missing).
4. **Teacher/student validation deferred**: Requires clean rerun with step_records logging enabled.

## Next Steps

1. Complete Spatial/Goal/LIBERO-10 clean runs.
2. Rerun clean Object with step_records logging to enable detector validation.
3. Regenerate teacher labels from step_records.
4. Run student external validation on Object R2.
5. Object can enter Milestone 3A as optional clean-stable candidates (exclude bbq_sauce).
