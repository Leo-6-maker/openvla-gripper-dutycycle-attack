# Reproducing Milestone 2D Phase C.2 — Privileged Artifact-Rich Object Smoke

## Environment

- Python: `/data/aviary/envs/openvla_official_libero_20260525/bin/python`
- torch 2.2.0+cu121, timm 0.9.10, transformers 4.40.1
- Wheelhouse: `/data/liuyu/wheelhouse_openvla_official_20260525/`
- Polluted .pth files disabled, fallback .pth for libero/robosuite/mujoco
- See `reports/ENVIRONMENT_FIX_AUDIT.md` in Phase B output

## Code

- `scripts/run_official_eval_artifact_rich.py` — Artifact-rich runner with privileged state
- `src/utils/libero_privileged_state.py` — Privileged state extraction from obs dict + MuJoCo sim

## Parallel Launch Commands

### Worker A — milk (GPU 4,5)

```bash
CUDA_VISIBLE_DEVICES=4,5 MUJOCO_GL=egl PYTHONUNBUFFERED=1 python -u \
  scripts/run_official_eval_artifact_rich.py \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object --task_start 7 --task_count 1 \
  --num_trials_per_task 10 --worker_id c2_priv_milk_10_w0 \
  --save_rgb --save_step_records --save_privileged_teacher_state \
  --cuda_visible_devices 4,5 --render_gpu_device_id 4 \
  --output_root /data/liuyu/outputs/milestone_2d_phase_c2_privileged_artifact_rich_object_smoke_20260527 \
  --run_id_prefix c2_priv_milk
```

### Worker B — cream_cheese (GPU 2,6)

```bash
CUDA_VISIBLE_DEVICES=2,6 MUJOCO_GL=egl PYTHONUNBUFFERED=1 python -u \
  scripts/run_official_eval_artifact_rich.py \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object --task_start 1 --task_count 1 \
  --num_trials_per_task 10 --worker_id c2_priv_cream_cheese_10_w1 \
  --save_rgb --save_step_records --save_privileged_teacher_state \
  --cuda_visible_devices 2,6 --render_gpu_device_id 2 \
  --output_root /data/liuyu/outputs/milestone_2d_phase_c2_privileged_artifact_rich_object_smoke_20260527 \
  --run_id_prefix c2_priv_cream_cheese
```

### Worker C — bbq_sauce (GPU 1,3)

```bash
CUDA_VISIBLE_DEVICES=1,3 MUJOCO_GL=egl PYTHONUNBUFFERED=1 python -u \
  scripts/run_official_eval_artifact_rich.py \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object --task_start 3 --task_count 1 \
  --num_trials_per_task 10 --worker_id c2_priv_bbq_sauce_10_w2 \
  --save_rgb --save_step_records --save_privileged_teacher_state \
  --cuda_visible_devices 1,3 --render_gpu_device_id 1 \
  --output_root /data/liuyu/outputs/milestone_2d_phase_c2_privileged_artifact_rich_object_smoke_20260527 \
  --run_id_prefix c2_priv_bbq_sauce
```

## Post-Processing

```bash
# Aggregation
python scripts/aggregate_artifact_rich_manifests.py --output_root <ROOT> --overwrite

# Visual manifest
python scripts/build_visual_feature_manifest.py --artifact_root <ROOT>

# No-timestep export
python scripts/export_no_timestep_visual_proprio_dataset.py \
  --artifact_root <ROOT> --visual_feature_manifest <ROOT>/tables/visual_feature_manifest.csv \
  --output_root <ROOT>

# Teacher detector (requires privileged state adapter)
python /tmp/teacher_detector.py

# Labeled export
python /tmp/labeled_export.py
```

## Boundary Statement

This reproduction procedure generates privileged-state clean artifacts and teacher labels.
It does not run attacks. It does not train detectors. Privileged state is teacher-only.
