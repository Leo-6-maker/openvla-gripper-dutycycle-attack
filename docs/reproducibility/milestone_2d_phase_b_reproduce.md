# Reproducing Milestone 2D Phase B — Artifact-Rich Object Smoke

## Environment Requirements

- Python: `/data/aviary/envs/openvla_official_libero_20260525/bin/python`
- torch 2.2.0+cu121, timm 0.9.10, transformers 4.40.1
- Wheelhouse at `/data/liuyu/wheelhouse_openvla_official_20260525/`
- Libero/robosuite/mujoco from openvla_sparse fallback

### Known .pth Pollution Fix

Before running, disable polluting .pth files in the conda env's site-packages:

```bash
cd /data/aviary/envs/openvla_official_libero_20260525/lib/python3.10/site-packages
mv z_conda_py310_extra.pth z_conda_py310_extra.pth.disabled
mv z_openvla_sparse_extra.pth z_openvla_sparse_extra.pth.disabled
```

Then install correct packages from wheelhouse:

```bash
PIP=/data/aviary/envs/openvla_official_libero_20260525/bin/pip
WH=/data/liuyu/wheelhouse_openvla_official_20260525
$PIP install --no-index --no-deps --find-links $WH/torch torch==2.2.0+cu121
$PIP install --no-index --no-deps --find-links $WH/pypi timm==0.9.10 accelerate==0.29.3 ...
```

Add fallback for non-conflicting packages:

```bash
echo "/data/aviary/envs/openvla_sparse/lib/python3.10/site-packages" > \
  z_openvla_sparse_fallback.pth
```

## Artifact-Rich Runner Command

```bash
CUDA_VISIBLE_DEVICES=<GPU_PAIR> MUJOCO_GL=egl \
  /data/aviary/envs/openvla_official_libero_20260525/bin/python \
  scripts/run_official_eval_artifact_rich.py \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --task_start <TASK_IDX> \
  --task_count 1 \
  --num_trials_per_task 10 \
  --worker_id <UNIQUE_WORKER_ID> \
  --cuda_visible_devices <GPU_PAIR> \
  --render_gpu_device_id <RENDER_GPU_ID> \
  --output_root /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527 \
  --run_id_prefix <TASK_PREFIX>
```

Task indices: milk=7, cream_cheese=1, bbq_sauce=3.

## Post-Processing Commands

### Aggregation

```bash
python scripts/aggregate_artifact_rich_manifests.py \
  --output_root /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527 \
  --overwrite
```

### Visual Feature Manifest

```bash
python scripts/build_visual_feature_manifest.py \
  --artifact_root /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527
```

### No-Timestep Export

```bash
python scripts/export_no_timestep_visual_proprio_dataset.py \
  --artifact_root /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527 \
  --visual_feature_manifest /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527/tables/visual_feature_manifest.csv \
  --output_root /data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527
```

## Boundary Statement

This reproduction procedure generates artifact-rich clean data for detector calibration.
It does not run attacks.
It does not train detectors.
It does not use attack/oracle/random/manual outcomes.
