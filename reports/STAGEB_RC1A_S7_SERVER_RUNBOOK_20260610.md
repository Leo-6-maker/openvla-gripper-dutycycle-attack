# S7 Server Runbook

**Server**: klfy-SYS-4028GR-TR2 (8× RTX 2080 Ti, user: liuyu)
**Date**: 2026-06-10

## Quick Access

```bash
ssh vla  # jump: scene@10.60.133.3 → liuyu@10.60.133.4
cd /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
```

## Environment

```bash
conda activate openvla_official_libero_20260525
# or direct path:
PY=/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
```

GPU rendering:
```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset DISPLAY
```

## GPU Allocation

| Pair | CUDA_VISIBLE_DEVICES | Status |
|------|---------------------|--------|
| worker_10 | 1,0 | primary |
| worker_45 | 4,5 | secondary |
| worker_26 | 2,6 | auxiliary |
| BLACKLIST | 3,7 | NEVER USE |

## Launch Template

```bash
cd /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
export CUDA_VISIBLE_DEVICES=<pair>
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl; unset DISPLAY
PYTHONPATH=src nohup $PY -u scripts/<name>.py --gpu_pair 0,1 > <log> 2>&1 &
```

## Key Output Dirs

```
/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/
├── action_logit_full/          # S6 action-logit extraction + readout
├── action_hidden_full/         # S7 hidden extraction (running)
├── action_hidden_smoke_4w/     # S7 hidden smoke (done)
├── action_hidden_smoke_gpu26_4w/ # GPU 2,6 hidden smoke (done)
├── action_logit_duplicate_gpu45/ # reproducibility sidecar
├── pipeline_v0_3_confirmation/ # S6 confirmation results
└── pipeline_v0_3_robustness_seed78/ # S6 robustness results
```

## EGL Anti-Pattern

```python
# WRONG — uses CUDA-remapped index, EGL needs physical GPU
render_gpu_device_id=gpu_ids[1]

# CORRECT — map CUDA index to physical GPU
visible = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
render_physical_gpu = visible[gpu_ids[1]]
```

## Model Loading Order

```python
# 1. Load PyTorch model FIRST (before LIBERO/TF)
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(..., device_map='auto', ...)

# 2. Block TF from GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# 3. Now import LIBERO
import gym; from libero.libero import ...
```
