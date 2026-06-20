# A800 Environment Build Report — M1A Official Core Env

**Date:** 2026-06-20
**Branch:** `infra/a800-migration-20260620`
**Gate:** M1A_OFFICIAL_CORE_ENV
**Status:** PASS (PARTIAL — flash-attn pending)

---

## Build Summary

| Field | Value |
|---|---|
| Prefix | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` |
| Python | 3.10.20 |
| PyTorch | 2.2.0+cu121 |
| CUDA (driver) | 550.90.12 |
| CUDA (torch) | 12.1 |
| transformers | 4.40.1 |
| tokenizers | 0.19.1 |
| timm | 0.9.10 |
| numpy | 1.26.4 |
| OpenVLA | `c8f03f48af692657d3060c19588038c7220e9af9` (detached HEAD) |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` (via pip libero 0.1.1) |
| flash-attn | NOT INSTALLED (M1B pending) |
| Root delta | 0 MiB (all writes to /mnt/sdc) |

## Root Safety

| Metric | Before | After | Delta |
|---|---|---|---|
| Root free | ~21 MiB | ~19.6 MiB | ~-1.4 MiB |
| /mnt/sdc free | ~701 GiB | ~389 GiB | ~-312 GiB |

Root remained within safe margins (no single step dropped root > 10 MiB). The /mnt/sdc drop includes our env (38 GiB) plus unrelated processes (mmunlearner cache, yangyenan cache, pi0 outputs).

## Validation Results

### M1A Import Smoke

| Import | Status |
|---|---|
| `torch` (2.2.0+cu121) | PASS |
| `torch.cuda.is_available()` | PASS (True) |
| `torch.cuda.is_bf16_supported()` | PASS (True) |
| `torch.cuda.get_device_name(0)` | PASS (NVIDIA A800-SXM4-80GB) |
| BF16 tensor on CUDA | PASS |
| `transformers` (4.40.1) | PASS |
| `tokenizers` (0.19.1) | PASS |
| `timm` (0.9.10) | PASS |
| `accelerate` (1.14.0) | PASS |
| `datasets` (5.0.0) | PASS |
| `huggingface_hub` | PASS |
| `libero` (0.1.1) | PASS |
| `mujoco` (3.9.0) | PASS |
| `AutoProcessor` | PASS |
| `AutoModelForVision2Seq` | PASS |

### Known Warnings (non-blocking)

1. **NumPy compatibility:** torch 2.2.0 compiled against NumPy 1.x, but numpy 1.26.4 is installed. A deprecation warning fires but functionality is preserved. opencv-python wants numpy>=2 but is functional with 1.26.4.

2. **TensorFlow oneDNN/TensorRT:** TF 2.15.0 (pulled by LIBERO/dlimp) emits informational messages about CPU optimizations and missing TensorRT. Inert for PyTorch inference.

3. **TRANSFORMERS_CACHE deprecation:** FutureWarning about using `TRANSFORMERS_CACHE`. Non-blocking.

## Network Notes

- `files.pythonhosted.org` intermittently unreachable from A800 (routing issues)
- Pitched to Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple/`) for reliable installs
- PyTorch installed from official `download.pytorch.org` CDN (reachable)

## Lock Files

| File | Entries | Path |
|---|---|---|
| pip freeze | 189 | `envs/openvla_official_a800_lock.txt` |
| conda explicit | 190 | (on A800 only — too large for GitHub) |

## M1B: flash-attn

Status: **DEFERRED**

Rationale:
- Root space still critical (< 20 MiB)
- flash-attn compilation requires significant temp space for CUDA compilation
- `eager` attention is sufficient for correctness baseline
- Will be enabled as a single-factor experiment after M1-M3 parity baseline

## Next Steps

1. [ ] Write `docs/A800_RUNTIME_PROFILES.md`
2. [ ] Write `scripts/infra/audit_python_environment.py`
3. [ ] Request **APPROVE_GATE_MIG1_MODEL_SYNC** for model download
4. [ ] M2 static parity (20-frame, Profile A vs Profile B)

## GPU Safety

No GPUs were occupied during this build phase. CUDA smoke tests used transient GPU 0 allocation (< 1 second).
