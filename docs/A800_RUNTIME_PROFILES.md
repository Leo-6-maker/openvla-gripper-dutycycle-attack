# A800 Runtime Profiles — OpenVLA Gripper Duty-Cycle Attack

**Version:** v1.0-draft
**Date:** 2026-06-20
**Status:** Definitions only (no runtime validation yet)

---

## Profile A: legacy_2080ti_replay

**Purpose:** Reproduce old 2080Ti results on A800 hardware, isolating pure GPU differences.

**Status:** DEFINED — not yet validated

| Parameter | Value |
|---|---|
| Python | 3.10 (match legacy env) |
| PyTorch | 2.2.0 (match legacy) |
| Model | OpenVLA 7B fine-tuned (4 suites) |
| Attention | `eager` |
| Device map | `auto` |
| Per-GPU memory cap | 10,000 MiB |
| Dtype | FP32 (match legacy) |
| Preprocessing | Legacy PIL path (2048×2048→224×224, bicubic) |
| EOS | Not explicitly appended (legacy behavior) |
| Action decode | Legacy bin-center with q01/q99 |
| Unnorm key | Programmatic from action stats |
| Gripper | Legacy normalize → binarize |

### Constraints
- Must reproduce the 2080Ti action token distribution exactly (allowing for minor FP differences)
- Used ONLY for historical comparison — not for new experiments

## Profile B: openvla_official_a800

**Purpose:** Clean OpenVLA reference following upstream README exactly.

**Status:** DEFINED — not yet validated

| Parameter | Value |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.2.0+cu121 |
| Transformers | 4.40.1 |
| Tokenizers | 0.19.1 |
| TIMM | 0.9.10 |
| Attention (baseline) | `eager` |
| Attention (optimized) | `flash_attention_2` (separate experiment, M1B) |
| Dtype | BF16 |
| Device | Single A800 GPU (no sharding) |
| Memory cap | None (use full 80 GiB) |
| Image preprocessing | Official: 180° rotate, RGB, PIL LANCZOS, center crop, no JPEG round-trip |
| Image size | 224×224 |
| EOS | Explicitly appended (token 29871) |
| Prompt | `"What action should the robot take to {task.lower()}?"` + EOS |
| Action decode | Official: bin-center from action stats, q01/q99 mask |
| Unnorm key | Matched to suite from model metadata |
| Gripper | `[0,1] → [-1,1]` normalize → sign binarize → sign inversion (project convention) |

### Upstream References
- OpenVLA: `c8f03f48af692657d3060c19588038c7220e9af9`
- LIBERO: `8f1084e3132a39270c3a13ebe37270a43ece2a01`

### Suites
| Suite | Checkpoint | Unnorm Key |
|---|---|---|
| LIBERO-Object | `openvla/openvla-7b-finetuned-libero-object` | `libero_object` |
| LIBERO-Spatial | `openvla/openvla-7b-finetuned-libero-spatial` | `libero_spatial` |
| LIBERO-Goal | `openvla/openvla-7b-finetuned-libero-goal` | `libero_goal` |
| LIBERO-10 | `openvla/openvla-7b-finetuned-libero-10` | `libero_10` |

Checkpoint and unnorm key must be confirmed from model metadata at load time, not hardcoded.

## Profile C: project_a800

**Purpose:** Profile B + project telemetry + detector hook. Must preserve CLEAN action path.

**Status:** NOT YET DEFINED — blocked by M2/M3

| Contract | Detail |
|---|---|
| CLEAN action path | Must be byte-identical to Profile B |
| Allowed additions | Telemetry, SC5 streaming, artifact provenance, detector runtime, attack adapter interface |
| Forbidden changes | Prompt, preprocessing, EOS, generation, action decode, unnorm stats, gripper semantics, environment |

---

## Parity Sequence (Phase B)

Single-factor changes on A800, applied sequentially:

| Step | Change | Profile |
|---|---|---|
| P0 | Legacy 2080Ti replay (multi-GPU, eager, 10G cap, legacy preproc) | A |
| P1 | P0 + official PIL preprocessing | — |
| P2 | P1 + explicit EOS 29871 | — |
| P3 | P2 + single-GPU BF16 (no memory cap) | — |
| P4 | P3 + official step semantics | — |
| P5 | P4 + flash_attention_2 | B (optimized) |

Each step reuses the same 20 frozen frames (5 per suite). Output a diff matrix per step.

---

## GPU Lease Protocol

1. Before ANY GPU use: `nvidia-smi` + `nvidia-smi pmon -c 1`
2. Target: GPU 4 or 6 (historically idle)
3. Create lease: `/mnt/sdc/dty_user/openvla_attack/gpu_locks/gpu{N}.lock`
4. Lease fields: hostname, GPU index, UUID, owner, PID, start time, expected end time, git commit, runtime profile, output root, command SHA
5. After completion: remove lease file

**Never assume GPU is free based on prior snapshots.**
