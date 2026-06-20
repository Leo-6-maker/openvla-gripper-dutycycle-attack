# A800 Migration Protocol — OpenVLA Gripper Duty-Cycle Attack

**Version:** v1.0-draft
**Date:** 2026-06-20
**Branch:** `infra/a800-migration-20260620`
**Owner:** DeepSeek (migration lead)

---

## 1. Objective

Establish an auditable, reproducible, official-aligned OpenVLA execution chain on the A800 server (pm-364c0001), prove hardware/protocol parity with the legacy 2080Ti server, and migrate frozen data assets with SHA-verified integrity.

## 2. Upstream Reference Versions

### OpenVLA

| Field | Value |
|---|---|
| Repo | `https://github.com/openvla/openvla` |
| HEAD commit | `c8f03f48af692657d3060c19588038c7220e9af9` |
| Default branch | `main` |
| Total commits | 63 |
| Python | 3.10 |
| PyTorch | 2.2.0 |
| Transformers | 4.40.1 |
| CUDA | 12.4 |
| Flash-Attention | 2.5.5 |

### LIBERO

| Field | Value |
|---|---|
| Repo | `https://github.com/Lifelong-Robot-Learning/LIBERO` |
| HEAD commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Old server checkout | `8f1084e3132a39270c3a13ebe37270a43ece2a01` (matches HEAD) |

## 3. Three Runtime Profiles

### Profile A: legacy_2080ti_replay

Purpose: Reproduce old 2080Ti results on A800, isolating hardware effects.

| Parameter | Value |
|---|---|
| Python | Match `openvla_official_libero_20260525` env |
| Attention | `eager` |
| Device map | `auto` with per-GPU 10 GiB cap |
| Model load | Multi-GPU sharded |
| Preprocessing | Legacy PIL path (from old server) |
| EOS handling | Legacy (no explicit append) |
| Checkpoint | Matched to old server suite checkpoints |
| Unnorm key | Programmatic from action stats |

### Profile B: openvla_official_a800

Purpose: Clean OpenVLA reference following upstream README exactly.

| Parameter | Value |
|---|---|
| Python | 3.10 |
| Attention | `eager` (baseline), then `flash_attention_2` (separate experiment) |
| Device map | Single A800, BF16 |
| Model load | No artificial memory cap |
| Preprocessing | Official PIL/LANCZOS/center crop |
| EOS handling | Explicit EOS 29871 appended |
| Image rotation | 180° (agentview convention) |
| JPEG round-trip | None |
| Checkpoint | Suite-specific from official HuggingFace |
| Unnorm key | Matched to suite from model metadata |

### Profile C: project_a800

Purpose: Project telemetry + detector hook on top of Profile B, without changing the CLEAN action path.

Core contract:
```
Project CLEAN action path == Official reference CLEAN action path
```

## 4. Parity Plan

### Phase A: Hardware-only parity

- Fixed: code, env, preprocessing, EOS, checkpoint, unnorm, attention, device_map, frame, seed
- Variable: 2080Ti → A800
- Frames: 20 (5 per suite)
- Gate: input SHA 100%, prompt IDs 100%, preprocess tensor 100%, action stats 100%, gripper semantics 100%

### Phase B: Official-alignment parity

Sequential single-variable changes:
- P0: legacy_2080ti_replay (full profile)
- P1: + official PIL preprocess
- P2: + explicit EOS
- P3: + single-GPU load (no memory cap)
- P4: + official step semantics
- P5: + flash_attention_2

Each step: same 20 frames, output diff matrix.

## 5. Environment Build (Post-M0)

```bash
conda create -p /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800 python=3.10 -y
conda activate /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers==4.40.1
pip install "flash-attn==2.5.5" --no-build-isolation
# LIBERO and MuJoCo
pip install libero
# OpenVLA from source
git clone https://github.com/openvla/openvla.git /mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream
cd /mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream
git checkout c8f03f48af692657d3060c19588038c7220e9af9
pip install -e .
```

## 6. Gate Sequence

```
M0: Host Safety          → BLOCKED (root full)
M1: Environment Ready    → BLOCKED (depends on M0)
M2: Static Parity        → BLOCKED (depends on M1)
M3: Closed-Loop Parity   → BLOCKED (depends on M2)
M4: Transfer Acceptance  → BLOCKED (depends on M3)
M5: Primary Cutover      → BLOCKED (depends on M4)
```

## 7. Files NOT to Commit to GitHub

- Private IPs, ports, usernames
- SSH key paths or key contents
- Passwords or tokens
- Absolute paths containing `/mnt/sdc/dty_user/` etc. (use `$OPENVLA_MIGRATION_ROOT`)
- Model weights
- Rollout videos
- NPZ/dataset files > 1 MB

## 8. Related Documents

- [A800_HOST_AUDIT_20260620.md](gpu/A800_HOST_AUDIT_20260620.md)
- [OLD_SERVER_ASSET_CENSUS_20260620.md](gpu/OLD_SERVER_ASSET_CENSUS_20260620.md)
- [MIGRATION_GAP_REGISTER_20260620.md](gpu/MIGRATION_GAP_REGISTER_20260620.md)
- [SERVER_MIGRATION_REPORT_PM_364C0001.md](gpu/SERVER_MIGRATION_REPORT_PM_364C0001.md) (pre-handoff survey)
