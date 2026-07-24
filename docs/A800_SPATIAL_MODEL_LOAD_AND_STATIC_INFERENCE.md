# A800 Spatial Model — Transfer, Load & Static Inference Report

**Date:** 2026-06-21 (00:10–01:06 CST)
**Branch:** `infra/a800-migration-20260620`
**Gate:** MIG1_SPATIAL_LOCAL_TRANSFER_AND_STATIC_INFERENCE
**Status:** PASS

---

## 1. Source Model

| Field | Value |
|---|---|
| Host | klfy-SYS-4028GR-TR2 (old 2080Ti server) |
| Path | `/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial` |
| Repo | `openvla/openvla-7b-finetuned-libero-spatial` |
| Classification | LOCAL_MODIFIED_SNAPSHOT (has .bak_for_tfm5 files, .pyc, incomplete HF downloads) |
| Total size | 16 GB (source), 15 GB (transferred, excluding cache/pyc/bak) |
| Excluded | `__pycache__/`, `.cache/`, `*.bak*`, `*.incomplete`, `*.lock` |

## 2. Transfer

| Field | Value |
|---|---|
| Method | Server-to-server rsync (old-server → A800 direct) |
| Setup | New ed25519 key on old server → A800 authorized_keys |
| Destination | `/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620/` |
| Files transferred | 19 |
| Bytes transferred | 13.86 GB |
| Duration | ~25 min |
| SHA mismatches | **0** |

## 3. Processor & Tokenizer

| Field | Value |
|---|---|
| Processor class | `PrismaticProcessor` |
| Tokenizer class | `LlamaTokenizerFast` |
| EOS token ID | **2** (`</s>`) |
| PAD token ID | 32000 |
| BOS token ID | 1 |
| Vocab size | 32,000 |
| Prompt format | `In: What action should the robot take to {task.lower()}?\nOut:` |
| EOS appended by processor | No — must be appended explicitly |

### Prompt Tokenization (example)
```
Task: "pick up the black bowl next to the ramekin and place it on the plate"
Tokens: [1, 512, 29901, 1724, ..., 29973, 13, 3744, 29901]
Count: 32 (Llama tokenizer)
EOS appended: False (2 not at end)
```

## 4. Action Stats

| Field | Value |
|---|---|
| Unnorm key | `libero_spatial` |
| Action dim | 7 (6 EEF + 1 gripper) |
| q01 | `[-0.7455, -0.6616, -0.9375, -0.1071, -0.2068, -0.1843, 0.0]` |
| q99 | `[0.9375, 0.8759, 0.9321, 0.1039, 0.1768, 0.1457, 1.0]` |
| SHA256 | `498d59072d79e32715b5fb1b817a025ca8034a4acec42fd62686392251ed182a` |

## 5. Model Load (Profile B)

| Field | Value |
|---|---|
| GPU | Index 6, UUID `GPU-92963392-f77a-85ce-4ba7-7a8288429ca5` |
| Visible GPU | `CUDA_VISIBLE_DEVICES=6` → `cuda:0` |
| Model class | `OpenVLAForActionPrediction` |
| Params | 7,541,237,184 (7.5B) |
| Dtype | `torch.bfloat16` ✅ |
| Attention | `eager` ✅ |
| Device map | `{cpu:0}` (single GPU, no sharding) |
| Load time | 13.1s (4 shards) |
| Peak VRAM (allocated) | 14.09 GiB |
| Peak VRAM (reserved) | 14.13 GiB |

## 6. Static Inference (3-run Determinism)

| Run | Pixel SHA | Token SHA | Generated Tokens | Time |
|---|---|---|---|---|
| 1 | `b396d96b...` | `eff22144...` | `[31887,31863,31872,31870,31862,31857,31857]` | 1.208s |
| 2 | `b396d96b...` | `eff22144...` | `[31887,31863,31872,31870,31862,31857,31857]` | 0.227s |
| 3 | `b396d96b...` | `eff22144...` | `[31887,31863,31872,31870,31862,31857,31857]` | 0.243s |

**DETERMINISM: PASS (3/3 identical)**

### Action Decode

| Field | Value |
|---|---|
| Raw action | `[0.0960, 0.1071, -0.0027, -0.0016, -0.0150, -0.0193, 0.0]` |
| Unnormed action | `[-0.5839, -0.4969, -0.9425, -0.1075, -0.2125, -0.1907, 0.0]` |
| Gripper raw | 0.0 |
| Gripper normalized | -1.0 (CLOSE) |
| Gripper env | 1.0 (inverted) |
| Classification | **CLOSE** |

## 7. Disk Status

| Metric | Before Transfer | After Transfer | After Model Load |
|---|---|---|---|
| Root | ~154 MiB | ~154 MiB | ~154 MiB |
| /mnt/sdc free | ~138 GiB | ~101 GiB | 101 GiB |

## 8. Dependency Limitations

| Issue | Status | Inference Impact |
|---|---|---|
| opencv vs numpy 1.26 | pip check FAIL | None (PIL used) |
| protobuf prismatic import | FAIL | None (inference path OK) |
| flash-attn | NOT INSTALLED | None (eager baseline) |

---

**SPATIAL_MODEL_TRANSFER = PASS**
**SPATIAL_MODEL_SINGLE_A800_LOAD = PASS**
**STATIC_INFERENCE_DETERMINISM = PASS**
