# A800 Model Sync Plan — 2026-06-20

**Status:** INVENTORY ONLY — NO DOWNLOADS
**Next Gate:** APPROVE_GATE_MIG1_MODEL_SYNC_SPATIAL_ONLY

---

## 1. Existing Assets on A800

**No OpenVLA models found anywhere on A800.**

Searched:
- `/mnt/sdc/dty_user/` — only `openvla-oft` config files (no weights)
- `/mnt/sdc/dty_user/cache/huggingface/hub/` — only `BAAI/Emu3.5`
- `/llm_jzm/` — no openvla models
- `/mnt/sdc/b1/` — no openvla models

All 4 suite checkpoints must be downloaded from Hugging Face Hub.

## 2. Model Inventory

### P0: LIBERO-Spatial (first priority)

| Field | Value |
|---|---|
| Hugging Face repo | `openvla/openvla-7b-finetuned-libero-spatial` |
| Requested revision | `main` (default) |
| Expected files | `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`, `model.safetensors` (or shards), `dataset_statistics.json` |
| Estimated size | ~14 GB |
| Destination | `$OPENVLA_MODEL_ROOT/libero-spatial/` |
| Existing cache | NONE |
| Processor/tokenizer | Included in repo (PrismaticAutoProcessor) |
| Action stats | In `dataset_statistics.json` |
| Priority | **P0 — first sync target** |

### P1: LIBERO-Object

| Field | Value |
|---|---|
| Hugging Face repo | `openvla/openvla-7b-finetuned-libero-object` |
| Estimated size | ~14 GB |
| Priority | P1 |

### P2: LIBERO-Goal

| Field | Value |
|---|---|
| Hugging Face repo | `openvla/openvla-7b-finetuned-libero-goal` |
| Estimated size | ~14 GB |
| Priority | P2 |

### P3: LIBERO-10

| Field | Value |
|---|---|
| Hugging Face repo | `openvla/openvla-7b-finetuned-libero-10` |
| Estimated size | ~14 GB |
| Priority | P3 |

## 3. Space Budget

### Per-checkpoint estimate: ~14 GB

| Scenario | Models | Download Size | /mnt/sdc Free After |
|---|---|---|---|
| Spatial only (P0) | 1 | ~14 GB | ~129 GB |
| Spatial + Object (P0+P1) | 2 | ~28 GB | ~115 GB |
| All 4 suites | 4 | ~56 GB | ~87 GB |

### Gate Requirements

| Gate | Required | Current | Status |
|---|---|---|---|
| Root free after sync | ≥ 20 GiB | 76 MiB | **FAIL** |
| /mnt/sdc free after sync | ≥ 200 GiB | ~129 GiB (even with 1 model) | **FAIL** |

**Model sync cannot proceed**: both ROOT and /mnt/sdc gates fail.

## 4. Unblocking Actions

### Required for spatial-only sync gate:
1. **Root ≥ 20 GiB**: Admin must clean other users' home dirs (ysc2: 177G, sz: 44G, huanzze: 27G)
2. **/mnt/sdc ≥ 214 GiB**: dty_user must clean pi0/ data (~377 GB lerobot_datasets)

## 5. Download Protocol (for future execution)

```bash
export HF_HOME=$OPENVLA_CACHE_ROOT/huggingface
export OPENVLA_MODEL_ROOT=/mnt/sdc/dty_user/openvla_attack/models

# Spatial first
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial \
  --local-dir $OPENVLA_MODEL_ROOT/libero-spatial \
  --local-dir-use-symlinks False
```

### Post-download verification
- File count match
- SHA256 of `model.safetensors` (or all shards)
- `dataset_statistics.json` unnorm key matches suite
- `preprocessor_config.json` processor class confirmed
- Action dim = 7
- Load test via `AutoModelForVision2Seq.from_pretrained(...)` (CPU-only; no GPU load until M2)
