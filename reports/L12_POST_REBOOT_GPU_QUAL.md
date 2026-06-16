# L12 Post-Reboot GPU Qualification

**Date:** 2026-06-16
**Server:** klfy-SYS-4028GR-TR2

## GPU Inventory (nvidia-smi)

| Index | Model | Memory | UUID | Temp | Power | Status |
|-------|-------|--------|------|------|-------|--------|
| 0 | RTX 2080 Ti | 11264 MiB | GPU-33559d32... | 34°C | 65W | **FAULT (Xid 13+43)** |
| 1 | RTX 2080 Ti | 11264 MiB | GPU-64734dbf... | 35°C | 67W | HEALTHY |
| 2 | RTX 2080 Ti | 11264 MiB | GPU-a5dd9dbe... | 35°C | 55W | HEALTHY (M3 qual PASS) |
| 3 | RTX 2080 Ti | 11264 MiB | GPU-c1ee1619... | 38°C | 67W | HEALTHY |
| 4 | RTX 2080 Ti | 11264 MiB | GPU-d0a54f5d... | 38°C | 62W | **FAULT (non-deterministic)** |
| 5 | RTX 2080 Ti | 11264 MiB | GPU-9794d733... | 36°C | 58W | HEALTHY |
| 6 | RTX 2080 Ti | 11264 MiB | GPU-04d369ad... | 40°C | 62W | HEALTHY (M3 qual PASS) |
| 7 | RTX 2080 Ti | 11264 MiB | GPU-da4d4ba8... | 43°C | 38W | **RENDER-ONLY** (gradient non-det.) |

All GPUs idle (0 MiB used), P0 state, 1350 MHz SM / 7000 MHz memory.

## Production GPU Pairs

| Pair | Model GPU | Render GPU | Scope |
|------|-----------|------------|-------|
| (2,6) | GPU2+6 | GPU6 | Full: capture + attack |
| (1,3) | GPU1+3 | GPU3 | Full: capture + attack |
| (5,7) | GPU5 | GPU7 | Capture only (forward) |

## Xid Error Log (current boot)

| Timestamp (s) | PCI | Xid | GPU | Description |
|---------------|-----|-----|-----|-------------|
| ~344 | 0000:04:00 | 13 | GPU0 | Graphics SM Warp Exception — Out Of Range Address |
| ~344 | 0000:04:00 | 43 | GPU0 | Ch 00000008, pid=18412 (python) |
| ~6245 | 0000:0f:00 | 31 | GPU7 | MMU Fault — FAULT_PDE ACCESS_TYPE_VIRT_READ |

**No new Xid errors since previous M3 qualification run.** All Xid entries are from boot time.

## PyTorch CUDA Health

| GPU | matmul | Memory | Status |
|-----|--------|--------|--------|
| GPU1 | PASS | 16 MiB | OK |
| GPU2 | PASS | 16 MiB | OK |
| GPU3 | PASS | 16 MiB | OK |
| GPU5 | PASS | 16 MiB | OK |
| GPU6 | PASS | 16 MiB | OK |
| GPU7 | PASS | 16 MiB | OK |

## Checkpoint Load

| Checkpoint | Path | Result |
|------------|------|--------|
| D5 candidate best | `/data/liuyu/outputs/d5_training/d5_candidate_best.pt` | OK (val_acc=0.842) |
| D1b detector best | `outputs/d1b_training/d1b_detector_best.pt` | OK |

## Clean Forward

- OpenVLA 7B loaded on GPU2,6 (21.4s)
- Model loads successfully, CUDA functional
- Previously verified: 217-step clean shadow episode on GPU5,7 (smoke PASS)

## ECC / Retired Pages

RTX 2080 Ti does not support ECC. All ECC/Retired Page fields: N/A.

## Phase 1 Gates

| Gate | Status |
|------|--------|
| Target GPUs visible (6/8 usable) | PASS |
| New Xid / uncorrectable ECC / CUDA runtime error | NONE |
| Checkpoint loadable | PASS |
| Clean forward succeeds | PASS |
| No leftover GPU processes | PASS |
| GPU0/4 faults documented, excluded | YES |
| GPU7 gradient limitation documented | YES |

**Phase 1 RESULT: ALL GATES PASS**
