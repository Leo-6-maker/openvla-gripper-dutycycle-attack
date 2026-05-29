# GPU Preflight — Attack Pilot Readiness

**Date**: 2026-05-29 12:00 CST

## Current GPU Status

| GPU | Util% | Mem | Temp | Worker | Status |
|-----|-------|-----|------|--------|--------|
| 0 | 0% | 2318M | 43C | — | ⛔ QUARANTINED (Xid13, CUDA illegal mem) |
| 1 | 20% | 8838M | 57C | L10-B (tasks 5-9) | Running |
| 2 | 43% | 8700M | 66C | **Goal-100** | Running (63/100) |
| 3 | 30% | 8212M | 74C | L10-B (tasks 5-9) | ⚠️ Xid31 May 29 11:42 |
| 4 | 32% | 8516M | 76C | L10-A (tasks 0-4) | Running |
| 5 | 44% | 8226M | 66C | L10-A (tasks 0-4) | Running |
| 6 | 45% | 8206M | 73C | **Goal-100** | Running (63/100) |
| 7 | 0% | 4M | 36C | — | ⛔ OOM for OpenVLA 7B inf |

## Xid Log

| Date | GPU (PCI) | Error | Severity |
|------|-----------|-------|----------|
| May 27 21:17 | 0000:0f:00 (GPU3) | Xid 31 MMU Fault | Historical |
| May 28 16:18 | 0000:08:00 (GPU2) | Xid 31 MMU Fault | Historical |
| **May 29 11:42** | **0000:0f:00 (GPU3)** | **Xid 31 MMU Fault** | **⚠️ TODAY** |

## Target GPU Pair: GPU2,6

| GPU | Xid History | Current Load | ETA Free |
|-----|-------------|-------------|----------|
| 2 | May 28 (historical) | 43% (Goal-100) | ~3h |
| 6 | None | 45% (Goal-100) | ~3h |

## Pre-launch Checklist

- [ ] GPU2,6 Goal-100 worker exited cleanly
- [ ] `nvidia-smi` shows 0% util, low memory on GPU2,6
- [ ] `dmesg | grep Xid` shows no new Xid on GPU2 or GPU6 since Goal-100 started
- [ ] `CUDA_VISIBLE_DEVICES=2,6` test load succeeds
- [ ] Output root directory created and empty
- [ ] Detector checkpoint verified (sha256: `4b3f3d47...`)

## Exclusion Log

| GPU | Reason | Date |
|-----|--------|------|
| GPU0 | Xid13, CUDA illegal memory access | Confirmed May 29 |
| GPU7 | OOM at inference (10.66/10.75 GB) | Confirmed May 29 |
| GPU3 | Xid31 May 29 11:42 — L10-B shards after 11:42 to be quarantined | May 29 |
