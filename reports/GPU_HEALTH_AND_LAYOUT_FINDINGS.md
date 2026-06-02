# GPU Health and Layout Findings

**Date**: 2026-06-02

## Summary

After extensive testing across ~50+ VIS PGD rollouts, we discovered that GPU0, GPU3, and GPU7
fail under PGD load when they are the PRIMARY GPU (cuda:0, handling both C+G rendering and compute).
However, they survive when used as PURE COMPUTE SECONDARY GPUs (cuda:1), with a healthy GPU
handling the C+G (rendering) load.

## Recovery Methods

1. **Server reboot**: Clears all Xid errors, temporary fix. ~2-3 minutes.
2. **Nvidia driver reload** (`rmmod nvidia* && modprobe nvidia*`): Faster (~10 seconds),
   achieves the same result, but kills ALL CUDA contexts on ALL GPUs. Only run this
   after verifying that no GPU job exists anywhere on the server.

## Stable GPU Layout

| Pair | Primary (C+G) | Secondary (Compute) | Status |
|------|---------------|---------------------|--------|
| g10 | GPU1 | GPU0 | Stable (GPU0 survives as secondary) |
| g23 | GPU2 | GPU3 | Stable (GPU3 survives as secondary) |
| g45 | GPU4 | GPU5 | Stable (always) |
| g67 | GPU6 | GPU7 | Stable (GPU7 survives as secondary) |

## Key Insight

The historically problematic GPUs (0: Xid13, 3: Xid31, 7: Xid13) only fail when they
must handle BOTH MuJoCo rendering (C+G) AND OpenVLA model computation. When offloaded
to compute-only (secondary in CUDA_VISIBLE_DEVICES), they survive full PGD workloads.

This allows all 8 GPUs to be used simultaneously across 4 parallel pairs.

## Xid History

| GPU | PCI | Xid Type | Count | Final Status |
|-----|-----|----------|-------|--------------|
| 0 | 04:00 | Xid13 (SM Warp) | 3+ | Usable as secondary |
| 3 | 08:00 | Xid31 (MMU Fault) | 10+ | Usable as secondary |
| 7 | 0f:00 | Xid13/31 | 5+ | Usable as secondary |
| 1,2,4,5,6 | — | None | 0 | Always healthy |
