# VIS Upgrade GPU0/7 Preflight

**Date**: 2026-06-01 | **Time**: 09:40

## GPU Status

| GPU | PCI | Memory | Temp | Status |
|-----|-----|--------|------|--------|
| 0 | 0000:04:00 | 1 MiB | 23°C | **Available** — historical Xid13/43 (5/29), currently idle |
| 7 | 0000:0F:00 | 1 MiB | 35°C | **Available** — Xid31 (5/31), currently idle |
| 4 | 0000:0C:00 | 1 MiB | 73°C | **QUARANTINED** — fresh Xid13 at 09:37 today |

## Fresh Xid

GPU4 (PCI 0000:0C:00.0) experienced Xid13 (SM Warp Exception / Illegal Instruction Encoding) at 09:37 during duration calibration rollout. Process pid=41821. GPU4 added to quarantine list alongside GPU0.

## Decision

GPU0 and GPU7 are both idle, healthy temperature, and free of fresh Xid. Proceed with short no-rollout VIS diagnostics on GPU0/7.

## Duration Calibration

Interrupted at 14/48 episodes. GPU4 output quarantined. Remaining 34 episodes to be restarted on GPU1,2,3,5,6 (GPU4 excluded) after VIS diagnostics.
