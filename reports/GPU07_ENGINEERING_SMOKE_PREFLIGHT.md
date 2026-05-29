# GPU0,7 Engineering Smoke Preflight

**Date**: 2026-05-29 13:15 CST
**Purpose**: Engineering-only smoke test for patched detector-triggered runner

## GPU Status

| GPU | PCI | Free Mem | Occupied Mem | Process | Power | Temp |
|-----|-----|----------|-------------|---------|-------|------|
| 0 | 0000:04:00.0 | 8693 MiB | 2318 MiB | PID 23288 (lgzhou RoboTwin) | 64W P2 | 43C |
| 7 | 0000:0F:00.0 | 11007 MiB | 4 MiB | — | 22W P8 | 36C |

**Total free**: ~19.2 GiB across both GPUs.

## Xid History (corrected PCI mapping)

| Date | GPU | PCI | Type | Severity |
|------|-----|-----|------|----------|
| May 27 21:17 | **7** | 0F:00.0 | Xid31 MMU READ | Historical |
| May 28 16:18 | 3 | 08:00.0 | Xid31 MMU READ | Historical (L10-B still running on it) |
| **May 29 11:42** | **7** | **0F:00.0** | **Xid31 MMU WRITE** | **TODAY — from our OOM attack tests** |

## Risk Assessment

| Risk Factor | GPU0 | GPU7 |
|-------------|------|------|
| Xid history | Xid13 (permanent) | Xid31 ×2 (May 27, May 29) |
| Fresh Xid today | None | Yes (11:42, from OOM scripts) |
| Occupied by others | lgzhou RoboTwin (2.3G) | None |
| Usable for smoke | ⚠️ Risky but testable | ⚠️ Risky (today's Xid) |

## Hard Stops Active
- [ ] GPU0 lgzhou process — NOT killing without approval
- [ ] Fresh Xid during smoke → stop immediately
- [ ] CUDA illegal memory access → stop immediately
- [x] No fresh Xid on GPU7 since 11:42 (2h clear)

## Verdict
GPU0,7 pair has 19.2 GiB free — enough for OpenVLA 7B (~14 GB bfloat16). Both have Xid history but are currently idle. Proceeding to Stage 1 with caution. This is ENGINEERING ONLY — not for final evidence.
