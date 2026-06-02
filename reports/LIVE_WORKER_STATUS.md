# Live Worker Status — 2026-05-29 10:35 CST

## GPU Status

| GPU | Util% | Mem | Temp | Worker | Status |
|-----|-------|-----|------|--------|--------|
| 0 | 27% | 2061 MiB | 42C | QUARANTINED (Xid13) | Do not use |
| 1 | 37% | 8661 MiB | 53C | L10-B (tasks 5-9) | Running |
| 2 | 10% | 8680 MiB | 60C | Goal-100 v2 | Running |
| 3 | 42% | 8211 MiB | 67C | L10-B (tasks 5-9) | Running (Xid risk) |
| 4 | 22% | 8925 MiB | 76C | L10-A (tasks 0-4) | Running |
| 5 | 43% | 8205 MiB | 59C | L10-A (tasks 0-4) | Running |
| 6 | 28% | 8205 MiB | 67C | Goal-100 v2 | Running |
| 7 | 0% | 3 MiB | 34C | IDLE | Visual extraction queue |

## Active Processes

| PID | Worker | Suite | GPU(s) | Started | Progress |
|-----|--------|-------|--------|---------|----------|
| 6981 | goal100_v2_w0 | libero_goal | 2,6 | 09:32 | ~19/100 eps |
| 25324 | l10_v2_w45_task0_4 | libero_10 | 4,5 | 10:07 | ~9/100 eps |
| 27271 | l10_v2_w13_task5_9 | libero_10 | 1,3 | 10:07 | Loading (0 eps yet) |
| 12903 | replay_audit | CPU | — | 10:32 | Running (32 min) |

## Xid Status

| Date | GPU (PCI) | Error | Severity |
|------|-----------|-------|----------|
| May 27 21:17 | 0000:0f:00 (GPU3) | Xid 31 MMU Fault | Historical |
| May 28 16:18 | 0000:08:00 (GPU2) | Xid 31 MMU Fault | Historical |
| May 29 | — | None | Clean |

## Disk

| Mount | Size | Used | Avail | Use% | Inode% |
|-------|------|------|-------|------|--------|
| /data | 1.8T | 601G | 1.2T | 35% | 1% |

## Running Tasks Summary

- **Goal-100 v2**: 19/100 eps on GPU2,6 (~1h runtime, ETA ~4h)
- **L10-A (tasks 0-4)**: 9 eps on GPU4,5 (~25min runtime)
- **L10-B (tasks 5-9)**: 0 eps on GPU1,3 (~25min runtime, still loading)
- **Object-100 Replay Audit**: CPU task (PID 12903), 32 min, still running

## Alerts

- L10-B has 0 episodes after 25 min — may be loading or on a long task. Monitor.
- GPU3 has historical Xid 31 (May 27). If fresh Xid appears, quarantine shard.
- 7/8 GPUs busy. GPU7 available for visual extraction queue.
- No fresh Xid today. All workers alive.
