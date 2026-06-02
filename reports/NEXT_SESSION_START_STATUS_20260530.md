# Session Start Status — 2026-05-30

## Server

- **Repo**: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524`
- **Branch**: `exp/sustained-proxy-burst-control-20260530` (production)
- **Commit**: `07e13a0`
- **Working tree**: clean (untracked config/artifact files only)

## GPU

| GPU | Memory | Utilization | Temp | Status |
|-----|--------|-------------|------|--------|
| 0 | 2060/11264 MiB | 0% | 41C | QUARANTINED (lgzhou RoboTwin, Xid13 history) |
| 1 | 3/11264 MiB | 0% | 23C | IDLE |
| 2 | 3/11264 MiB | 0% | 22C | IDLE |
| 3 | 3/11264 MiB | 0% | 30C | IDLE |
| 4 | 4/11264 MiB | 0% | 24C | IDLE |
| 5 | 3/11264 MiB | 0% | 21C | IDLE |
| 6 | 3/11264 MiB | 0% | 24C | IDLE |
| 7 | 3/11264 MiB | 0% | 29C | IDLE (NO OpenVLA — OOM risk) |

## dmesg

Last Xid: 2026-05-29 14:03 (Xid13 on GPU0 + Xid43), ~32h stale. No fresh Xid events.
GPU0 has lgzhou RoboTwin process (pid 24513, 2055 MiB).

## Disk

1.8T total, 611G used (36%), 1.2T available.

## Screen Sessions

None.

## Active Jobs

None. All GPUs idle.

## Production Detector

ProprioNoStep (`/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt`)

## Production Attack

sustained_command_open_proxy_30 (burst_steps=30, hold_mode=fixed)

## Visual Status

- VisualNoStep V6 pilot results frozen (non-production)
- VisualNoStep triggers but is non-selective at th=0.05
- ProprioNoStep remains the only production online detector
