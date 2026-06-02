# DETCLEAN Duplicate Loop Stop Snapshot

**Timestamp**: 2026-05-29 20:09 CST
**Action**: Stopping duplicate detector-clean for-loop that is re-running salad_dressing (TID=4) unnecessarily

## Pre-Kill GPU State

```
GPU0: 2317 MiB (RoboTwin, quarantined)
GPU1: idle
GPU2: idle
GPU3: idle
GPU4: 8712 MiB, 41% util, 76°C (pid 41719, salad_dressing re-run)
GPU5: 8225 MiB, 34% util, 64°C (pid 41719)
GPU6: idle
GPU7: idle
```

## Pre-Kill Process State

- Parent bash: pid 26612 (ppid 26598, sshd), uptime since 19:44
  - Command: `for TID in 1 2 4 5 7; do ... --task_start $TID ...; done`
- Child python: pid 41719 (ppid 26612), uptime since 20:03
  - Command: `--task_start 4 --worker_id scan2_det_t4` (salad_dressing re-run)
  - ~150+ threads (OpenVLA inference)

## Process Tree

```
sshd → bash(26612) → python(41719)
```

## Pre-Kill Xid State

All Xid on GPU0 (PCI 0000:04:00) only, last at 14:03. No fresh Xid on GPU4,5.

## Completed Valid Detector-Clean States (pre-kill)

- cream_cheese s0: True (complete)
- cream_cheese s1: False (complete, clean-failed)
- cream_cheese s2: True (complete)
- salad_dressing s0: True (complete, last modified 19:58)
- salad_dressing s1: True (complete, last modified 20:01)
- salad_dressing s2: True (complete, last modified 20:03)

## Partial/Incomplete (pre-kill)

- ketchup s0: step_records only (94 lines), no manifest
- ketchup s1,s2: missing
- tomato_sauce s0,s1,s2: missing
- milk s0,s1,s2: missing

## Reason for Stop

The for-loop ran TID=1 (cream_cheese), then TID=2 (ketchup, crashed leaving partial s0), then TID=4 (salad_dressing, RE-RUNNING despite 3 complete states). Continuing risks overwriting valid salad_dressing data and will waste ~4h re-running completed work before reaching missing tasks (tomato_sauce, milk).
