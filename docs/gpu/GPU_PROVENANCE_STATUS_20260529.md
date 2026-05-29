# GPU Provenance Status — 2026-05-29

## GPU Inventory

| GPU | PCI | Model | Status | Xid History | Current Use |
|-----|-----|-------|--------|-------------|-------------|
| 0 | 0000:04:00.0 | RTX 2080 Ti 11GB | ⛔ QUARANTINED | Xid13 (permanent), CUDA illegal mem access | lgzhou RoboTwin (2.3GB occupied) |
| 1 | 0000:06:00.0 | RTX 2080 Ti 11GB | ✅ Running | Clean | L10-B (tasks 5-9) |
| 2 | 0000:07:00.0 | RTX 2080 Ti 11GB | ✅ Running | Xid31 May 28 (historical) | Goal-100 v2 |
| 3 | 0000:08:00.0 | RTX 2080 Ti 11GB | ⚠️ Flagged | Xid31 May 28 (historical) | L10-B (tasks 5-9) |
| 4 | 0000:0C:00.0 | RTX 2080 Ti 11GB | ✅ Running | Clean | L10-A (tasks 0-4) |
| 5 | 0000:0D:00.0 | RTX 2080 Ti 11GB | ✅ Running | Clean | L10-A (tasks 0-4) |
| 6 | 0000:0E:00.0 | RTX 2080 Ti 11GB | ✅ Running | Clean | Goal-100 v2 |
| 7 | 0000:0F:00.0 | RTX 2080 Ti 11GB | ⛔ QUARANTINED for attack | Xid31 May 27 + May 29 | Idle (visual extraction only) |

## Xid Log (2026-05-27 to 2026-05-29)

| Date | GPU | PCI | PID | Xid | Type | Cause |
|------|-----|-----|-----|-----|------|-------|
| May 27 21:17 | 7 | 0F:00.0 | 44007 | 31 | MMU FAULT VIRT_READ | Unknown python process |
| May 28 16:18 | 3 | 08:00.0 | 26192 | 31 | MMU FAULT VIRT_READ | Unknown python process |
| May 29 11:42 | 7 | 0F:00.0 | 9050 | 31 | MMU FAULT VIRT_WRITE | GPU7 OOM attack test scripts |

## GPU0,7 Engineering Rehearsal Results

| Episode | Condition | Success | CUDA Error? |
|---------|-----------|---------|-------------|
| cream_cheese s0 | clean | False (291 steps) | No |
| milk s0 | clean | False (71 steps) | Yes: CUBLAS_STATUS_EXECUTION_FAILED |
| cream_cheese s0 | oracle_open | True (167 steps) | No |
| milk s0 | oracle_open | True (157 steps) | No |
| cream_cheese s0 | random_control | False (130 steps) | Yes: illegal memory access |
| milk s0 | random_control | True (151 steps) | No |

**Error rate: 2/6 = 33%**

## Pair Eligibility for Attack Pilot

| GPU Pair | Free? | Xid Risk | Memory | Verdict |
|----------|-------|----------|--------|---------|
| GPU0,7 | Now | HIGH (33% error rate) | 19.2 GB total | ❌ UNRELIABLE |
| GPU2,6 | ~30 min | LOW (GPU2 historical only) | 22 GB total | ✅ PREFERRED |
| GPU1,3 | Occupied | MEDIUM (GPU3 historical) | — | ⚠️ Occupied by L10 |
| GPU4,5 | Occupied | LOW | — | ⚠️ Occupied by L10 |

## Decision

- **GPU0,7**: Engineering smoke ONLY. Never for official evidence.
- **GPU2,6**: Primary target for formal 40-rollout Object attack pilot.
- **GPU1,3,4,5**: Reserved for L10 workers. Do not interrupt.
- **GPU3 L10 shards**: Quarantine shards produced around Xid31 timestamp (May 28) during aggregation.
