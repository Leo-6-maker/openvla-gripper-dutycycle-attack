# Phase 8 — Autonomous 5-Hour Run

**Start**: 2026-06-27 17:30 CST
**End target**: 2026-06-27 22:30 CST
**Mode**: Autonomous (user away)

## Initial State (17:30)

| Metric | Value |
|--------|-------|
| Spatial pending | 210 |
| Goal pending | 152 |
| Done (queue) | 10 |
| Done (runs_v2 .done) | ~25 |
| Active bridges | 8 (2 per GPU) |
| GPUs active | 1, 2, 3, 6 |
| GPU memory | 24-40GB (under 72GB limit) |
| LIBERO-10 size | 6.8GB / 15GB |
| LIBERO-10 shard2 | 278MB transferred (SCP in progress) |
| LIBERO-10 shard3 | not started |

## Worker Map

| GPU | Worker 1 | Worker 2 |
|-----|----------|----------|
| 1 | W01 (Goal) | W05 (Goal) |
| 2 | W02 (Goal) | W06 (Goal) |
| 3 | W03 (Goal) | W07 (Goal) |
| 6 | W04 (Goal) | W08 (Goal) |

All workers picking Goal first. Spatial (210 pending) will auto-start as Goal drains.

## Pending Actions

1. Wait for SCP shard2 (~50 min), then shard3 (~50 min)
2. When LIBERO-10 complete: validate, P1, P2, P3, P4
3. Clear Spatial/Goal backlog (~30-40 min at current rate)
4. Freeze results at 22:30

## Stop Conditions
- GPU OOM → kill second worker on that GPU
- Duplicate claims → pause, investigate
- Technical failure rate > 2% → pause
- ArmLock violation → record but continue non-ArmLock tasks
