# VIS-1R Full-Window Screening Plan

**Candidates input**: `tables/object_phase_response_batch4_candidates.csv`
**Commands prepared**: 0
**pgd_restarts**: 1
**pgd_steps**: 40

This is a CPU-only command plan. It did not run GPU, VIS, rollout, watcher, or detector training.

## Label Boundary

- 1R positives may be treated as `silver_positive_1r` only after audit.
- 1R negatives are `pending_negative_1r`, never gold negatives.
- Gold labels require full VIS 3R confirmation.