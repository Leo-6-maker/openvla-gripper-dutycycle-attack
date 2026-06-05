# Batch4 Candidate Schema Audit

**Status**: FAIL
**Input**: `tables/object_phase_response_batch4_candidates.csv`
**Rows**: 0

This is a CPU-only schema/safety audit. It does not run rollout, VIS, GPU work, watcher jobs, or detector training.

## Blocking Issues

- `candidate_rows_present`: candidate CSV has no rows
- `enough_hard_negatives`: hard negatives=0, min=6

## Warnings

- None.

## Expected Role Counts

- None.

## Safety Checks

- Candidate rows must not assume GPU3 or GPU7, or disabled pairs 2,3 / 6,7.
- `denominator_plan`, `expected_role`, and qpos verification fields are required.
- Phase D/E proxy labels are not valid Batch4 gold candidates.
- Duplicated task/state/window candidates hard-fail.
