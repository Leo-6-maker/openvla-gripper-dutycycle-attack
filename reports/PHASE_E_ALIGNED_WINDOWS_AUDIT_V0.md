# Phase E Aligned Windows Audit V0

**Status**: BLOCKED_MISSING_ALIGNED_WINDOWS
**Input**: `tables/missing_phaseE_aligned_windows_v0_server.csv`
**Total rows**: 0
**Recommended rows**: 0
**Recommended positives**: 0
**Recommended negatives**: 0
**Missing qpos rows**: 0
**Obs-only rows**: 0
**Natural-open rows**: 0
**Canary ready**: false
**Reason**: not enough safe recommended positive/negative rows or hard-fail audit

This audit must pass before any Phase E GPU canary. It is CPU-only.

## Blocked Reason Distribution

- None.

## Checks

- `input_exists`: fail (1) aligned windows CSV missing: tables/missing_phaseE_aligned_windows_v0_server.csv