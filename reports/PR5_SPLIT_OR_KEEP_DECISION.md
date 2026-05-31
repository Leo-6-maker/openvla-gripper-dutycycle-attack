# PR #5 Split or Keep Decision

**Date**: 2026-06-01

## Assessment

PR #5: 73 files, 6,059 additions. Structure:

| Category | Files | Content |
|----------|-------|---------|
| Core code | 2 | dtype fix + re-decode helper |
| VIS diagnostics | 4 scripts + 4 tests | token flip, arm drift, one-frame loader, contact-frame planner |
| CrossSuite prep | 2 scripts | feature audit, dataset index builder |
| Tests | 8 | pytest for all diagnostics |
| Reports | 30+ | gate status, handoffs, freeze reports |
| Tables | 15+ | diagnostic results, plans, audit CSVs |

## Decision: KEEP as single PR

Reasoning:
1. All changes are diagnostics/prep only — no production semantics touched
2. Core code changes are minimal (2 files, 3 lines)
3. Files are already organized by category
4. Splitting would create merge conflicts and overhead
5. PR is draft — can be reviewed as-is with clear summary

## Recommendation

- Keep as single diagnostic PR
- Update PR body to clearly state: production unchanged, VIS blocked, CrossSuite prep only
- Mark ready for review when body updated
