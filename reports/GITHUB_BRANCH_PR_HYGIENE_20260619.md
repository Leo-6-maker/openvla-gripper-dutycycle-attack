# GitHub Branch and PR Hygiene Audit

This is non-destructive. It does not close PRs or delete remote branches.

## Summary

```json
{
  "CLOSE_OR_ARCHIVE_PR_BEFORE_DELETE": 6,
  "DELETE_CANDIDATE_AFTER_WORKTREE_CHECK": 2,
  "KEEP": 4,
  "KEEP_OR_ARCHIVE_REVIEW_REQUIRED": 45
}
```

## First Low-Risk Delete Candidates

- `eval/official-libero-clean-20260525`
- `merge/sc5-mainline-20260618`

Connector spot-check note: the earlier suggested `denominator/full4-clean-20260519`,
`fix/protocol-schema-and-condition-config-20260523`, and
`fix/table1-generic-autowindow-baseline-20260524` did not appear in the current
GitHub branch search. They may already be gone from `origin` or only exist in
older local/server remotes; do not include them in a GitHub deletion command
unless a fresh remote ref check shows they exist.

## Old Draft / Historical PR Branches To Archive Then Close

- `audit/m3-v2-seed81-trajectory-feasibility-20260615`
- `exp/codex-autonomous-vis-crosssuite-20260531`
- `exp/m3-arm-constrained-logratio-v3-20260615`
- `exp/m3-arm-v3-fresh-seed82-canary-20260615`
- `exp/m3-arm-v4-fixed-frame-panel-prereg-20260615`
- `exp/m3-arm-v4-hard-feasible-selection-20260615`

## Guardrails

- Check server worktrees before deleting any branch.
- Use archive tags for historical M3 evidence before deleting old experiment heads.
- Keep `feature/sc5-cross-suite-generalization-20260619` until CLEAN300 is fully audited.
- Keep `feature/sc5-video-export-20260618` until video evidence is accepted.
