# Branch Policy

## Long-Lived Branches

- `main` is the only long-lived code branch.
- Active `feature/*` branches should have an open PR and should be merged, closed, or explicitly renewed within 14 days.
- `exp/*` and `audit/*` branches are temporary. When their evidence is accepted, preserve the result with an annotated tag or release artifact, then delete the branch.

## Evidence Preservation

- Use annotated tags for historical experiment states and negative-result lineages.
- Use GitHub Releases or external artifact bundles for large evidence packages.
- Do not keep evidence-only branches alive as a substitute for release artifacts.
- Do not delete branches that are referenced by active server worktrees, active processes, or unmerged evidence.

## Cleanup Gates

Before deleting any remote branch:

1. Confirm the remote ref still exists.
2. Confirm no local or server worktree uses it.
3. Confirm no active process is running from that checkout.
4. Confirm important commits are reachable from `main`, an annotated tag, or a release artifact.
5. Record the action in the relevant hygiene report or PR comment.
